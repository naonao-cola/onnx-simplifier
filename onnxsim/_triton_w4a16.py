"""Triton W4A16 kernels for ``com.microsoft::MatMulNBits`` weights (used by ``to_torch``).

Weights stay in MatMulNBits' own layout: ``packed`` ``[N, KB2]`` uint8 (two 4-bit values per
byte along K, low nibble first, K padded to whole groups), ``scales`` ``[N, NG]``, optional
unpacked ``zeros`` ``[N, NG]`` uint8 (default 8). ``y = x @ ((q - zp) * s)^T`` with the
dequantized weight rounded to ``x``'s dtype before the fp32-accumulated dot -- the same
arithmetic as dequantizing once and calling ``F.linear``.

Why a kernel of our own: TensorRT-LLM's weight-only int4 GEMMs
(``finegrained_mixed_dtype_gemm``, ``weight_only_quant_gemm``) refuse sm_120 ("SM120 GEMM
only supports nvfp4"), AutoDeploy's int4 ops are fake-quant (dequantize the whole weight in
PyTorch on every call), and PyTorch's ``_weight_int4pack_mm`` takes bf16 activations only.

All shapes on an RTX 5050, with weights streamed from DRAM as in a real model (not
L2-resident): decode (M <= 16) runs a split-K kernel that dequantizes in registers,
1.1-2.5x faster than dense fp16; M in (16, 256] runs the same kernel as a plain tile GEMM
(no split), which beats both a transient-dequant + cuBLAS path at every M <= 128 swept and
dense fp16 at M = 32 (1.35-1.64x); longer prefills dequantize to a transient fp16 weight
and call cuBLAS.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_kernel(
    x_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    y_ptr,
    M,
    N,
    K,
    KB2,
    NG,
    stride_xm,
    stride_ym,
    k_per_split,
    HAS_Z: tl.constexpr,
    SPLIT: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_lo = pid_k * k_per_split
    for k0 in range(k_lo, k_lo + k_per_split, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        offs_kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        b = tl.load(
            b_ptr + offs_n[:, None] * KB2 + offs_kb[None, :],
            mask=n_mask[:, None] & (offs_kb[None, :] < KB2),
            other=0,
        )
        q = tl.join(b & 0xF, b >> 4).reshape(BLOCK_N, BLOCK_K // GROUP, GROUP)
        # one scale (and zero) per group, broadcast over the group's GROUP weights
        offs_g = k0 // GROUP + tl.arange(0, BLOCK_K // GROUP)
        g_mask = n_mask[:, None] & (offs_g[None, :] < NG)
        s = tl.load(
            s_ptr + offs_n[:, None] * NG + offs_g[None, :], mask=g_mask, other=0.0
        )
        if HAS_Z:
            z = tl.load(
                z_ptr + offs_n[:, None] * NG + offs_g[None, :], mask=g_mask, other=8
            )
            w = (q.to(tl.float32) - z.to(tl.float32)[:, :, None]) * s.to(tl.float32)[
                :, :, None
            ]
        else:
            w = (q.to(tl.float32) - 8.0) * s.to(tl.float32)[:, :, None]
        w = w.reshape(BLOCK_N, BLOCK_K)
        w = tl.where(n_mask[:, None] & k_mask[None, :], w, 0.0).to(x.dtype)
        acc += tl.dot(x, tl.trans(w))
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)
    else:
        tl.atomic_add(y_ptrs, acc, mask=out_mask)


@triton.jit
def _w4_dequant_kernel(
    b_ptr,
    s_ptr,
    z_ptr,
    w_ptr,
    N,
    K,
    KB2,
    NG,
    HAS_Z: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = tl.program_id(1) * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)
    offs_kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
    n_mask = offs_n < N
    b = tl.load(
        b_ptr + offs_n[:, None] * KB2 + offs_kb[None, :],
        mask=n_mask[:, None] & (offs_kb[None, :] < KB2),
        other=0,
    )
    q = tl.join(b & 0xF, b >> 4).reshape(BLOCK_N, BLOCK_K // GROUP, GROUP)
    offs_g = k0 // GROUP + tl.arange(0, BLOCK_K // GROUP)
    g_mask = n_mask[:, None] & (offs_g[None, :] < NG)
    s = tl.load(s_ptr + offs_n[:, None] * NG + offs_g[None, :], mask=g_mask, other=0.0)
    if HAS_Z:
        z = tl.load(
            z_ptr + offs_n[:, None] * NG + offs_g[None, :], mask=g_mask, other=8
        )
        w = (q.to(tl.float32) - z.to(tl.float32)[:, :, None]) * s.to(tl.float32)[
            :, :, None
        ]
    else:
        w = (q.to(tl.float32) - 8.0) * s.to(tl.float32)[:, :, None]
    w = w.reshape(BLOCK_N, BLOCK_K)
    tl.store(
        w_ptr + offs_n[:, None] * K + offs_k[None, :],
        w.to(w_ptr.dtype.element_ty),
        mask=n_mask[:, None] & (offs_k[None, :] < K),
    )


def w4_dequant(packed, scales, zeros, k, n, group, dtype):
    """Dense ``[N, K]`` weight, dequantized on the GPU."""
    w = torch.empty((n, k), device=packed.device, dtype=dtype)
    bn, bk = 32, max(128, group)
    _w4_dequant_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk))](
        packed,
        scales,
        zeros if zeros is not None else scales,
        w,
        n,
        k,
        packed.shape[1],
        scales.shape[1],
        HAS_Z=zeros is not None,
        GROUP=group,
        BLOCK_N=bn,
        BLOCK_K=bk,
    )
    return w


_NUM_SMS: dict = {}


def w4a16_linear(x, packed, scales, zeros, k, n, group, bias=None):
    """``F.linear(x, dequant(packed, scales, zeros))``; the weight is only materialized
    (transiently) for M > 256."""
    if 128 % group and group % 128:
        raise ValueError(f"group size {group} must divide or be a multiple of 128")
    shape = x.shape
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if m > 256:
        # long prefill: compute-bound; cuBLAS on a transient dequantized weight is
        # within ~10% of the tile kernel either way (RTX 5050 sweep)
        w = w4_dequant(packed, scales, zeros, k, n, group, x.dtype)
        return torch.nn.functional.linear(x, w, bias)
    if m > 16:
        # short prefill / batched decode: still memory-bound on the weight, so the
        # tile kernel (dequantize in registers, no fp16 weight write) wins -- 1.4-1.6x
        # faster than dense fp16 at M=32 on 2048x8192, and faster than dequant+cuBLAS
        # at every M <= 128 swept
        bm = 32 if m <= 32 else 64
        bk = max(128 if m <= 32 else 64, group)
        y = torch.empty((m, n), device=x.device, dtype=x.dtype)
        _w4a16_kernel[(triton.cdiv(m, bm), triton.cdiv(n, 64), 1)](
            x2,
            packed,
            scales,
            zeros if zeros is not None else scales,
            y,
            m,
            n,
            k,
            packed.shape[1],
            scales.shape[1],
            x2.stride(0),
            y.stride(0),
            triton.cdiv(k, bk) * bk,
            HAS_Z=zeros is not None,
            SPLIT=1,
            GROUP=group,
            BLOCK_M=bm,
            BLOCK_N=64,
            BLOCK_K=bk,
            num_warps=4,
            num_stages=3,
        )
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], n)
    dev = x.device.index or 0
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(x.device).multi_processor_count
    bm, bn, bk = 16, (16 if n <= 1024 else 32), max(128, group)
    grid_mn = triton.cdiv(n, bn)
    k_steps = triton.cdiv(k, bk)
    split = 1  # split K until ~4 CTAs per SM (batch-1 decode has few N tiles)
    while grid_mn * split < 4 * _NUM_SMS[dev] and split * 2 <= k_steps:
        split *= 2
    k_per_split = triton.cdiv(k_steps, split) * bk
    split = triton.cdiv(k, k_per_split)
    if split == 1:
        y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    else:
        y = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    _w4a16_kernel[(triton.cdiv(m, bm), grid_mn, split)](
        x2,
        packed,
        scales,
        zeros if zeros is not None else scales,
        y,
        m,
        n,
        k,
        packed.shape[1],
        scales.shape[1],
        x2.stride(0),
        y.stride(0),
        k_per_split,
        HAS_Z=zeros is not None,
        SPLIT=split,
        GROUP=group,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=3,
    )
    y = y.to(x.dtype)
    if bias is not None:
        y = y + bias
    return y.reshape(*shape[:-1], n)
