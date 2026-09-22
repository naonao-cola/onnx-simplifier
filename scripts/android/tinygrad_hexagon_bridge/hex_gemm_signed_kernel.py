#!/usr/bin/env python3
"""Signed-weight 1x1-conv GEMM kernel via vrmpybusv, for real backbone 1x1 convs (int8 signed
weights) -- NOT a modification of hex_gemm_kernel.py (which is uint8 x uint8 via vrmpyub only,
confirmed by reading its own kernel_fn/docstring: "Computes C[M,N] = A[M,K] @ B[K,N] (uint8 x
uint8 -> int32)"). The REAL Mask R-CNN backbone's 1x1 convs have SIGNED int8 weights (confirmed
from backbone.onnx: ConvMulFusion_W_*_quantized tensors are int8, not uint8) -- discovered while
building the ResNet-50 stage1/block1 chained subgraph (this project's first attempt to run real
extracted backbone weights, not synthetic uint8 test data, through the 1x1 conv path). This is a
real, previously-unflagged gap in this project's own 1x1-conv coverage: every "1x1 conv" speedup
number reported elsewhere in this project (hex_gemm_kernel.py's own coverage tables) was measured
against synthetic UNSIGNED weights, not the real network's signed ones.

Structurally identical to hex_gemm_kernel.py's build_kernel() (same _reg_i32 multi-range
accumulator-init pattern, same M/N-tile/K-chunk loop nest, same pack_b()-compatible weight
layout), with exactly one change: vrmpyub_acc_128B(acc, weight_vec, a_scalar) (unsigned x
unsigned, activation as a plain scalar 4-byte read) is replaced with vrmpybusv_acc_128B(acc,
broadcast32(a_scalar), weight_vec) (unsigned x signed, activation must be broadcast to a full
32-wide vector first -- a different calling convention, matching the exact pattern
hex_conv3x3_kernel.py/hex_stem7x7_kernel.py already use for their own signed-weight vrmpybusv
calls, NOT the plain-scalar convention vrmpyub uses). Operand order also swaps: vrmpyub reads
(acc, weight, activation-scalar); vrmpybusv reads (acc, activation-broadcast, weight-vector).

pack_b()'s exact layout (weight packed as Bp[nt, kc, n_lane*4+ks] = B[kc*4+ks, nt*32+n_lane]) is
reused unchanged from hex_gemm_kernel.py -- packing is pure data layout, independent of
signedness; only the accumulate instruction and its calling convention change.
"""
from __future__ import annotations

import functools


def build_kernel(cin: int, cout: int, m: int, a, bp, kernel_name: str = "hex_gemm_signed"):
    """Same contract as hex_gemm_kernel.py's build_kernel(): a (m,cin) uint8 activation Tensor,
    bp (cout//32, cin//4, 128) uint8 Tensor pre-packed via hex_gemm_kernel.pack_b() from a SIGNED
    int8 weight matrix (viewed as uint8 bytes -- the bit pattern is what matters, vrmpybusv
    interprets the vector operand as signed regardless of the C array's declared type). Returns
    the (m, cout) int32 output Tensor."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert cin % 4 == 0 and cout % 32 == 0
    nt_count, kc_count = cout // 32, cin // 4
    i32x32 = "int __attribute__((vector_size(128)))"
    u8x128 = "unsigned char __attribute__((vector_size(128)))"
    broadcast32 = ",".join(["*(unsigned int*){2}"] * 32)

    def _reg_i32(shape, slot, *deps):
        ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp, Bp: UOp) -> UOp:
        m_rng = UOp.range(m, 0, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 1, AxisType.WEAK)
        acc = _reg_i32((32,), 0, m_rng, nt_rng)
        kc_rng = UOp.range(kc_count, 2, AxisType.REDUCE)
        acc_addr = acc.after(kc_rng)[0]
        a_idx = A[m_rng, kc_rng * 4]
        b_idx = Bp[nt_rng, kc_rng, 0]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (acc_addr, b_idx, a_idx),
            arg=(f"*({i32x32}*){{0}} = __builtin_HEXAGON_V6_vrmpybusv_acc_128B("
                 f"*({i32x32}*){{0}}, ({i32x32}){{{{{broadcast32}}}}}, *({u8x128}*){{1}});"),
        )
        update = step.end(kc_rng)
        final_addr = acc.after(update)[0]
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (C[m_rng, nt_rng * 32], final_addr),
            arg=f"*({i32x32}*){{0}} = *({i32x32}*){{1}};",
        )
        return out_step.end(nt_rng, m_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(m, cout, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a, bp, fxn=functools.partial(kernel_fn))[0]
