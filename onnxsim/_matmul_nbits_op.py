"""``torch.ops.onnxsim.matmul_nbits``: int4 weight-only linear on packed MatMulNBits weights.

Emitted by ``onnxsim.to_torch.onnx_to_torch(..., matmul_nbits="packed")``. Weights keep
``com.microsoft::MatMulNBits``' layout (``packed`` ``[N, KB2]`` uint8, ``scales`` ``[N, NG]``,
optional unpacked ``zeros`` ``[N, NG]`` uint8), so a converted int4 model stays int4 in
memory. On CUDA with Triton available it runs ``onnxsim._triton_w4a16``'s kernels;
elsewhere it dequantizes in PyTorch and calls ``F.linear`` -- the same arithmetic either
way. Registered as a ``torch.library`` custom op with a fake implementation, so
``torch.export`` (and TensorRT-LLM AutoDeploy, which exports) sees one opaque node.

Kept out of ``to_torch.py``: ``torch.library.custom_op`` infers the schema from real type
annotations, which that module's ``from __future__ import annotations`` would stringify.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def dequant_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor],
    k: int,
    n: int,
    group: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dense ``[N, K]`` weight from MatMulNBits-layout int4 tensors, in PyTorch."""
    q = torch.stack([packed & 0xF, packed >> 4], -1).reshape(n, -1, group)
    z = 8.0 if zeros is None else zeros.to(torch.float32)[:, :, None]
    w = (q.to(torch.float32) - z) * scales.to(torch.float32)[:, :, None]
    return w.reshape(n, -1)[:, :k].to(dtype)


@torch.library.custom_op("onnxsim::matmul_nbits", mutates_args=())
def matmul_nbits(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor],
    k: int,
    n: int,
    group: int,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if x.is_cuda:
        try:
            from onnxsim._triton_w4a16 import w4a16_linear
        except ImportError:  # no Triton: fall through to the PyTorch path
            pass
        else:
            if not (128 % group and group % 128):
                return w4a16_linear(x, packed, scales, zeros, k, n, group, bias)
    return F.linear(
        x, dequant_packed(packed, scales, zeros, k, n, group, x.dtype), bias
    )


@matmul_nbits.register_fake
def _(x, packed, scales, zeros, k, n, group, bias):
    return x.new_empty((*x.shape[:-1], n))
