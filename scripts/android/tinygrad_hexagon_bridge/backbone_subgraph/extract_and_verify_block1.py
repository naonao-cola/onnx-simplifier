#!/usr/bin/env python3
"""Extract real weights/biases/scales for ResNet-50 stage1/block1 -- the first bottleneck block
after Stage 2's stem+maxpool subgraph (`conv10(1x1 reduce)->requantize->conv17(3x3)->requantize->
conv24(1x1 expand)`, plus a `conv30(1x1 downsample)` shortcut, merged via a rescale-add-relu-
requantize step) -- from the real `backbone.onnx` (produced by
`scripts/android/maskrcnn_e2e/prepare.py`), write the C weight/bias constants
`native_transport/`'s fused driver embeds, generate real test input/reference-output data, and
verify the whole block numerically against a real ONNX Runtime reference sliced from the same
graph. See ../README.md's "Chaining a real subgraph" section (block1 follow-up) for the real
composition gaps this found -- most importantly, that the residual add's two branches do NOT
share a common raw-accumulator scale (unlike the earlier FPN-lateral `add` finding) and need an
explicit rescale before adding.

Requires a prepared `backbone.onnx` + `input.npy` (real image) -- run
`scripts/android/maskrcnn_e2e/prepare.py --workdir <dir>` first, then point `--workdir` here at
the same directory. Also requires `hex_gemm_signed_kernel.py`'s kernel bodies (conv10/conv24/
conv30) and `hex_conv3x3_kernel.py`'s (conv17) to already be captured into
`native_transport/block1_kernels.c` -- this script only handles weight/bias/test-data extraction,
not kernel-C capture (see this project's other `gen_*_kernel.py` files' own `main()`s for that
pattern; `block1_kernels.c`'s header comment documents which kernel-generation calls produced it).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hex_gemm_kernel as gemm  # noqa: E402
import hex_conv3x3_kernel as conv3x3  # noqa: E402


def compute_multiplier_shift(double_multiplier: float) -> tuple[int, int]:
    """Same as hex_requantize_kernel.py's own (duplicated here to keep this script standalone)."""
    if double_multiplier == 0.0:
        return 0, 0
    significand, exponent = math.frexp(double_multiplier)
    significand_int = round(significand * (1 << 31))
    if significand_int == (1 << 31):
        significand_int //= 2
        exponent += 1
    return significand_int, exponent


def rescale_noclamp(x_int32: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    """Same Q31 fixed-point math as hex_requantize_kernel.py's requantize_ref, WITHOUT the final
    clip(0,255)/uint8 cast -- brings a raw int32 accumulator into a shared scale domain (still
    int32, may be negative) so two differently-scaled branches can be added correctly before the
    one real final requantize. See this file's own docstring for why this is necessary here."""
    tensor = x_int32.astype(np.int64)
    left_shift = max(shift, 0)
    right_shift = max(-shift, 0)
    prod = (tensor << left_shift) * np.int64(multiplier)
    total_shift = right_shift + 31
    rounding = np.int64(1) << (total_shift - 1)
    return (prod + rounding) >> np.int64(total_shift)


def conv2d_int(x_u8_nchw: np.ndarray, w_i8: np.ndarray, bias_i32: np.ndarray, pad: int, stride: int) -> np.ndarray:
    """Plain int conv (numpy), x: (N,Cin,H,W) uint8 (zero_point=0, confirmed for every
    activation in this block), w: (Cout,Cin,Kh,Kw) int8."""
    n, cin, h, w_ = x_u8_nchw.shape
    cout, _, kh, kw = w_i8.shape
    xp = np.pad(x_u8_nchw, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=0).astype(np.int64)
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w_ + 2 * pad - kw) // stride + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.int64)
    wf = w_i8.astype(np.int64)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[:, :, ky:ky + oh * stride:stride, kx:kx + ow * stride:stride]
            out += np.einsum("oc,nchw->nohw", wf[:, :, ky, kx], patch)
    return out + bias_i32.astype(np.int64).reshape(1, cout, 1, 1)


def c_array(name: str, arr: np.ndarray, ctype: str) -> str:
    flat = arr.astype(np.int64 if ctype == "int" else np.uint8).ravel()
    body = ",".join(str(int(x)) for x in flat)
    return f"static const {ctype} {name}[{flat.size}] __attribute__((aligned(128))) = {{{body}}};\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True, help="prepare.py's --workdir (has backbone.onnx, input.npy)")
    args = p.parse_args()

    workdir = Path(args.workdir)
    nt_dir = Path(__file__).resolve().parent.parent / "native_transport"
    m = onnx.load(str(workdir / "backbone.onnx"))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    def w(name):
        return inits[name]

    w10, b10, ws10 = w("ConvMulFusion_W_10_quantized").astype(np.int8), w("ConvAddFusion_Add_B_13_quantized").astype(np.int32), float(w("ConvMulFusion_W_10_scale"))
    w17, b17, ws17 = w("ConvMulFusion_W_17_quantized").astype(np.int8), w("ConvAddFusion_Add_B_20_quantized").astype(np.int32), float(w("ConvMulFusion_W_17_scale"))
    w24, b24, ws24 = w("ConvMulFusion_W_24_quantized").astype(np.int8), w("ConvAddFusion_Add_B_27_quantized").astype(np.int32), float(w("ConvMulFusion_W_24_scale"))
    w30, b30, ws30 = w("ConvMulFusion_W_30_quantized").astype(np.int8), w("ConvAddFusion_Add_B_33_quantized").astype(np.int32), float(w("ConvMulFusion_W_30_scale"))
    s_stem, s15, s22, s37 = (float(w(n)) for n in ("7_scale", "15_scale", "22_scale", "37_scale"))

    # ---- weight consts (packed for the on-device kernels) ----
    w10_packed = gemm.pack_b(w10.reshape(64, 64).T.astype(np.uint8))
    w24_packed = gemm.pack_b(w24.reshape(256, 64).T.astype(np.uint8))
    w30_packed = gemm.pack_b(w30.reshape(256, 64).T.astype(np.uint8))
    w17_packed = conv3x3.pack_weight_3x3(w17)

    consts = "".join([
        c_array("g_stem_conv10_weight_packed", w10_packed, "unsigned char"),
        c_array("g_conv10_bias", b10, "int"),
        c_array("g_conv17_weight_packed", w17_packed, "unsigned char"),
        c_array("g_conv17_bias", b17, "int"),
        c_array("g_conv24_weight_packed", w24_packed, "unsigned char"),
        c_array("g_conv24_bias", b24, "int"),
        c_array("g_conv30_weight_packed", w30_packed, "unsigned char"),
        c_array("g_conv30_bias", b30, "int"),
    ])
    (nt_dir / "gen_block1_weight_consts.c").write_text(consts)
    print(f"wrote gen_block1_weight_consts.c ({len(consts)} bytes)")

    # ---- real multiplier/shift constants (also baked into block1_kernels.c / block1_glue.c) ----
    mul10, sh10 = compute_multiplier_shift(s_stem * ws10 / s15)
    mul17, sh17 = compute_multiplier_shift(s15 * ws17 / s22)
    s_main_raw, s_shortcut_raw = s22 * ws24, s_stem * ws30
    mul_m, sh_m = compute_multiplier_shift(s_main_raw / s37)
    mul_s, sh_s = compute_multiplier_shift(s_shortcut_raw / s37)
    print(f"requantize x15: multiplier={mul10} shift={sh10}")
    print(f"requantize x22: multiplier={mul17} shift={sh17}")
    print(f"rescale main(conv24)->s37: multiplier={mul_m} shift={sh_m}")
    print(f"rescale shortcut(conv30)->s37: multiplier={mul_s} shift={sh_s}")

    # ---- real test input (Stage 2's maxpool output, packed NCHWc) + ORT reference ----
    extract_out = nt_dir / "extract_block1.onnx"
    onnx.utils.extract_model(str(workdir / "backbone.onnx"), str(extract_out),
                              input_names=["image"], output_names=["9_quantized", "37_38_QuantizeLinear"])
    sess = ort.InferenceSession(str(extract_out), providers=["CPUExecutionProvider"])
    image = np.load(str(workdir / "input.npy")).astype(np.float32)
    maxpool_out, ref_out = sess.run(["9_quantized", "37_38_QuantizeLinear"], {"image": image})
    x0 = maxpool_out  # (1,64,200,272) uint8, zp=0
    ref_out = ref_out[0]  # (256,200,272) uint8

    packed_in = maxpool_out[0].reshape(2, 32, 200, 272).transpose(0, 2, 3, 1).copy()  # (2,200,272,32)
    packed_in.tofile(nt_dir / "block1_input_packed.bin")
    ref_out.transpose(1, 2, 0).reshape(-1).tofile(nt_dir / "block1_ref_output.bin")
    print(f"wrote block1_input_packed.bin ({packed_in.nbytes}B), block1_ref_output.bin ({ref_out.nbytes}B)")

    # ---- numpy-level verification against the real ORT reference, before any C is trusted ----
    raw10 = conv2d_int(x0, w10, b10, pad=0, stride=1)
    x15 = np.clip(rescale_noclamp(raw10, mul10, sh10), 0, 255).astype(np.uint8)
    raw17 = conv2d_int(x15, w17, b17, pad=1, stride=1)
    x22 = np.clip(rescale_noclamp(raw17, mul17, sh17), 0, 255).astype(np.uint8)
    raw24 = conv2d_int(x22, w24, b24, pad=0, stride=1)
    raw30 = conv2d_int(x0, w30, b30, pad=0, stride=1)
    summed = rescale_noclamp(raw24, mul_m, sh_m) + rescale_noclamp(raw30, mul_s, sh_s)
    my_out = np.clip(np.maximum(summed, 0), 0, 255).astype(np.uint8)[0].transpose(1, 2, 0)  # (200,272,256) NHWC

    diff = np.abs(my_out.astype(np.int64).reshape(-1) - ref_out.transpose(1, 2, 0).astype(np.int64).reshape(-1))
    print(f"numpy-vs-ORT: max abs diff {diff.max()}, mean abs diff {diff.mean():.6f}, "
          f"exact match {(diff == 0).mean() * 100:.4f}%")


if __name__ == "__main__":
    main()
