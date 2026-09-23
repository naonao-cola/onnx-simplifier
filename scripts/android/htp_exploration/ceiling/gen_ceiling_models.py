#!/usr/bin/env python3
"""Generate QDQ ONNX models for measuring the HTP's practical int8 throughput ceiling.

Each model is a chain of L identical layers in the backbone's own QDQ convention (uint8
activations, int8 weights, int32 bias, per-tensor scales unless noted):

    x(fp32) -> Q -> DQ -> [ Conv/MatMul(DQ(W int8), DQ(b int32)) -> Q -> DQ ] * L -> y(fp32)

Two chain lengths per shape (L=LO and L=HI) let run_ceiling.sh difference away everything that
isn't per-layer work (graph launch, fp32 input quantize / output dequantize, host copies):
    per_layer_ms = (t(HI) - t(LO)) / (HI - LO),   TMAC/s = MACs_per_layer / per_layer_ms / 1e9

Variants: default (u8 act / s8 weight, per-tensor), 'pc' (per-channel weight scales),
'a16' (uint16 activations, the other activation width QNN HTP supports).

usage: gen_ceiling_models.py <outdir>   (writes *.onnx + manifest.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LO, HI = 2, 6


def _q(name, x, scale, zp):
    return helper.make_node("QuantizeLinear", [x, scale, zp], [name])


def _dq(name, x, scale, zp, axis=None):
    kw = {} if axis is None else {"axis": axis}
    return helper.make_node("DequantizeLinear", [x, scale, zp], [name], **kw)


def build(kind, cin, cout, hw, layers, variant):
    """kind: conv1x1 | conv3x3 | matmul.  For matmul, hw is M and cin=K, cout=N."""
    rng = np.random.default_rng(0)
    a16 = variant == "a16"
    act_t = TensorProto.UINT16 if a16 else TensorProto.UINT8
    act_zp = 32768 if a16 else 128
    inits, nodes = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(arr, name))
        return name

    const("a_s", np.array(0.05, np.float32))
    inits.append(helper.make_tensor("a_z", act_t, [], [act_zp]))
    if kind == "matmul":
        in_shape = [hw, cin]
    else:
        in_shape = [1, cin, hw, hw]
    nodes.append(_q("x_q", "x", "a_s", "a_z"))
    nodes.append(_dq("x_dq", "x_q", "a_s", "a_z"))
    cur = "x_dq"
    k = 3 if kind == "conv3x3" else 1
    for i in range(layers):
        if kind == "matmul":
            w = rng.integers(-127, 128, (cin, cout), dtype=np.int8)
            waxis = 1
        else:
            w = rng.integers(-127, 128, (cout, cin, k, k), dtype=np.int8)
            waxis = 0
        const(f"w{i}", w)
        if variant == "pc":
            const(f"w{i}_s", np.full((cout,), 0.002, np.float32))
            const(f"w{i}_z", np.zeros((cout,), np.int8))
            nodes.append(_dq(f"w{i}_dq", f"w{i}", f"w{i}_s", f"w{i}_z", axis=waxis))
        else:
            const(f"w{i}_s", np.array(0.002, np.float32))
            const(f"w{i}_z", np.array(0, np.int8))
            nodes.append(_dq(f"w{i}_dq", f"w{i}", f"w{i}_s", f"w{i}_z"))
        if kind == "matmul":
            nodes.append(helper.make_node("MatMul", [cur, f"w{i}_dq"], [f"y{i}"]))
        else:
            b = rng.integers(-1000, 1000, (cout,), dtype=np.int32)
            const(f"b{i}", b)
            bs = 0.05 * 0.002
            const(f"b{i}_s", np.full((cout,), bs, np.float32) if variant == "pc" else np.array(bs, np.float32))
            const(f"b{i}_z", np.zeros((cout,), np.int32) if variant == "pc" else np.array(0, np.int32))
            nodes.append(_dq(f"b{i}_dq", f"b{i}", f"b{i}_s", f"b{i}_z", axis=0 if variant == "pc" else None))
            nodes.append(helper.make_node("Conv", [cur, f"w{i}_dq", f"b{i}_dq"], [f"y{i}"],
                                          kernel_shape=[k, k], pads=[k // 2] * 4))
        nodes.append(_q(f"y{i}_q", f"y{i}", "a_s", "a_z"))
        nodes.append(_dq(f"y{i}_dq", f"y{i}_q", "a_s", "a_z"))
        cur = f"y{i}_dq"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    out_shape = [hw, cout] if kind == "matmul" else [1, cout, hw, hw]
    g = helper.make_graph(nodes, "ceiling",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)], inits)
    # opset 21: 16-bit Q/DQ are standard ops there (needed for the a16 variant)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m)
    macs = (hw * cin * cout) if kind == "matmul" else (hw * hw * cin * cout * k * k)
    return m, macs


# (kind, cin, cout, hw, variant). hw = spatial side for convs, M for matmul.
SWEEP = [
    ("conv1x1", 256, 256, 64, "u8"),
    ("conv1x1", 512, 512, 64, "u8"),
    ("conv1x1", 1024, 1024, 64, "u8"),
    ("conv1x1", 2048, 2048, 64, "u8"),
    ("conv1x1", 1024, 1024, 128, "u8"),
    ("conv1x1", 2048, 2048, 128, "u8"),
    ("conv3x3", 256, 256, 128, "u8"),
    ("conv3x3", 512, 512, 64, "u8"),
    ("matmul", 4096, 4096, 4096, "u8"),
    ("conv1x1", 1024, 1024, 64, "pc"),
    ("conv3x3", 256, 256, 128, "pc"),
    ("conv1x1", 1024, 1024, 64, "a16"),
    ("conv3x3", 256, 256, 128, "a16"),
]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for kind, cin, cout, hw, variant in SWEEP:
        tag = f"{kind}_{cin}x{cout}_{hw}_{variant}"
        entry = {"tag": tag, "kind": kind, "cin": cin, "cout": cout, "hw": hw, "variant": variant,
                 "lo": LO, "hi": HI}
        for L in (LO, HI):
            m, macs = build(kind, cin, cout, hw, L, variant)
            p = out / f"{tag}_L{L}.onnx"
            onnx.save(m, p)
            entry["macs_per_layer"] = macs
        manifest.append(entry)
        print(tag, f"{entry['macs_per_layer'] / 1e9:.2f} GMAC/layer")
    # a near-empty model: Q -> DQ only, to measure fixed per-inference overhead
    m, _ = build("conv1x1", 32, 32, 8, 0, "u8")
    onnx.save(m, out / "tiny_L0.onnx")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
