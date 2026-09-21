"""Write ``<name>.orig.onnx`` / ``<name>.sim.onnx`` pairs for ``trt_harness.py compare``.

Run under the onnxsim interpreter (Python >= 3.11). The models mirror the ones in
``tests/test_tensorrt_qdq_conventions.py`` (TensorRT's explicit-quantization Q/DQ
placement conventions) but at sizes real TensorRT INT8 kernels accept (channel
counts that are multiples of 32), so the *engine* -- not just the ONNX graph --
can be inspected to confirm simplification did not break Q/DQ fusion.

    python qdq_pairs.py OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper as nh
from onnx import parser

import onnxsim


def _model(body, inits):
    m = parser.parse_model(f'<ir_version: 8, opset_import: ["": 17]> {body}')
    m.graph.initializer.extend(nh.from_array(a, n) for n, a in inits)
    return m


def _conv_weights(rng, cout, cin):
    w = (rng.standard_normal((cout, cin, 3, 3)) * 0.3).astype(np.float32)
    scale = np.maximum(np.abs(w).reshape(cout, -1).max(1) / 127.0, 1e-6).astype(np.float32)
    return w, scale


def per_channel_conv():
    rng = np.random.default_rng(0)
    c = 32
    w, ws = _conv_weights(rng, c, c)
    return _model(
        f"""g (float[1,{c},32,32] X) => (float[1,{c},32,32] Y)
        <float a_scale = {{0.05}}, int8 a_zp = {{0}}> {{
          Xq = QuantizeLinear(X, a_scale, a_zp)
          Xdq = DequantizeLinear(Xq, a_scale, a_zp)
          Wq = QuantizeLinear<axis = 0>(W, w_scale, w_zp)
          Wdq = DequantizeLinear<axis = 0>(Wq, w_scale, w_zp)
          Y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(Xdq, Wdq)
        }}""",
        [("W", w), ("w_scale", ws), ("w_zp", np.zeros(c, np.int8))],
    )


def symmetric_matmul():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((64, 64)) * 0.2).astype(np.float32)
    return _model(
        """g (float[32,64] X) => (float[32,64] Y)
        <float a_scale = {0.03}, int8 a_zp = {0}, float w_scale = {0.01}, int8 w_zp = {0}> {
          Xq = QuantizeLinear(X, a_scale, a_zp)
          Xdq = DequantizeLinear(Xq, a_scale, a_zp)
          Wq = QuantizeLinear(W, w_scale, w_zp)
          Wdq = DequantizeLinear(Wq, w_scale, w_zp)
          Y = MatMul(Xdq, Wdq)
        }""",
        [("W", w)],
    )


def residual_branches():
    rng = np.random.default_rng(2)
    c = 32
    w1, _ = _conv_weights(rng, c, c)
    w2, _ = _conv_weights(rng, c, c)
    return _model(
        f"""g (float[1,{c},32,32] X) => (float[1,{c},32,32] Y)
        <float s = {{0.02}}, int8 zp = {{0}}> {{
          c1 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W1)
          q1 = QuantizeLinear(c1, s, zp)
          B1 = DequantizeLinear(q1, s, zp)
          c2 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W2)
          q2 = QuantizeLinear(c2, s, zp)
          B2 = DequantizeLinear(q2, s, zp)
          Y = Add(B1, B2)
        }}""",
        [("W1", w1), ("W2", w2)],
    )


def conv_bn_relu_no_qdq():
    """Control: plain fp32 net where simplify() folds BN into Conv."""
    rng = np.random.default_rng(3)
    c = 32
    w, _ = _conv_weights(rng, c, c)
    bn = [rng.uniform(0.5, 1.5, c), rng.standard_normal(c) * 0.1,
          rng.standard_normal(c) * 0.1, rng.uniform(0.5, 1.5, c)]
    return _model(
        f"""g (float[1,{c},32,32] X) => (float[1,{c},32,32] Y) {{
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
          b = BatchNormalization(c, gamma, beta, mean, var)
          Y = Relu(b)
        }}""",
        [("W", w)] + [(n, a.astype(np.float32)) for n, a in zip(["gamma", "beta", "mean", "var"], bn)],
    )


MODELS = {
    "per_channel_conv": per_channel_conv,
    "symmetric_matmul": symmetric_matmul,
    "residual_branches": residual_branches,
    "conv_bn_relu": conv_bn_relu_no_qdq,
}


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, make in MODELS.items():
        model = make()
        sim, ok = onnxsim.simplify(model)
        assert ok, f"{name}: onnxsim correctness check failed"
        # TensorRT 10.3's parser reads up to opset 21; simplify keeps our opset 17.
        onnx.save(model, out / f"{name}.orig.onnx")
        onnx.save(sim, out / f"{name}.sim.onnx")
        print(f"{name}: {len(model.graph.node)} -> {len(sim.graph.node)} nodes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trt_pairs")
