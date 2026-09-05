"""Real-hardware follow-up to the SmolLM2-135M accuracy investigation
(see scripts/axera/README.md's "Mitigation attempts: none reproduce
llm_build()'s accuracy via the generic path"): locks in a real,
reproducible Pulsar2-internal bug found while trying `model_type:
"QuantONNX"` as a way to feed onnxsim's own weight-only/static quantizers
(`onnxsim.quantize_weight_only()`, `onnxsim.quantize_static()`) to
`pulsar2 build` instead of letting Pulsar2's own PTQ run.

`QuantONNX` is a real, distinct ingestion path (confirmed: `pulsar2 build`
prints `"... is a QuantONNX model, disable concat align config"` and skips
requesting calibration ranges for tensors that already carry
QuantizeLinear/DequantizeLinear) -- but any `MatMul` whose weight input
comes through a `DequantizeLinear` (the standard ONNX QDQ per-channel
weight-quantization pattern) crashes Pulsar2's own PPQ-based
`ax_quant_graph_optimize` pass: one of MatMul's two inputs is silently
dropped, and its executor then raises `ValueError: Can not feed value to
operation <node>, expects exact 2 inputs, however 1 was given`. This file
locks in the minimal, 2-node repro of that -- no LLM/onnxsim reconstruction
involved at all, confirming it's a general `QuantONNX` + quantized-`MatMul`
limitation in Pulsar2 itself.

Needs a loaded `pulsar2:*` Docker image -- skip-guarded like
tests/test_pulsar2_hf_to_axmodel.py.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(),
    reason=f"pulsar2 Docker image not loaded: {pulsar2_docker.DEFAULT_IMAGE}",
)


def _weight_only_quantized_matmul_model() -> onnx.ModelProto:
    """`y = MatMul(x, DequantizeLinear(wq, scale, zp))` -- the minimal shape
    of what both `onnxsim.quantize_weight_only()` and
    `onnxsim.quantize_static()` emit around a MatMul's weight."""
    rng = np.random.RandomState(0)
    w = rng.randn(8, 8).astype(np.float32)
    wq = np.clip(np.round(w / 0.01), -127, 127).astype(np.int8)
    scale = np.full((8,), 0.01, dtype=np.float32)
    zp = np.zeros((8,), dtype=np.int8)

    dq = helper.make_node("DequantizeLinear", ["wq", "scale", "zp"], ["w_dq"], axis=1)
    mm = helper.make_node("MatMul", ["x", "w_dq"], ["y"])
    graph = helper.make_graph(
        [dq, mm],
        "g_minimal_wo_quant",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 8])],
        initializer=[
            numpy_helper.from_array(wq, name="wq"),
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(zp, name="zp"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_quantonnx_matmul_with_dequantizelinear_weight_crashes(tmp_path):
    """Confirmed real: `pulsar2 build --model_type QuantONNX` recognizes the
    already-quantized graph (prints "is a QuantONNX model") but its own
    internal PPQ graph-optimization pass crashes on a quantized-weight
    MatMul specifically -- a real Pulsar2 bug, not something fixable from
    the ONNX side. This is why feeding onnxsim's own weight-only/static
    quantizer output as QuantONNX can't currently substitute for Pulsar2's
    own (accuracy-losing, for deep transformers) PTQ."""
    model = _weight_only_quantized_matmul_model()
    onnx.checker.check_model(model)

    work_dir = tmp_path / "quantonnx_minimal"
    work_dir.mkdir()
    onnx.save(model, str(work_dir / "model.onnx"))

    rng = np.random.RandomState(0)
    (work_dir / "dataset").mkdir()
    samples = [rng.randn(1, 8).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        str(work_dir / "dataset" / "x.tar"), samples
    )

    cfg = {
        "model_type": "QuantONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    (work_dir / "config").mkdir()
    with open(work_dir / "config" / "cfg.json", "w") as f:
        json.dump(cfg, f)

    result = pulsar2_docker.build(
        str(work_dir), "model.onnx", "output", config_path="config/cfg.json"
    )
    assert not result.success
    assert "expects exact 2 inputs" in result.error
