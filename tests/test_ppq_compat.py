"""Tests for onnxsim.ppq_compat -- the PPQ-API-shaped shim backed entirely
by onnxsim.quantize_static (see that module's docstring for why this
exists instead of trying to get the real, confirmed-broken PPQ running).

Unlike tests/test_ppq_integration.py (which can only test a graceful
failure, since real PPQ can't be imported here), this module needs no
optional dependency at all and its actual quantization behavior is fully
testable.
"""

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim import ppq_compat


def _matmul_model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[N,4] x) => (float[N,4] y)
        <float[4,4] w = {1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0}>
        {
            y = MatMul(x, w)
        }
        """
    )


def _two_input_model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[N,4] x, float[N,4] z) => (float[N,4] y)
        <float[4,4] w = {1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0}>
        {
            m = MatMul(x, w)
            y = Add(m, z)
        }
        """
    )


def _has_qdq(model) -> bool:
    op_types = {n.op_type for n in model.graph.node}
    return "QuantizeLinear" in op_types and "DequantizeLinear" in op_types


def test_quantize_onnx_model_with_dict_batches():
    rng = np.random.default_rng(0)
    loader = [{"x": rng.standard_normal((2, 4)).astype(np.float32)} for _ in range(4)]

    quantized = ppq_compat.quantize_onnx_model(
        onnx_import_file=_matmul_model(),
        calib_dataloader=loader,
        calib_steps=4,
        input_shape=None,
        platform=ppq_compat.TargetPlatform.ONNXRUNTIME,
    )
    onnx.checker.check_model(quantized)
    assert _has_qdq(quantized)


def test_quantize_onnx_model_with_raw_array_batches_single_input():
    """Real PPQ's own single-input convention: calib_dataloader yields raw
    arrays (no dict wrapper) when the model has exactly one input."""
    rng = np.random.default_rng(0)
    loader = [rng.standard_normal((2, 4)).astype(np.float32) for _ in range(4)]

    quantized = ppq_compat.quantize_onnx_model(
        onnx_import_file=_matmul_model(),
        calib_dataloader=loader,
        calib_steps=4,
    )
    onnx.checker.check_model(quantized)
    assert _has_qdq(quantized)


def test_quantize_onnx_model_with_tuple_batches_multi_input():
    rng = np.random.default_rng(0)
    loader = [
        (
            rng.standard_normal((2, 4)).astype(np.float32),
            rng.standard_normal((2, 4)).astype(np.float32),
        )
        for _ in range(4)
    ]

    quantized = ppq_compat.quantize_onnx_model(
        onnx_import_file=_two_input_model(),
        calib_dataloader=loader,
        calib_steps=4,
    )
    onnx.checker.check_model(quantized)


def test_raw_array_batch_with_multi_input_model_raises():
    rng = np.random.default_rng(0)
    loader = [rng.standard_normal((2, 4)).astype(np.float32) for _ in range(2)]

    with pytest.raises(ValueError, match="has 2 inputs"):
        ppq_compat.quantize_onnx_model(
            onnx_import_file=_two_input_model(),
            calib_dataloader=loader,
            calib_steps=2,
        )


def test_collate_fn_is_applied():
    rng = np.random.default_rng(0)
    raw_loader = [rng.standard_normal((2, 4)).astype(np.float32) for _ in range(3)]
    calls = []

    def collate(batch):
        calls.append(batch)
        return {"x": batch}

    quantized = ppq_compat.quantize_onnx_model(
        onnx_import_file=_matmul_model(),
        calib_dataloader=raw_loader,
        calib_steps=3,
        collate_fn=collate,
    )
    onnx.checker.check_model(quantized)
    assert len(calls) == 3


def test_calib_steps_limits_batches_consumed():
    rng = np.random.default_rng(0)
    consumed = []

    def loader():
        for _ in range(100):
            batch = rng.standard_normal((2, 4)).astype(np.float32)
            consumed.append(batch)
            yield {"x": batch}

    ppq_compat.quantize_onnx_model(
        onnx_import_file=_matmul_model(),
        calib_dataloader=loader(),
        calib_steps=5,
    )
    assert len(consumed) == 5


def test_unsupported_platform_raises():
    with pytest.raises(NotImplementedError, match="ONNXRUNTIME"):
        ppq_compat.quantize_onnx_model(
            onnx_import_file=_matmul_model(),
            calib_dataloader=[{"x": np.zeros((1, 4), dtype=np.float32)}],
            calib_steps=1,
            platform=999,
        )


def test_missing_calib_dataloader_raises_type_error():
    with pytest.raises(TypeError, match="calib_dataloader"):
        ppq_compat.quantize_onnx_model(onnx_import_file=_matmul_model())


def test_do_quantize_false_returns_model_unchanged():
    model = _matmul_model()
    result = ppq_compat.quantize_onnx_model(onnx_import_file=model, do_quantize=False)
    assert not _has_qdq(result)
    assert result is model


def test_setting_calib_algorithm_entropy_is_accepted():
    rng = np.random.default_rng(0)
    loader = [{"x": rng.standard_normal((2, 4)).astype(np.float32)} for _ in range(8)]
    setting = ppq_compat.QuantizationSettingFactory.default_setting()
    setting.calib_algorithm = "kl"

    quantized = ppq_compat.quantize_onnx_model(
        onnx_import_file=_matmul_model(),
        calib_dataloader=loader,
        calib_steps=8,
        setting=setting,
    )
    onnx.checker.check_model(quantized)
    assert _has_qdq(quantized)


def test_export_ppq_graph_appends_onnx_extension(tmp_path):
    model = _matmul_model()
    out_prefix = str(tmp_path / "quantized")

    ppq_compat.export_ppq_graph(
        model, ppq_compat.TargetPlatform.ONNXRUNTIME, out_prefix
    )

    reloaded = onnx.load(out_prefix + ".onnx")
    assert reloaded.graph.node[0].op_type == "MatMul"


def test_export_ppq_graph_unsupported_platform_raises(tmp_path):
    with pytest.raises(NotImplementedError, match="ONNXRUNTIME"):
        ppq_compat.export_ppq_graph(_matmul_model(), 999, str(tmp_path / "out"))


def test_target_platform_onnxruntime_matches_real_ppq_value():
    """Confirmed from PPQ 0.6.6's own ppq/core/quant.py."""
    assert int(ppq_compat.TargetPlatform.ONNXRUNTIME) == -7
