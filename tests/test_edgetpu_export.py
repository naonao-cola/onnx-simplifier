"""Tests for Coral Edge TPU support (``onnxsim/edgetpu_export.py``).

Covers the parts that need no hardware and no heavy dependencies (the ONNX
compatibility check, calibration-data generation, compiler-log parsing with a
stubbed compiler), plus end-to-end paths gated on their optional dependencies
-- TensorFlow for quantization, LiteRT for ``.tflite`` inspection/inference,
and the ``edgetpu_compiler`` binary for real compilation -- following the same
``importorskip`` pattern as ``tests/test_tflite_export.py``.
"""

import os
import subprocess

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import edgetpu_export
from onnxsim.edgetpu_export import (
    EdgeTPUCompileResult,
    check_onnx_for_edgetpu,
    compile_for_edgetpu,
)


def _model(
    body: str, initializer=(), opset: int = 17, ir_version: int = 8
) -> onnx.ModelProto:
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _relu_model() -> onnx.ModelProto:
    model = _model(
        """
        relu (float[2,3] x) => (float[2,3] y)
        {
            y = Relu (x)
        }
        """
    )
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# ONNX compatibility check (onnx only)
# ---------------------------------------------------------------------------


def test_supported_model_is_fully_supported():
    rng = np.random.RandomState(0)
    w = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), name="w")
    b = numpy_helper.from_array(np.zeros(4, np.float32), name="b")
    model = _model(
        """
        cnn (float[1,3,8,8] x) => (float[1,4] y)
        {
            c = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w, b)
            r = Relu (c)
            g = GlobalAveragePool (r)
            f = Flatten <axis=1> (g)
            y = Softmax <axis=-1> (f)
        }
        """,
        initializer=[w, b],
    )
    report = check_onnx_for_edgetpu(model)
    assert report.errors == []
    assert report.fully_supported
    assert "0 error(s)" in report.summary()


def test_unsupported_ops_warn_naming_the_op():
    model = _model(
        """
        mixed (float[1,8] x) => (float[1,8] y)
        {
            g = Gelu (x)
            y = LeakyRelu <alpha=0.1> (g)
        }
        """,
        opset=20,
    )
    report = check_onnx_for_edgetpu(model)
    assert report.errors == []
    assert not report.fully_supported
    by_op = {f.op_type: f for f in report.warnings}
    assert "Gelu" in by_op and "LeakyRelu" in by_op
    assert "Relu" in by_op["LeakyRelu"].message


def test_dynamic_input_is_an_error():
    model = _model(
        """
        dyn (float[N,3] x) => (float[N,3] y)
        {
            y = Relu (x)
        }
        """
    )
    report = check_onnx_for_edgetpu(model)
    assert len(report.errors) == 1
    assert report.errors[0].node == "x"
    assert not report.fully_supported


def test_oversized_leading_dims_are_an_error():
    model = _model(
        """
        big (float[2,3,4,4] x) => (float[2,3,4,4] y)
        {
            y = Relu (x)
        }
        """
    )
    report = check_onnx_for_edgetpu(model)
    assert any(f.op_type == "tensor shape" for f in report.errors)


def test_unknown_op_warns():
    model = _model(
        """
        unsup (float[2,3] x) => (float[2,3] y)
        {
            y = Selu (x)
        }
        """
    )
    report = check_onnx_for_edgetpu(model)
    assert [f.op_type for f in report.warnings] == ["Selu"]


def _big_conv_model(channels: int = 64, size: int = 32) -> onnx.ModelProto:
    rng = np.random.RandomState(0)
    w = numpy_helper.from_array(
        rng.randn(channels, channels, 3, 3).astype(np.float32), name="w"
    )
    b = numpy_helper.from_array(np.zeros(channels, np.float32), name="b")
    model = _model(
        f"""
        big (float[1,{channels},{size},{size}] x) => (float[1,{channels},{size},{size}] y)
        {{
            c = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w, b)
            y = Relu (c)
        }}
        """,
        initializer=[w, b],
    )
    onnx.checker.check_model(model)
    return model


def test_large_activation_warns_about_nchw_entry_transpose():
    report = check_onnx_for_edgetpu(_big_conv_model(64, 32))
    size_warnings = [f for f in report.warnings if f.op_type == "activation size"]
    assert len(size_warnings) == 1
    assert "nhwc" in size_warnings[0].message
    assert not report.fully_supported


def test_small_model_has_no_activation_size_warning():
    report = check_onnx_for_edgetpu(_relu_model())
    assert [f for f in report.warnings if f.op_type == "activation size"] == []
    report = check_onnx_for_edgetpu(_big_conv_model(4, 128))
    assert [f for f in report.warnings if f.op_type == "activation size"] == []


def test_activation_size_warning_uses_inferred_shapes():
    # The input itself is small; only the inferred intermediate is large.
    rng = np.random.RandomState(0)
    w = numpy_helper.from_array(rng.randn(64, 4, 3, 3).astype(np.float32), name="w")
    b = numpy_helper.from_array(np.zeros(64, np.float32), name="b")
    model = _model(
        """
        expand (float[1,4,64,64] x) => (float[1,64,64,64] y)
        {
            c = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w, b)
            y = Relu (c)
        }
        """,
        initializer=[w, b],
    )
    onnx.checker.check_model(model)
    report = check_onnx_for_edgetpu(model)
    assert [f.op_type for f in report.warnings] == ["activation size"]


# ---------------------------------------------------------------------------
# Calibration data (onnx + numpy only)
# ---------------------------------------------------------------------------


def test_random_representative_dataset_shapes_and_count():
    from onnxsim import tflite_export

    model = _model(
        """
        two (float[1,3] x, int64[2,2] idx) => (float[1,3] y)
        {
            y = Relu (x)
        }
        """
    )
    gen = tflite_export.random_representative_dataset(model, num_samples=3, seed=1)
    batches = list(gen())
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 2
        assert batch[0].shape == (1, 3) and batch[0].dtype == np.float32
        assert batch[1].shape == (2, 2) and batch[1].dtype == np.int64
        assert batch[0].min() >= 0.0 and batch[0].max() < 1.0


# ---------------------------------------------------------------------------
# Quantization argument validation (no TensorFlow needed: misuse raises first)
# ---------------------------------------------------------------------------


def test_int8_rejects_dynamic_range_optimizations():
    with pytest.raises(ValueError, match="mutually exclusive"):
        onnxsim.export_tflite(
            _relu_model(), optimizations=["DEFAULT"], int8_quantize=True
        )


def test_representative_dataset_requires_int8():
    with pytest.raises(ValueError, match="requires int8_quantize=True"):
        onnxsim.export_tflite(_relu_model(), representative_dataset=lambda: iter([]))


def test_inference_io_dtype_requires_int8():
    with pytest.raises(ValueError, match="requires int8_quantize=True"):
        onnxsim.export_tflite(_relu_model(), inference_io_dtype="uint8")


def test_inference_io_dtype_rejects_unknown_names():
    pytest.importorskip("tensorflow", reason="tensorflow is not installed")
    with pytest.raises(ValueError, match="uint8.*int8"):
        onnxsim.export_tflite(
            _relu_model(), int8_quantize=True, inference_io_dtype="float16"
        )


# ---------------------------------------------------------------------------
# Compiler wrapper (stubbed compiler binary)
# ---------------------------------------------------------------------------

_SAMPLE_COMPILER_LOG = """Edge TPU Compiler version 16.0.384591198
Model compiled successfully in 63 ms.
Number of Edge TPU subgraphs: 1
Total number of operations: 8
Operator                       Count      Status

TRANSPOSE                      2          Mapped to Edge TPU
MUL                            2          Mapped to Edge TPU
PAD                            1          Mapped to Edge TPU
QUANTIZE                       2          Mapped to Edge TPU
AVERAGE_POOL_2D                1          Mapped to Edge TPU
Compilation succeeded!
"""


def _stub_compiler(monkeypatch, tmp_path, log_text=_SAMPLE_COMPILER_LOG):
    """Replace subprocess.run with a fake edgetpu_compiler writing canned outputs."""
    monkeypatch.setattr(
        edgetpu_export,
        "find_edgetpu_compiler",
        lambda compiler=None: "/fake/edgetpu_compiler",
    )

    def fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("-o") + 1]
        src = cmd[-1]
        stem = os.path.splitext(os.path.basename(src))[0]
        compiled = os.path.join(out_dir, stem + "_edgetpu.tflite")
        with open(compiled, "wb") as f:
            f.write(b"fake-edgetpu-model")
        return subprocess.CompletedProcess(cmd, 0, log_text, "")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_compile_for_edgetpu_parses_operator_log(tmp_path, monkeypatch):
    _stub_compiler(monkeypatch, tmp_path)
    result = compile_for_edgetpu(b"fake-tflite", out_dir=str(tmp_path))
    assert isinstance(result, EdgeTPUCompileResult)
    assert result.success
    assert result.num_subgraphs == 1
    assert result.total_ops == 8
    assert result.fully_mapped
    assert result.compiled_model == b"fake-edgetpu-model"
    assert "8/8 ops mapped" in result.summary()


def test_compile_for_edgetpu_writes_output_path(tmp_path, monkeypatch):
    _stub_compiler(monkeypatch, tmp_path)
    out = tmp_path / "model_edgetpu.tflite"
    result = compile_for_edgetpu(b"fake-tflite", output_path=str(out))
    assert out.read_bytes() == b"fake-edgetpu-model"
    assert result.output_path == str(out)


def test_compile_for_edgetpu_reports_cpu_fallback(monkeypatch, tmp_path):
    log = _SAMPLE_COMPILER_LOG.replace(
        "AVERAGE_POOL_2D                1          Mapped to Edge TPU",
        "LEAKY_RELU                     1          Operation not supported",
    )
    _stub_compiler(monkeypatch, tmp_path, log_text=log)
    result = compile_for_edgetpu(b"fake-tflite")
    assert result.success  # the compiler still succeeds with CPU fallback
    assert not result.fully_mapped
    assert [o.op for o in result.operators if not o.mapped] == ["LEAKY_RELU"]


def test_compile_for_edgetpu_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        edgetpu_export,
        "find_edgetpu_compiler",
        lambda compiler=None: "/fake/edgetpu_compiler",
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 1, "error: non-broadcastable operands", ""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = compile_for_edgetpu(b"fake-tflite")
    assert not result.success
    assert result.compiled_model is None


def test_missing_compiler_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(
        edgetpu_export, "find_edgetpu_compiler", lambda compiler=None: None
    )
    with pytest.raises(RuntimeError, match="edgetpu-compiler"):
        compile_for_edgetpu(b"fake-tflite")


# ---------------------------------------------------------------------------
# LiteRT runtime helpers (no hardware)
# ---------------------------------------------------------------------------


def test_find_edgetpu_library_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "libedgetpu.so.1"
    fake.write_bytes(b"fake")
    monkeypatch.setenv("EDGETPU_LIBRARY", str(fake))
    assert edgetpu_export.find_edgetpu_library() == str(fake)


def test_make_interpreter_without_device_raises_helpful_error(tmp_path):
    pytest.importorskip("ai_edge_litert", reason="LiteRT is not installed")
    fake = tmp_path / "libedgetpu.so.1"
    fake.write_bytes(b"not-a-real-library")
    with pytest.raises(RuntimeError, match="Edge TPU"):
        edgetpu_export.make_litert_interpreter(
            b"model", use_edgetpu=True, edgetpu_library=str(fake)
        )


def test_setup_hint_covers_udev_and_litert():
    hint = edgetpu_export.edgetpu_setup_hint()
    assert "udev" in hint and "ai-edge-litert" in hint and "edgetpu_compiler" in hint


# ---------------------------------------------------------------------------
# End to end with real tools (gated)
# ---------------------------------------------------------------------------


def _needs_compiler():
    return pytest.mark.skipif(
        edgetpu_export.find_edgetpu_compiler() is None,
        reason="edgetpu_compiler is not installed",
    )


@_needs_compiler()
def test_quantize_and_compile_end_to_end(tmp_path):
    pytest.importorskip("tensorflow", reason="tensorflow is not installed")
    pytest.importorskip("ai_edge_litert", reason="LiteRT is not installed")
    model = _relu_model()
    quantized = onnxsim.export_tflite(
        model,
        int8_quantize=True,
        inference_io_dtype="uint8",
        num_calibration_samples=5,
    )
    assert isinstance(quantized, bytes) and len(quantized) > 0

    # The converted model itself already passes the TFLite-level check...
    tflite_report = edgetpu_export.check_tflite_for_edgetpu(quantized)
    assert tflite_report.fully_supported, tflite_report.summary()

    # ...and the real compiler agrees.
    out = tmp_path / "relu_edgetpu.tflite"
    result = edgetpu_export.compile_for_edgetpu(quantized, output_path=str(out))
    assert result.success, result.log_text[-2000:]
    assert result.fully_mapped, result.summary()
    assert out.is_file()

    # The uncompiled quantized model still runs on CPU through LiteRT.
    x = np.array([[0.25, 0.5, 0.75], [0.1, 0.9, 0.4]], np.float32)
    [y] = edgetpu_export.run_litert(quantized, {"x": x})
    np.testing.assert_allclose(
        np.maximum(x, 0).astype(y.dtype), y, rtol=0.05, atol=2.0 / 255
    )


@_needs_compiler()
def test_export_edgetpu_one_shot(tmp_path):
    pytest.importorskip("tensorflow", reason="tensorflow is not installed")
    out = tmp_path / "relu_edgetpu.tflite"
    result = onnxsim.export_edgetpu(
        _relu_model(), output_path=str(out), num_calibration_samples=5
    )
    assert len(result.quantized_model) > 0
    assert result.compile_result.success
    assert result.compile_result.fully_mapped
    assert out.is_file()


@_needs_compiler()
def test_nhwc_compiles_where_nchw_fails_at_size_cliff(tmp_path):
    """The money test for ``io_layout="nhwc"``: a 64ch x 32x32 conv refuses
    compilation with the NCHW entry transpose (``large activation tensors``)
    while the identical channel-last graph maps fully (measured against
    edgetpu_compiler 16.0)."""
    pytest.importorskip("tensorflow", reason="tensorflow is not installed")
    pytest.importorskip("ai_edge_litert", reason="LiteRT is not installed")
    model = _big_conv_model(64, 32)

    nchw = onnxsim.export_tflite(
        model,
        int8_quantize=True,
        inference_io_dtype="uint8",
        num_calibration_samples=5,
    )
    nchw_result = edgetpu_export.compile_for_edgetpu(nchw)
    assert not nchw_result.success

    nhwc = onnxsim.export_tflite(
        model,
        int8_quantize=True,
        inference_io_dtype="uint8",
        num_calibration_samples=5,
        io_layout="nhwc",
    )
    # No transposes at all in the channel-last graph...
    tflite_report = edgetpu_export.check_tflite_for_edgetpu(nhwc)
    assert "TRANSPOSE" not in [o.op for o in tflite_report.operators]
    # ...and the real compiler maps every op.
    out = tmp_path / "big_nhwc_edgetpu.tflite"
    nhwc_result = edgetpu_export.compile_for_edgetpu(nhwc, output_path=str(out))
    assert nhwc_result.success, nhwc_result.log_text[-2000:]
    assert nhwc_result.fully_mapped, nhwc_result.summary()
    assert out.is_file()


@_needs_compiler()
def test_export_edgetpu_nhwc_one_shot(tmp_path):
    pytest.importorskip("tensorflow", reason="tensorflow is not installed")
    out = tmp_path / "big_nhwc_edgetpu.tflite"
    result = onnxsim.export_edgetpu(
        _big_conv_model(64, 32),
        output_path=str(out),
        num_calibration_samples=5,
        io_layout="nhwc",
    )
    assert result.compile_result.success
    assert result.compile_result.fully_mapped
    assert out.is_file()
