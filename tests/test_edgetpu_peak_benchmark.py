"""Unit tests for scripts/edgetpu/ peak-benchmark pure logic.

Covers model construction (onnx-only), the roofline estimator, and
``peak_benchmark.run_suite`` with stubbed onnxsim conversion/compilation --
no TensorFlow, compiler binary, or Edge TPU hardware needed. On-device timing
(``benchmark_device.py``) can't run without hardware, like the other vendor
runners under scripts/.
"""

import json
import os
import sys
import types

import onnx
import pytest

_EDGETPU_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edgetpu"
)
if _EDGETPU_DIR not in sys.path:
    sys.path.insert(0, _EDGETPU_DIR)

import models  # noqa: E402
import peak_benchmark  # noqa: E402


def test_all_models_build_and_validate():
    suite = models.all_models()
    assert set(suite) == {
        "pointwise-8",
        "dense3x3-6",
        "mbblock-4",
        "fc-4k",
        "cliff-64x32",
        "pointwise-48",
    }
    for name, bm in suite.items():
        onnx.checker.check_model(bm.model)
        assert bm.gmac > 0
        initializer_names = {t.name for t in bm.model.graph.initializer}
        real_inputs = [
            i for i in bm.model.graph.input if i.name not in initializer_names
        ]
        assert len(real_inputs) == 1, name
        dims = [d.dim_value for d in real_inputs[0].type.tensor_type.shape.dim]
        assert dims == bm.input_shape, name


def test_gmac_spot_check_pointwise_8():
    bm = models.all_models()["pointwise-8"]
    # 8 layers of 1x1 256ch @ 16x16, stride 1, same padding.
    assert bm.gmac == pytest.approx(8 * 256 * 16 * 16 * 256 / 1e9)
    assert bm.input_shape == [1, 256, 16, 16]
    assert bm.io_bytes == 2 * 256 * 16 * 16


def test_fc_model_shapes_and_macs():
    bm = models.all_models()["fc-4k"]
    assert bm.gmac == pytest.approx(1024 * 4096 / 1e9)
    assert bm.input_shape == [1, 64, 8, 8]
    assert bm.output_shape == [1, 1024]
    assert bm.io_bytes == 4096 + 1024


def test_suite_is_deterministic():
    first = {n: m.model.SerializeToString() for n, m in models.all_models().items()}
    second = {n: m.model.SerializeToString() for n, m in models.all_models().items()}
    assert first == second


def test_estimate_roofline_math():
    # 0.805 GMAC like pointwise-48 with 128 KiB of I/O.
    est = peak_benchmark.estimate(0.805306368, 131072)
    # Compute at the 40% ceiling dominates USB3 but not USB2.
    assert est["t_compute_ms"] == pytest.approx(1.611 / 1.6, rel=1e-3)
    assert est["t_usb3_ms"] == pytest.approx(131072 / 350e6 * 1e3, rel=1e-9)
    assert est["usb3_regime"] == "compute"
    assert est["tops_usb3"] == pytest.approx(1.6, rel=1e-3)
    assert est["tops_usb2"] < est["tops_usb3"]
    # Tiny model: link-bound everywhere.
    small = peak_benchmark.estimate(0.004, 5120)
    assert small["usb3_regime"] == "link"


def test_compiler_memories_parsing():
    log = (
        "On-chip memory used for caching model parameters: 10.00KiB\n"
        "Off-chip memory used for streaming uncached model parameters: 0.00B\n"
    )
    assert peak_benchmark._compiler_memories(log) == ("10.00KiB", "0.00B")
    assert peak_benchmark._compiler_memories("no memory lines\n") == ("", "")


def _fake_compile(out_path):
    with open(out_path, "wb") as f:
        f.write(b"fake-edgetpu-model")
    return types.SimpleNamespace(
        success=True,
        mapped_ops=5,
        total_ops=5,
        fully_mapped=True,
        num_subgraphs=1,
        log_text=(
            "On-chip memory used for caching model parameters: 10.00KiB\n"
            "Off-chip memory used for streaming uncached model parameters: 0.00B\n"
        ),
    )


def test_run_suite_writes_manifest_with_stubs(tmp_path, monkeypatch):
    import onnxsim

    seen = {}

    def fake_quantize(model, **kwargs):
        seen["kwargs"] = kwargs
        return b"fake-quantized"

    def fake_compile(blob, output_path=None, **kwargs):
        assert blob == b"fake-quantized"
        return _fake_compile(output_path)

    monkeypatch.setattr(onnxsim, "quantize_for_edgetpu", fake_quantize)
    monkeypatch.setattr(onnxsim, "compile_for_edgetpu", fake_compile)
    rows, manifest = peak_benchmark.run_suite(
        str(tmp_path), names=["pointwise-8", "fc-4k"], samples=3
    )
    assert seen["kwargs"]["num_calibration_samples"] == 3
    assert seen["kwargs"]["io_layout"] == "nhwc"
    assert [r["model"] for r in rows] == ["pointwise-8", "fc-4k"]
    assert rows[0]["mapped"] == "5/5"
    assert set(manifest) == {"pointwise-8", "fc-4k"}
    entry = manifest["pointwise-8"]
    assert entry["gmac"] == pytest.approx(0.134217728)
    assert entry["input_shape"] == [1, 256, 16, 16]
    assert (tmp_path / entry["file"]).is_file()


def test_run_suite_skips_unmapped_in_manifest(tmp_path, monkeypatch):
    import onnxsim

    monkeypatch.setattr(
        onnxsim, "quantize_for_edgetpu", lambda model, **kwargs: b"fake"
    )

    def fake_compile(blob, output_path=None, **kwargs):
        return types.SimpleNamespace(
            success=False,
            mapped_ops=0,
            total_ops=0,
            fully_mapped=False,
            num_subgraphs=0,
            log_text="Compilation failed",
        )

    monkeypatch.setattr(onnxsim, "compile_for_edgetpu", fake_compile)
    rows, manifest = peak_benchmark.run_suite(str(tmp_path), names=["cliff-64x32"])
    assert rows[0]["mapped"] == "FAIL"
    assert manifest == {}


def test_run_suite_skip_compile(tmp_path, monkeypatch):
    import onnxsim

    def fail_quantize(model, **kwargs):
        raise AssertionError("must not convert with --skip-compile")

    monkeypatch.setattr(onnxsim, "quantize_for_edgetpu", fail_quantize)
    rows, manifest = peak_benchmark.run_suite(
        str(tmp_path), names=["pointwise-8"], skip_compile=True
    )
    # Without compilation there is nothing to time on-device.
    assert manifest == {}
    assert rows[0]["mapped"] == ""
    assert rows[0]["tops_usb3"] > 0


def test_main_writes_manifest_file(tmp_path, monkeypatch, capsys):
    import onnxsim

    monkeypatch.setattr(
        onnxsim, "quantize_for_edgetpu", lambda model, **kwargs: b"fake"
    )

    def fake_compile(blob, output_path=None, **kwargs):
        return _fake_compile(output_path)

    monkeypatch.setattr(onnxsim, "compile_for_edgetpu", fake_compile)
    rc = peak_benchmark.main(
        ["--out-dir", str(tmp_path), "--models", "fc-4k", "--samples", "2"]
    )
    assert rc == 0
    manifest_path = tmp_path / "peak_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert list(manifest) == ["fc-4k"]
    out = capsys.readouterr().out
    assert "fc-4k" in out and "TOPS" in out
