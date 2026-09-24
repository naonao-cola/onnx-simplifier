"""Tests for ``onnxsim/webnn_tinygrad_tuning.py``: per-node timing of rustnn's
native WebNN vs. tinygrad (with and without BEAM search), validated against
onnx's reference evaluator, with the result stored on the node.

Node extraction and result bookkeeping need neither backend. Tests that run
tinygrad skip without it; a missing/unusable rustnn is itself part of what's
tested (it must be recorded as an errored timing, not raise), so only the
test that needs a real WebNN timing skips on it.

``ONNXSIM_RUSTNN_DEVICE_TYPES`` / ``ONNXSIM_TINYGRAD_DEVICES`` (comma
separated) widen that test to more WebNN device types and tinygrad devices;
``.github/workflows/apple-integration.yml`` sets ``cpu,npu`` and
``CPU,METAL`` on a macOS runner.
"""

import os

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim import webnn_tinygrad_tuning as tuning
from onnxsim.rustnn_runtime import probe_rustnn
from onnxsim.webnn_tinygrad_tuning import (
    TUNING_METADATA_KEY,
    BackendTiming,
    NodeTuningResult,
    extract_node_model,
    random_feeds,
    read_tuning_result,
    tune_node,
)


def _model(body, initializer=(), opset=17, ir_version=8):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _conv_model():
    rng = np.random.default_rng(0)
    return _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,8,8] y)
        {
          conv = Conv<pads = [1, 1, 1, 1]>(x, w, b)
          relu = Relu(conv)
          y = Add(relu, x)
        }
        """,
        [
            numpy_helper.from_array(
                rng.standard_normal((4, 4, 3, 3)).astype(np.float32), "w"
            ),
            numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "b"),
        ],
    )


def _named(model):
    # The text format has no node-name syntax; name each node after its output.
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def test_extract_node_model_isolates_node_with_its_constants():
    model = _named(_conv_model())
    sub = extract_node_model(model, "relu")

    onnx.checker.check_model(sub, full_check=True)
    assert [n.op_type for n in sub.graph.node] == ["Relu"]
    assert [i.name for i in sub.graph.input] == ["conv"]
    assert [d.dim_value for d in sub.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        4,
        8,
        8,
    ]
    assert [o.name for o in sub.graph.output] == ["relu"]

    sub = extract_node_model(model, "conv")
    assert [i.name for i in sub.graph.input] == ["x"]
    assert sorted(i.name for i in sub.graph.initializer) == ["b", "w"]


def test_extract_node_model_turns_constant_nodes_into_initializers():
    model = _named(
        _model(
            """
            g (float[2,6] x) => (float[3,4] y)
            {
              shape = Constant<value = int64[2] {3, 4}>()
              y = Reshape(x, shape)
            }
            """
        )
    )
    sub = extract_node_model(model, "y")
    assert [i.name for i in sub.graph.initializer] == ["shape"]
    assert numpy_helper.to_array(sub.graph.initializer[0]).tolist() == [3, 4]


def test_extract_node_model_needs_static_shapes():
    model = _named(
        _model(
            """
            g (float[N,4] x) => (float[N,4] y)
            {
              y = Relu(x)
            }
            """
        )
    )
    with pytest.raises(ValueError, match="no static shape"):
        extract_node_model(model, "y")
    with pytest.raises(ValueError, match="no node named"):
        extract_node_model(model, "missing")


def test_random_feeds_is_deterministic_and_typed():
    model = _model(
        """
        g (float[2,3] x, int64[4] i) => (float[2,3] y)
        {
          y = Identity(x)
        }
        """
    )
    a, b = random_feeds(model, seed=3), random_feeds(model, seed=3)
    assert a["x"].dtype == np.float32 and a["x"].shape == (2, 3)
    assert a["i"].dtype == np.int64 and set(a["i"].tolist()) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(a["x"], b["x"])


def test_node_tuning_result_json_roundtrip():
    result = NodeTuningResult(
        "conv",
        "Conv",
        [
            BackendTiming("webnn", "cpu/auto", error="rustnn unavailable: x"),
            BackendTiming("tinygrad", "CPU BEAM=0", 1.5, 1.2, 5, 0.0),
        ],
    )
    result.winner = result.timings[1]
    back = NodeTuningResult.from_json(result.to_json())
    assert back.winner == result.winner
    assert back.timings[1] == result.timings[1]
    assert not back.timings[0].ok and back.timings[0].error == "rustnn unavailable: x"


def test_tune_node_records_missing_webnn_and_picks_tinygrad(monkeypatch):
    pytest.importorskip("tinygrad")
    monkeypatch.setattr(
        tuning, "probe_rustnn", lambda *a: (False, "pywebnn not installed")
    )
    model = _named(_conv_model())

    result = tune_node(model, "conv", beams=(0,), warmup=1, runs=2)

    webnn, tiny = result.timings
    assert webnn.backend == "webnn" and not webnn.ok
    assert "pywebnn not installed" in webnn.error
    assert tiny.backend == "tinygrad" and tiny.ok and tiny.config.endswith("BEAM=0")
    assert tiny.max_abs_error is not None and tiny.max_abs_error < 1e-3
    assert result.winner == tiny

    stored = read_tuning_result(model, "conv")
    assert stored is not None and stored.winner == tiny
    keys = [
        e.key
        for e in next(n for n in model.graph.node if n.name == "conv").metadata_props
    ]
    assert keys == [TUNING_METADATA_KEY]


def test_tune_node_excludes_backends_that_disagree_with_reference(monkeypatch):
    pytest.importorskip("tinygrad")

    def wrong_webnn(model, feeds, **kwargs):
        wrong = [
            np.full(
                [d.dim_value for d in o.type.tensor_type.shape.dim], 7.0, np.float32
            )
            for o in model.graph.output
        ]
        return BackendTiming("webnn", "cpu/fake", 0.001, 0.001, 1), wrong

    monkeypatch.setattr(tuning, "time_webnn", wrong_webnn)
    result = tune_node(
        _named(_conv_model()), "conv", beams=(0,), warmup=1, runs=2, write_back=False
    )

    webnn = result.timings[0]
    assert webnn.ok and webnn.max_abs_error > 1.0
    # WebNN is "faster" here but wrong, so tinygrad must win.
    assert result.winner is not None and result.winner.backend == "tinygrad"


def _env_list(name, default):
    return [d.strip() for d in os.environ.get(name, default).split(",") if d.strip()]


def test_tune_node_with_rustnn_and_beam_search():
    pytest.importorskip("tinygrad")
    pytest.importorskip(
        "webnn", reason="pywebnn (rustnn's Python bindings) is not installed"
    )
    device_types = []
    for device_type in _env_list("ONNXSIM_RUSTNN_DEVICE_TYPES", "cpu"):
        ok, reason = probe_rustnn(device_type)
        if ok:
            device_types.append(device_type)
        elif device_type == "cpu":
            pytest.skip(f"rustnn cpu context unavailable: {reason}")
    tinygrad_devices = _env_list("ONNXSIM_TINYGRAD_DEVICES", "") or [None]
    model = _named(_conv_model())

    # Loose enough for Core ML's float16 math to count as correct.
    result = tune_node(
        model,
        "conv",
        webnn_device_types=device_types,
        tinygrad_devices=tinygrad_devices,
        beams=(0, 1),
        warmup=1,
        runs=3,
        atol=2e-2,
        rtol=2e-2,
    )

    expected = ["webnn"] * len(device_types) + ["tinygrad"] * (
        2 * len(tinygrad_devices)
    )
    assert [t.backend for t in result.timings] == expected
    for t in result.timings:
        assert t.ok, (t.config, t.error)
        assert t.max_abs_error < (1e-3 if t.config.startswith("cpu/") else 2e-2), t
        assert 0 < t.min_ms <= t.median_ms
    assert result.winner == min(result.timings, key=lambda t: t.median_ms)


def test_tune_model_defaults_to_compute_heavy_ops(monkeypatch):
    seen = []
    monkeypatch.setattr(
        tuning, "tune_node", lambda model, name, **kw: seen.append(name)
    )
    tuning.tune_model(_named(_conv_model()))
    assert seen == ["conv"]
    with pytest.raises(TypeError, match="feeds"):
        tuning.tune_model(_named(_conv_model()), feeds={})
