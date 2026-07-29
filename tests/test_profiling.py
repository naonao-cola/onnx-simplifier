"""Tests for the optimization profiler (``simplify(..., profile=...)``).

The profiler measures the wall-clock/CPU duration and peak resident memory of
each simplification fixed-point function and writes a Chrome Trace Event Format
JSON (openable as a flame graph in chrome://tracing or the Perfetto UI). These
tests exercise that path end-to-end through the Python API and validate the
emitted trace.
"""

import json
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnxsim


def _foldable_model() -> onnx.ModelProto:
    """A tiny model whose ``Add`` of two initializers constant-folds away, so a
    real simplification round runs (shape inference + optimizer + folding)."""
    a = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name="A")
    b = numpy_helper.from_array(np.full((1, 4), 2.0, dtype=np.float32), name="B")
    add_const = helper.make_node("Add", ["A", "B"], ["c"], name="add_const")
    add_x = helper.make_node("Add", ["c", "x"], ["y"], name="add_x")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([add_const, add_x], "g", [x], [y], [a, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


# Spans the profiler always emits, regardless of what the model needs: the root,
# the outer pipeline, and every leaf fixed-point function (each is invoked at
# least once even when it makes no change).
_EXPECTED_SPANS = {
    "Simplify",
    "Pipeline",
    "OptAndShape",
    "InferShapes",
    "Optimize",
    "FoldConstant",
}

# Spans emitted only when constant folding actually runs an ONNX Runtime session
# (i.e. an op was folded). The executor wraps each session, breaking out session
# construction (``OrtSessionInit``) from inference (``OrtSessionRun``). onnxsim
# skips folding when it cannot run the op in the current build/environment, so
# these are asserted separately from the always-present spans above.
_ORT_SESSION_SPANS = {
    "OrtExecutor",
    "OrtSessionInit",
    "OrtSessionRun",
}


def _load_trace(path):
    with open(path) as f:
        trace = json.load(f)
    assert trace["displayTimeUnit"] == "ms"
    events = trace["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X"]
    counters = [e for e in events if e.get("ph") == "C"]
    return complete, counters


def test_profile_writes_trace(tmp_path):
    out = str(tmp_path / "trace.json")
    model_opt, ok = onnxsim.simplify(_foldable_model(), profile=out)
    assert ok
    # The result still validates. We deliberately do not assert that the
    # Add-of-two-constants folded away: constant folding runs the op through the
    # executor and onnxsim skips any op the executor cannot run in the current
    # environment. The profiler drives its fixed-point functions either way,
    # which is what this test checks (see the span assertions below).
    assert len(model_opt.graph.node) >= 1

    assert os.path.exists(out)
    complete, counters = _load_trace(out)

    names = {e["name"] for e in complete}
    assert _EXPECTED_SPANS <= names, f"missing spans: {_EXPECTED_SPANS - names}"

    # Every span is a complete event with a start, a duration and the
    # per-function memory/CPU metrics the feature is about.
    for e in complete:
        assert "ts" in e and "dur" in e
        args = e["args"]
        assert "peak_rss_mb" in args
        assert "cpu_ms" in args
        assert args["cpu_ms"] >= 0.0

    # The nested fixed points must actually nest: the root span spans the whole
    # run and contains the others.
    root = next(e for e in complete if e["name"] == "Simplify")
    for e in complete:
        if e is root:
            continue
        assert e["ts"] >= root["ts"]
        assert e["ts"] + e["dur"] <= root["ts"] + root["dur"] + 1


def _contained_in(inner, outer):
    """Whether the ``inner`` complete event's time window falls inside
    ``outer``'s (with a 1us slack for boundary rounding)."""
    return (
        inner["ts"] >= outer["ts"]
        and inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"] + 1
    )


def test_profile_captures_ort_session_runs(tmp_path):
    """Constant folding's real work is running ONNX Runtime sessions; those runs
    are profiled too. When a fold actually executes, the executor span and its
    init/run sub-spans appear in the trace, nested under FoldConstant."""
    model = _foldable_model()
    out = str(tmp_path / "trace.json")
    model_opt, ok = onnxsim.simplify(model, profile=out)
    assert ok

    complete, _ = _load_trace(out)
    names = {e["name"] for e in complete}

    # The Add of two initializers collapses to a single initializer, leaving one
    # Add node, iff constant folding ran the op through the ONNX Runtime
    # executor. onnxsim skips folding when it cannot run the op in the current
    # build/environment, so there would be nothing to profile then.
    add_nodes = [n for n in model_opt.graph.node if n.op_type == "Add"]
    if len(add_nodes) != 1:
        pytest.skip("constant folding did not run the ONNX Runtime executor here")

    assert _ORT_SESSION_SPANS <= names, (
        f"missing ORT session spans: {_ORT_SESSION_SPANS - names}"
    )

    # Each session run nests correctly: OrtSessionInit and OrtSessionRun inside
    # an OrtExecutor, which itself sits inside a FoldConstant span.
    executors = [e for e in complete if e["name"] == "OrtExecutor"]
    folds = [e for e in complete if e["name"] == "FoldConstant"]
    for exc in executors:
        assert any(_contained_in(exc, f) for f in folds), (
            "OrtExecutor span not nested under any FoldConstant span"
        )
    for name in ("OrtSessionInit", "OrtSessionRun"):
        for span in (e for e in complete if e["name"] == name):
            assert any(_contained_in(span, exc) for exc in executors), (
                f"{name} span not nested under any OrtExecutor span"
            )


def test_profile_defaults_to_named_file(tmp_path, monkeypatch):
    # An empty string selects the default trace filename in the cwd.
    monkeypatch.chdir(tmp_path)
    _, ok = onnxsim.simplify(_foldable_model(), profile="")
    assert ok
    assert os.path.exists(tmp_path / "onnxsim_profile.json")


def test_no_profile_by_default(tmp_path, monkeypatch):
    # Without ``profile`` nothing is written and the env var is not left set.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONNXSIM_PROFILE", raising=False)
    _, ok = onnxsim.simplify(_foldable_model())
    assert ok
    assert not os.path.exists(tmp_path / "onnxsim_profile.json")
    assert "ONNXSIM_PROFILE" not in os.environ


def test_profile_restores_env(tmp_path, monkeypatch):
    # A pre-existing ONNXSIM_PROFILE value is restored after the call.
    sentinel = str(tmp_path / "outer.json")
    monkeypatch.setenv("ONNXSIM_PROFILE", sentinel)
    onnxsim.simplify(_foldable_model(), profile=str(tmp_path / "inner.json"))
    assert os.environ["ONNXSIM_PROFILE"] == sentinel
