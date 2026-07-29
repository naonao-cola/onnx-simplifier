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
from onnxsim import backend


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

# Span emitted for each fold group's ONNX Runtime session run, i.e. only when
# constant folding actually runs the op through the executor. It is profiled at
# the executor call site, so it appears for every executor (the built-in ONNX
# Runtime one and the Python trampoline that ``simplify()`` injects). The
# built-in executor additionally nests ``OrtSessionInit``/``OrtSessionRun`` under
# it, but the Python path used here runs the session opaquely inside the
# trampoline, so only the ``OrtSession`` span is asserted. onnxsim skips folding
# when it cannot run the op in the current build/environment, so this is checked
# separately from the always-present spans above.
_ORT_SESSION_SPAN = "OrtSession"


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
    are profiled too. When a fold actually executes, an ``OrtSession`` span
    appears in the trace for each fold group, nested under FoldConstant."""
    model = _foldable_model()
    out = str(tmp_path / "trace.json")
    model_opt, ok = onnxsim.simplify(model, profile=out)
    assert ok

    complete, _ = _load_trace(out)
    names = {e["name"] for e in complete}

    # The Add of two initializers collapses to a single initializer, leaving one
    # Add node, iff constant folding ran the op through the executor. onnxsim
    # skips folding when it cannot run the op in the current build/environment,
    # so there would be nothing to profile then.
    add_nodes = [n for n in model_opt.graph.node if n.op_type == "Add"]
    if len(add_nodes) != 1:
        pytest.skip("constant folding did not run the ONNX Runtime executor here")

    assert _ORT_SESSION_SPAN in names, (
        f"missing {_ORT_SESSION_SPAN!r} span; got {sorted(names)}"
    )

    # Each session run nests inside a FoldConstant span.
    sessions = [e for e in complete if e["name"] == _ORT_SESSION_SPAN]
    folds = [e for e in complete if e["name"] == "FoldConstant"]
    for sess in sessions:
        assert any(_contained_in(sess, f) for f in folds), (
            f"{_ORT_SESSION_SPAN} span not nested under any FoldConstant span"
        )


def test_ort_profile_writes_session_traces(tmp_path, monkeypatch):
    """``ort_profile`` turns on onnxruntime's own per-operator session profiler
    for the folding sessions, writing a Chrome trace per session (separate from
    onnxsim's own ``profile`` trace)."""
    if not backend.has_onnxruntime():
        pytest.skip("onnxruntime not installed; native session profiling unavailable")

    # onnxruntime writes the trace(s) relative to the cwd, so run in tmp_path.
    monkeypatch.chdir(tmp_path)
    model_opt, ok = onnxsim.simplify(_foldable_model(), ort_profile="ortprof")
    assert ok

    # The trace only exists if folding actually ran a session (see the fold-count
    # reasoning in test_profile_captures_ort_session_runs).
    add_nodes = [n for n in model_opt.graph.node if n.op_type == "Add"]
    if len(add_nodes) != 1:
        pytest.skip("constant folding did not run the ONNX Runtime executor here")

    # We did not pass ``profile``, so any JSON here is an onnxruntime session
    # trace. (Its exact name depends on the onnxruntime version's support for a
    # custom prefix, so match on directory rather than prefix.)
    traces = list(tmp_path.glob("*.json"))
    assert traces, "expected an onnxruntime session profiling trace to be written"

    # Each trace is a valid onnxruntime profile: a JSON array of event objects.
    with open(traces[0]) as f:
        events = json.load(f)
    assert isinstance(events, list) and events


def test_ort_profile_restores_env(tmp_path, monkeypatch):
    # A pre-existing ONNXSIM_ORT_PROFILE value is restored after the call.
    monkeypatch.chdir(tmp_path)
    sentinel = "outer_ort_prefix"
    monkeypatch.setenv("ONNXSIM_ORT_PROFILE", sentinel)
    onnxsim.simplify(_foldable_model(), ort_profile="inner")
    assert os.environ["ONNXSIM_ORT_PROFILE"] == sentinel


def test_no_ort_profile_by_default(tmp_path, monkeypatch):
    # Without ``ort_profile`` no session trace is written and the env is untouched.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONNXSIM_ORT_PROFILE", raising=False)
    _, ok = onnxsim.simplify(_foldable_model())
    assert ok
    assert not list(tmp_path.glob("*.json"))
    assert "ONNXSIM_ORT_PROFILE" not in os.environ


def test_merge_ort_profile_unifies_trace(tmp_path, monkeypatch):
    """``merge_ort_profile`` folds onnxruntime's own per-operator session traces
    into onnxsim's ``profile`` trace, so onnxruntime operator events appear in the
    single unified trace (and no stray onnxruntime files are left behind)."""
    if not backend.has_onnxruntime():
        pytest.skip("onnxruntime not installed; native session profiling unavailable")
    if not hasattr(backend.rt.SessionOptions(), "profile_file_prefix"):
        pytest.skip("onnxruntime too old to redirect its profile output for merging")

    monkeypatch.chdir(tmp_path)
    out = str(tmp_path / "trace.json")
    model_opt, ok = onnxsim.simplify(
        _foldable_model(), profile=out, merge_ort_profile=True
    )
    assert ok

    add_nodes = [n for n in model_opt.graph.node if n.op_type == "Add"]
    if len(add_nodes) != 1:
        pytest.skip("constant folding did not run the ONNX Runtime executor here")

    complete, _ = _load_trace(out)
    # onnxsim's own spans are still there...
    assert _EXPECTED_SPANS <= {e["name"] for e in complete}
    # ...and onnxruntime's per-operator events were merged in on their own track.
    ort_ops = [e for e in complete if e.get("cat") == "onnxruntime"]
    assert ort_ops, "expected merged onnxruntime operator events in the trace"

    # The temporary onnxruntime traces were cleaned up: the only JSON here is our
    # unified trace.
    assert list(tmp_path.glob("*.json")) == [tmp_path / "trace.json"]


def test_merge_ort_profile_implies_profile(tmp_path, monkeypatch):
    # merge_ort_profile without an explicit ``profile`` still writes the default
    # onnxsim trace (it needs one to merge into).
    if not backend.has_onnxruntime():
        pytest.skip("onnxruntime not installed; native session profiling unavailable")
    monkeypatch.chdir(tmp_path)
    _, ok = onnxsim.simplify(_foldable_model(), merge_ort_profile=True)
    assert ok
    assert os.path.exists(tmp_path / "onnxsim_profile.json")


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
