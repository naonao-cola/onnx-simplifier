"""Tests for ``onnxsim.profile_plot`` (the "node reduction per loop" PNG).

``simplify(..., profile=...)`` writes "NodeCount" counter events into its
trace (see ``test_profiling.py``); this module turns those into a plot.
matplotlib is an optional dependency (``onnxsim[plot]``), so these tests are
skipped when it is not installed.
"""

import json
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnxsim

matplotlib = pytest.importorskip("matplotlib")

from onnxsim import profile_plot  # noqa: E402  (after importorskip)


def _foldable_model() -> onnx.ModelProto:
    """A tiny model whose ``Add`` of two initializers constant-folds away, so a
    real simplification round runs and produces ``NodeCount`` events."""
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


def test_load_node_counts(tmp_path):
    trace = str(tmp_path / "trace.json")
    _, ok = onnxsim.simplify(_foldable_model(), profile=trace)
    assert ok

    counts = profile_plot.load_node_counts(trace)
    # "Initial" is recorded once unconditionally; "Optimize" every inner-loop
    # round, which always runs at least once.
    assert "Initial" in counts and "Optimize" in counts
    assert len(counts["Initial"]) == 1
    assert counts["Initial"][0] == 2  # add_const, add_x
    assert all(isinstance(c, int) for cs in counts.values() for c in cs)


def test_plot_node_reduction_writes_png(tmp_path):
    trace = str(tmp_path / "trace.json")
    _, ok = onnxsim.simplify(_foldable_model(), profile=trace)
    assert ok

    out = profile_plot.plot_node_reduction(trace)
    assert out == f"{trace}_node_reduction.png"
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_node_reduction_custom_out_path(tmp_path):
    trace = str(tmp_path / "trace.json")
    _, ok = onnxsim.simplify(_foldable_model(), profile=trace)
    assert ok

    out = str(tmp_path / "custom.png")
    result = profile_plot.plot_node_reduction(trace, out)
    assert result == out
    assert os.path.exists(out)


def test_plot_node_reduction_raises_without_node_count_events(tmp_path):
    empty_trace = tmp_path / "empty.json"
    empty_trace.write_text(json.dumps({"traceEvents": []}))
    with pytest.raises(RuntimeError, match="NodeCount"):
        profile_plot.plot_node_reduction(str(empty_trace))
