"""Automatic calibration-method choice.

Two layers:

- ``calibrate(method="auto")`` / ``CalibrationStats.auto_ranges``: per tensor,
  the candidate method with the lowest expected uint8 quantization error on
  that tensor's own histogram -- with graph outputs, Sigmoid/Softmax outputs
  and inputs, and ``minmax_tensor_names`` never clipped.
- ``pick_calibration``: one method for the whole model, the one a metric on
  the quantized model's *outputs* scores best -- all candidates from one
  calibration run.
"""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim.calibration import collect_calibration_stats


def _model(body, initializer=(), opset=17, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _heavy_tailed(num_batches, n=1 << 20, outliers=1, value=40.0, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_batches):
        x = rng.standard_normal((1, n)).astype(np.float32)
        x[0, rng.integers(n, size=outliers)] = value
        batches.append({"x": x})
    return batches


# h is an internal tensor with x's heavy-tailed distribution; y is the output.
# Rounding error on a 256-level grid over [-V, V] is ~V^2 / 780000 per value,
# clipping one value at V costs ~V^2: clipping only pays off below ~1 outlier
# in 780k values -- hence the 1M-value tensors (one outlier each).
_SCALE = """g (float[1, 1048576] x) => (float[1, 1048576] y) {
    h = Mul(x, k)
    y = Add(h, c)
}"""
_SIGMOID = """g (float[1, 1048576] x) => (float[1, 1048576] y) {
    h = Mul(x, k)
    s = Sigmoid(h)
    y = Mul(s, k)
}"""


def _consts():
    return [
        numpy_helper.from_array(np.array(1.0, np.float32), "k"),
        numpy_helper.from_array(np.array(0.5, np.float32), "c"),
    ]


def test_auto_clips_a_heavy_tailed_tensor():
    m = _model(_SCALE, _consts())
    data = _heavy_tailed(4)
    stats = collect_calibration_stats(m, data, tensor_names=["h", "y"])
    ranges, choices = stats.auto_ranges()
    lo, hi = stats.observed["h"]
    assert hi == pytest.approx(40.0)
    # one outlier in 1M: clipping it beats spending the grid on it
    assert choices["h"] != "minmax"
    assert ranges["h"][1] < 10.0 and ranges["h"][0] == pytest.approx(lo, rel=0.3)
    # the graph output keeps its exact range
    assert choices["y"] == "minmax (protected)"
    assert ranges["y"] == stats.observed["y"]
    # calibrate(method="auto") is the same thing
    assert onnxsim.calibrate(m, data, method="auto", tensor_names=["h", "y"]) == ranges


def test_auto_keeps_minmax_without_outliers():
    m = _model(_SCALE, _consts())
    rng = np.random.default_rng(1)
    data = [
        {"x": rng.uniform(-1, 1, (1, 1 << 20)).astype(np.float32)} for _ in range(2)
    ]
    _, choices = collect_calibration_stats(m, data, tensor_names=["h"]).auto_ranges()
    # a uniform tensor has no tail: every clip costs more than it saves
    assert choices["h"] == "minmax"


def test_auto_never_clips_sigmoid_inputs_or_outputs():
    m = _model(_SIGMOID, _consts())
    data = _heavy_tailed(2, value=12.0)
    stats = collect_calibration_stats(m, data, tensor_names=["h", "s"])
    ranges, choices = stats.auto_ranges()
    for t in ("h", "s"):
        assert choices[t] == "minmax (protected)"
        assert ranges[t] == stats.observed[t]
    # ... unless asked to: then the logit gets clipped like any tensor
    ranges, choices = stats.auto_ranges(protect_bounded=False)
    assert choices["h"] != "minmax" and ranges["h"][1] < 12.0


def test_auto_protects_head_tensors_and_named_tensors():
    m = _model(_SCALE, _consts())
    stats = collect_calibration_stats(m, _heavy_tailed(2), tensor_names=["h", "y"])
    assert stats.head_tensors(1) == {"h", "c"}
    _, choices = stats.auto_ranges(protect_head_depth=1)
    assert choices["h"] == "minmax (protected)"
    _, choices = stats.auto_ranges(minmax_tensor_names=["h"])
    assert choices["h"] == "minmax (protected)"


def test_percentile_spelling_and_stats_ranges_match_calibrate():
    m = _model(_SCALE, _consts())
    data = _heavy_tailed(2)
    stats = collect_calibration_stats(m, data, tensor_names=["h"])
    for method in ("minmax", "mse", "entropy", "percentile"):
        assert stats.ranges(method) == onnxsim.calibrate(
            m, data, method=method, tensor_names=["h"]
        )
    assert stats.ranges("percentile:99.9") == stats.ranges(
        "percentile", percentile=99.9
    )
    with pytest.raises(ValueError):
        stats.ranges("median")
    with pytest.raises(ValueError):
        collect_calibration_stats(m, data, tensor_names=["h"], histograms=False).ranges(
            "mse"
        )


def test_quantize_static_auto_full_graph_runs():
    m = _model(_SIGMOID, _consts())
    data = _heavy_tailed(2, value=12.0)
    q = onnxsim.quantize_static(m, data, method="auto", full_graph=True)
    assert any(n.op_type == "QuantizeLinear" for n in q.graph.node)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort.InferenceSession(q.SerializeToString(), so).run(None, data[0])


# Detector-like: y is a "score map" of background noise with one rare
# detection (12) per 1M values. Clipping at the 99.99th percentile is the
# better trade by SQNR (one clipped value vs a ~4x finer grid for all the
# others) but erases every detection.
def _detector_case():
    m = _model(_SCALE, _consts())
    return m, _heavy_tailed(4, value=12.0, seed=2), _heavy_tailed(3, value=12.0, seed=3)


def _detections_kept(float_out, quant_out):
    kept = 0
    for f, q in zip(float_out, quant_out):
        hit = f["y"] > 8.0
        kept += int(np.sum(np.abs(q["y"][hit] - f["y"][hit]) < 1.0))
    return float(kept)


def test_pick_follows_the_metric_not_rel_l2():
    m, calib, held_out = _detector_case()
    candidates = ("minmax", "percentile:99.99")
    by_sqnr = onnxsim.pick_calibration(
        m, calib, held_out, candidates=candidates, full_graph=True
    )
    assert by_sqnr.method == "percentile:99.99"
    by_task = onnxsim.pick_calibration(
        m,
        calib,
        held_out,
        metric=_detections_kept,
        candidates=candidates,
        full_graph=True,
    )
    assert by_task.method == "minmax"
    assert by_task.scores == {"minmax": 3.0, "percentile:99.99": 0.0}
    assert by_task.model is not None and by_task.ranges


def test_pick_calibrates_once(monkeypatch):
    m, calib, held_out = _detector_case()
    calib_ids = {id(b) for b in calib}
    runs = {"calib": 0}
    real_run = ort.InferenceSession.run

    def counting_run(self, output_names, feed, *a, **k):
        if id(feed) in calib_ids:
            runs["calib"] += 1
        return real_run(self, output_names, feed, *a, **k)

    monkeypatch.setattr(ort.InferenceSession, "run", counting_run)
    pick = onnxsim.pick_calibration(m, calib, held_out, full_graph=True)
    assert len(pick.scores) == len(onnxsim.calibration_pick.DEFAULT_PICK_CANDIDATES)
    # the min/max pass and the histogram pass, shared by all six candidates
    assert runs["calib"] == 2 * len(calib)
    assert pick.auto_choices  # "auto" was a candidate: its choices are kept
