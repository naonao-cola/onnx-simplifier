import numpy as np
import pytest
from onnx import numpy_helper, parser

import onnxsim.activation_sensitivity as A
from onnxsim.activation_sensitivity import (
    analyze_activation_sensitivity,
    group_nodes,
    search_activation_precision_for_budget,
)
from onnxsim.calibration_pick import worst_output_sqnr


def _model(body, initializer=(), opset=17):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializer)
    return model


def _two_blocks():
    """Two parallel blocks. Block 1's hidden tensor carries one channel ~1000x
    the others, which its projection then ignores: a per-tensor uint8 range
    sized for the outlier crushes the channels the output is made of."""
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((8, 8)).astype(np.float32)
    w1 = rng.standard_normal((8, 8)).astype(np.float32)
    w1[:, 0] *= 1000.0
    p1 = rng.standard_normal((8, 8)).astype(np.float32)
    p1[0, :] = 0.0  # the projection drops the outlier channel
    p0 = rng.standard_normal((8, 8)).astype(np.float32)
    return _model(
        """g (float[16,8] x) => (float[16,8] y0, float[16,8] y1) {
            ["/blocks.0/fc/MatMul"] h0 = MatMul(x, w0)
            ["/blocks.0/proj/MatMul"] y0 = MatMul(h0, p0)
            ["/blocks.1/fc/MatMul"] h1 = MatMul(x, w1)
            ["/blocks.1/proj/MatMul"] y1 = MatMul(h1, p1)
        }""",
        [
            numpy_helper.from_array(w0, "w0"),
            numpy_helper.from_array(p0, "p0"),
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(p1, "p1"),
        ],
    )


def _data(n, seed):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((16, 8)).astype(np.float32)} for _ in range(n)]


CAL, EV = _data(6, 1), _data(3, 2)


def test_groups_by_prefix_op_type_node_and_windows():
    m = _two_blocks()
    assert group_nodes(m) == {
        "/blocks.0": ["/blocks.0/fc/MatMul", "/blocks.0/proj/MatMul"],
        "/blocks.1": ["/blocks.1/fc/MatMul", "/blocks.1/proj/MatMul"],
    }
    assert group_nodes(m, block_regex=r"/(fc|proj)/") == {
        "fc": ["/blocks.0/fc/MatMul", "/blocks.1/fc/MatMul"],
        "proj": ["/blocks.0/proj/MatMul", "/blocks.1/proj/MatMul"],
    }
    assert list(group_nodes(m, "op_type")) == ["MatMul"]
    assert len(group_nodes(m, "node")) == 4
    # unnamed nodes (keyed by their output) fall into topological windows
    u = _model(
        """g (float[4,4] x) => (float[4,4] z) {
            a = Relu(x)
            b = Sigmoid(a)
            z = Tanh(b)
        }"""
    )
    assert group_nodes(u, window=2) == {"window:a": ["a", "b"], "window:z": ["z"]}
    with pytest.raises(ValueError):
        group_nodes(m, {"g": ["nope"]})


def test_sensitivity_ranks_the_outlier_block_first_and_calibrates_once(monkeypatch):
    calls = []
    real = A.collect_calibration_stats
    monkeypatch.setattr(
        A, "collect_calibration_stats", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    for mode in ("only", "all_but"):
        rep = analyze_activation_sensitivity(_two_blocks(), CAL, EV, mode=mode)
        assert [r.group for r in rep.groups][0] == "/blocks.1", mode
        assert rep.groups[0].sensitivity > rep.groups[1].sensitivity
    assert len(calls) == 2  # one calibration per analysis, none per variant
    rep = analyze_activation_sensitivity(_two_blocks(), CAL, EV, mode="only")
    worst = rep.groups[0]
    assert worst.nodes == ["/blocks.1/fc/MatMul", "/blocks.1/proj/MatMul"]
    assert worst.delta_vs_float < 0


def test_search_promotes_exactly_the_outlier_block_and_one_calibration(monkeypatch):
    calls = []
    real = A.collect_calibration_stats
    monkeypatch.setattr(
        A, "collect_calibration_stats", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    res = search_activation_precision_for_budget(_two_blocks(), CAL, EV, budget=20.0)
    assert len(calls) == 1  # the ranking reuses the search's calibration
    assert res.meets_budget and res.score >= 20.0
    assert {g for g, lv in res.levels.items() if lv != "uint8"} == {"/blocks.1"}
    assert res.levels["/blocks.0"] == "uint8"
    assert res.promoted_nodes == 2
    assert [t[0] for t in res.trace[1:]] == ["/blocks.1"] * (len(res.trace) - 1)
    # the returned policy reproduces the returned model's behavior
    from onnxsim.calibration_pick import run_outputs
    from onnxsim.full_qdq import quantize_full_qdq

    again = quantize_full_qdq(
        _two_blocks(),
        CAL,
        exclude_nodes=res.exclude_nodes,
        tensor_dtypes=res.tensor_dtypes,
    )
    a, b = run_outputs(res.model, EV), run_outputs(again, EV)
    for x, y in zip(a, b):
        for k in x:
            np.testing.assert_allclose(x[k], y[k], rtol=1e-5, atol=1e-5)


def test_search_stops_at_the_first_step_meeting_the_budget():
    full = search_activation_precision_for_budget(
        _two_blocks(), CAL, EV, budget=1e9, rerank=True
    )
    assert not full.meets_budget
    assert all(lv == "float" for lv in full.levels.values())
    first = full.trace[1][2]
    res = search_activation_precision_for_budget(
        _two_blocks(), CAL, EV, budget=first, rerank=True
    )
    assert res.meets_budget and len(res.trace) == 2
    # already within budget: no promotion at all
    start = full.trace[0][2]
    res0 = search_activation_precision_for_budget(_two_blocks(), CAL, EV, budget=start)
    assert res0.meets_budget and len(res0.trace) == 1 and res0.promoted_nodes == 0


def test_custom_metric_is_honored_over_sqnr():
    def y0_only(f, q):
        return worst_output_sqnr(
            [{"y0": b["y0"]} for b in f], [{"y0": b["y0"]} for b in q]
        )

    rep = analyze_activation_sensitivity(_two_blocks(), CAL, EV, metric=y0_only)
    # quantizing block 1 only touches y0 through the shared input x's Q/DQ,
    # so under a y0-only metric block 0 is the one that hurts
    assert rep.groups[0].group == "/blocks.0"
    assert rep.groups[1].score > rep.groups[0].score + 10


def test_uint16_and_cost_hints():
    m = _two_blocks()
    rep = analyze_activation_sensitivity(
        m, CAL, EV, activation_dtypes=("uint8", "uint16")
    )
    by = {(r.group, r.dtype): r for r in rep.groups}
    assert by[("/blocks.1", "uint16")].score > by[("/blocks.1", "uint8")].score
    costs = A.estimate_group_macs(m, group_nodes(m))
    assert costs["/blocks.0"] == costs["/blocks.1"] == 2 * 16 * 8 * 8
