"""Tests for the opt-in ``rewrite_scatterelements_to_scatternd`` pass.

Models are built with the ONNX text format parser per CLAUDE.md. A scatter
is only well-defined for in-range, duplicate-free indices, so onnxsim's own
random-input equivalence check (``check_n``) can't be used here -- random
int64 indices are out of range. Instead each rewritten model is run through
ONNX Runtime next to the original on hand-built valid inputs (including
negative indices) and compared bit-exactly.
"""

import collections

import numpy as np
import onnxruntime as ort
import pytest
from onnx import parser

import onnxsim

PASS = "rewrite_scatterelements_to_scatternd"


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _simplify(model):
    sim, _ = onnxsim.simplify(model, check_n=0, extra_optimizers=[PASS])
    return sim, collections.Counter(n.op_type for n in sim.graph.node)


def _assert_rewritten_and_exact(model, feeds_list):
    sim, ops = _simplify(model)
    assert "ScatterElements" not in ops, ops
    assert ops["ScatterND"] == 1, ops
    for feeds in feeds_list:
        for a, b in zip(_run(model, feeds), _run(sim, feeds)):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert np.array_equal(a, b)
    return sim, ops


def _assert_declined(model, feeds=None):
    sim, ops = _simplify(model)
    assert ops["ScatterElements"] == 1, ops
    assert "ScatterND" not in ops, ops
    if feeds is not None:
        for a, b in zip(_run(model, feeds), _run(sim, feeds)):
            assert np.array_equal(a, b)
    return sim, ops


def _row_feeds(t, n, rest, rng, idx_dtype=np.int64, negative=False):
    rows = rng.choice(t, size=n, replace=False)
    if negative:
        rows = rows - t  # same rows, spelled as negative indices
    return {
        "data": rng.standard_normal((t, *rest)).astype(np.float32),
        "idx": rows.astype(idx_dtype),
        "updates": rng.standard_normal((n, *rest)).astype(np.float32),
    }


# torchvision MultiScaleRoIAlign's exported level merge, dynamic row counts:
#   ScatterElements(data, Expand(Reshape(idx, [-1,1,1,1]), Shape(updates)), updates, axis=0)
MASKRCNN_DYNAMIC = """
<ir_version: 8, opset_import: ["": {opset}]>
g (float[T,8,3,3] data, {itype}[N] idx, float[N,8,3,3] updates) => (float[T,8,3,3] Y)
<int64[4] s = {{-1, 1, 1, 1}}>
{{
    r = Reshape(idx, s)
    shp = Shape(updates)
    ind = Expand(r, shp)
    Y = ScatterElements<axis = {axis}>(data, ind, updates)
}}
"""


@pytest.mark.parametrize("opset", [11, 13, 16, 18])
def test_maskrcnn_roialign_merge_pattern(opset):
    model = parser.parse_model(
        MASKRCNN_DYNAMIC.format(opset=opset, itype="int64", axis=0)
    )
    rng = np.random.default_rng(0)
    feeds = [_row_feeds(10, n, (8, 3, 3), rng) for n in (1, 4, 10)]
    sim, ops = _assert_rewritten_and_exact(model, feeds)
    # The full-size int64 index tensor is no longer materialized: the only
    # Expand left broadcasts the [n,1] row index, never to updates' shape.
    for node in sim.graph.node:
        if node.op_type == "Expand":
            assert node.input[0] != "r"


def test_maskrcnn_constantofshape_merge_buffer():
    # The real export: the merge buffer is ConstantOfShape of a shape vector
    # whose trailing entries are constants. Shape inference can't recover
    # data's dims from that, so the pass proves them from the Concat instead.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 12]>
        g (int64[1] t, int64[N] idx, float[N,8,3,3] updates) => (float[T,8,3,3] Y)
        <int64[4] s = {-1, 1, 1, 1}, int64[1] c = {8}, int64[1] h = {3}, int64[1] w = {3}>
        {
            dshape = Concat<axis = 0>(t, c, h, w)
            data = ConstantOfShape<value = float[1] {0.0}>(dshape)
            r = Reshape(idx, s)
            shp = Shape(updates)
            ind = Expand(r, shp)
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    rng = np.random.default_rng(11)
    feeds = []
    for n in (1, 4):
        f = _row_feeds(10, n, (8, 3, 3), rng)
        del f["data"]
        f["t"] = np.array([10], np.int64)
        feeds.append(f)
    _assert_rewritten_and_exact(model, feeds)


def test_maskrcnn_chained_level_merge():
    # torchvision scatters each FPN level into the previous level's result;
    # every scatter in the chain must be rewritten, with data's trailing
    # dims followed back to the ConstantOfShape.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 12]>
        g (int64[1] t, int64[A] ia, float[A,8,3,3] ua, int64[B] ib, float[B,8,3,3] ub,
           int64[C] ic, float[C,8,3,3] uc) => (float[T,8,3,3] Y)
        <int64[4] s = {-1, 1, 1, 1}, int64[1] c = {8}, int64[1] h = {3}, int64[1] w = {3}>
        {
            dshape = Concat<axis = 0>(t, c, h, w)
            d0 = ConstantOfShape<value = float[1] {0.0}>(dshape)
            ra = Reshape(ia, s)
            sa = Shape(ua)
            xa = Expand(ra, sa)
            d1 = ScatterElements<axis = 0>(d0, xa, ua)
            rb = Reshape(ib, s)
            sb = Shape(ub)
            xb = Expand(rb, sb)
            d2 = ScatterElements<axis = 0>(d1, xb, ub)
            rc = Reshape(ic, s)
            sc = Shape(uc)
            xc = Expand(rc, sc)
            Y = ScatterElements<axis = 0>(d2, xc, uc)
        }
        """
    )
    sim, ops = _simplify(model)
    assert "ScatterElements" not in ops and ops["ScatterND"] == 3, ops
    rng = np.random.default_rng(12)
    rows = rng.permutation(10)
    feeds = {"t": np.array([10], np.int64)}
    for name, lo, hi in (("a", 0, 3), ("b", 3, 7), ("c", 7, 10)):
        n = hi - lo
        feeds[f"i{name}"] = rows[lo:hi].astype(np.int64)
        feeds[f"u{name}"] = rng.standard_normal((n, 8, 3, 3)).astype(np.float32)
    for a, b in zip(_run(model, feeds), _run(sim, feeds)):
        assert np.array_equal(a, b)


def test_declines_constantofshape_with_mismatched_trailing_dim():
    # Same, but the buffer is wider (W=4) than updates (W=3): a sub-block
    # scatter, must be left alone.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 12]>
        g (int64[1] t, int64[N] idx, float[N,8,3,3] updates) => (float[T,8,3,4] Y)
        <int64[4] s = {-1, 1, 1, 1}, int64[1] c = {8}, int64[1] h = {3}, int64[1] w = {4}>
        {
            dshape = Concat<axis = 0>(t, c, h, w)
            data = ConstantOfShape<value = float[1] {0.0}>(dshape)
            r = Reshape(idx, s)
            shp = Shape(updates)
            ind = Expand(r, shp)
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    _assert_declined(model)


def test_negative_indices():
    model = parser.parse_model(MASKRCNN_DYNAMIC.format(opset=13, itype="int64", axis=0))
    rng = np.random.default_rng(1)
    feeds = [_row_feeds(10, 5, (8, 3, 3), rng, negative=True)]
    _assert_rewritten_and_exact(model, feeds)


def test_int32_indices_are_cast():
    model = parser.parse_model(MASKRCNN_DYNAMIC.format(opset=13, itype="int32", axis=0))
    rng = np.random.default_rng(2)
    feeds = [_row_feeds(10, 6, (8, 3, 3), rng, idx_dtype=np.int32)]
    _, ops = _assert_rewritten_and_exact(model, feeds)
    assert ops["Cast"] == 1


def test_negative_axis_equal_to_zero():
    model = parser.parse_model(
        MASKRCNN_DYNAMIC.format(opset=13, itype="int64", axis=-4)
    )
    rng = np.random.default_rng(3)
    _assert_rewritten_and_exact(model, [_row_feeds(10, 3, (8, 3, 3), rng)])


def test_static_shapes_need_no_row_broadcast():
    # Row count statically equal on both sides: the [n,1] index is used as-is,
    # no Shape/Expand side graph is emitted.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[6,4] data, int64[3] idx, float[3,4] updates) => (float[6,4] Y)
        <int64[2] s = {-1, 1}, int64[2] shp = {3, 4}>
        {
            r = Reshape(idx, s)
            ind = Expand(r, shp)
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    rng = np.random.default_rng(4)
    sim, ops = _assert_rewritten_and_exact(model, [_row_feeds(6, 3, (4,), rng)])
    assert "Expand" not in ops and "Shape" not in ops, ops


def test_lower_rank_expand_source_broadcasts_single_row():
    # X has rank 1 < r: Expand left-pads it, so the same index lands on every
    # row -- only valid (duplicate-free) for a single row, which is what the
    # rewrite must reproduce via its [1,1] -> [N,1] broadcast.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[5,4] data, int64[1] idx, float[N,4] updates) => (float[5,4] Y)
        {
            shp = Shape(updates)
            ind = Expand(idx, shp)
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    rng = np.random.default_rng(5)
    feeds = {
        "data": rng.standard_normal((5, 4)).astype(np.float32),
        "idx": np.array([3], np.int64),
        "updates": rng.standard_normal((1, 4)).astype(np.float32),
    }
    _assert_rewritten_and_exact(model, [feeds])


def test_constant_invariant_indices():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[5,3] data, float[2,3] updates) => (float[5,3] Y)
        <int64[2,3] ind = {4, 4, 4, -5, -5, -5}>
        {
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    rng = np.random.default_rng(6)
    feeds = {
        "data": rng.standard_normal((5, 3)).astype(np.float32),
        "updates": rng.standard_normal((2, 3)).astype(np.float32),
    }
    _assert_rewritten_and_exact(model, [feeds])


# ------------------------------- near misses ------------------------------- #


def test_declines_constant_indices_varying_along_axis1():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[5,3] data, float[2,3] updates) => (float[5,3] Y)
        <int64[2,3] ind = {4, 1, 4, 0, 0, 2}>
        {
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    rng = np.random.default_rng(7)
    feeds = {
        "data": rng.standard_normal((5, 3)).astype(np.float32),
        "updates": rng.standard_normal((2, 3)).astype(np.float32),
    }
    _assert_declined(model, feeds)


def test_declines_expand_source_varying_along_axis1():
    # X is [N,4,1]: it genuinely varies along axis 1, so rows are not uniform.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[9,4,2] data, int64[N,4] idx, float[N,4,2] updates) => (float[9,4,2] Y)
        <int64[3] s = {-1, 4, 1}>
        {
            r = Reshape(idx, s)
            shp = Shape(updates)
            ind = Expand(r, shp)
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    _assert_declined(model)


def test_declines_runtime_indices_without_expand():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[5,3] data, int64[2,3] ind, float[2,3] updates) => (float[5,3] Y)
        {
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    _assert_declined(model)


def test_declines_updates_smaller_than_data_along_other_axis():
    # ScatterElements may scatter a sub-block (updates narrower than data on
    # axis 1); ScatterND cannot, so this must be left alone.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 13]>
        g (float[6,4] data, int64[3] idx, float[3,2] updates) => (float[6,4] Y)
        <int64[2] s = {-1, 1}, int64[2] shp = {3, 2}>
        {
            r = Reshape(idx, s)
            ind = Expand(r, shp)
            Y = ScatterElements<axis = 0>(data, ind, updates)
        }
        """
    )
    rng = np.random.default_rng(8)
    feeds = {
        "data": rng.standard_normal((6, 4)).astype(np.float32),
        "idx": np.array([5, 0, 2], np.int64),
        "updates": rng.standard_normal((3, 2)).astype(np.float32),
    }
    _assert_declined(model, feeds)


def test_declines_nonzero_axis():
    model = parser.parse_model(
        MASKRCNN_DYNAMIC.format(opset=13, itype="int64", axis=1).replace(
            "{-1, 1, 1, 1}", "{1, -1, 1, 1}"
        )
    )
    _assert_declined(model)


@pytest.mark.parametrize(
    "reduction,opset", [("add", 16), ("mul", 16), ("max", 18), ("min", 18)]
)
def test_declines_reductions(reduction, opset):
    body = MASKRCNN_DYNAMIC.format(opset=opset, itype="int64", axis=0).replace(
        "<axis = 0>", f'<axis = 0, reduction = "{reduction}">'
    )
    model = parser.parse_model(body)
    rng = np.random.default_rng(9)
    _assert_declined(model, _row_feeds(10, 4, (8, 3, 3), rng))


def test_explicit_reduction_none_rewrites():
    body = MASKRCNN_DYNAMIC.format(opset=16, itype="int64", axis=0).replace(
        "<axis = 0>", '<axis = 0, reduction = "none">'
    )
    model = parser.parse_model(body)
    rng = np.random.default_rng(10)
    _assert_rewritten_and_exact(model, [_row_feeds(10, 4, (8, 3, 3), rng)])


def test_not_run_by_default():
    model = parser.parse_model(MASKRCNN_DYNAMIC.format(opset=13, itype="int64", axis=0))
    sim, _ = onnxsim.simplify(model, check_n=0)
    ops = collections.Counter(n.op_type for n in sim.graph.node)
    assert ops["ScatterElements"] == 1 and "ScatterND" not in ops
