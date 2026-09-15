"""Tests for the opt-in ``rewrite_tensor_scatter_to_scatter_elements`` pass.

Every model is a single ``TensorScatter`` node (opset 24), built with the ONNX
text format parser (``onnx.parser``) per CLAUDE.md's convention for this
repo's tests. Rewriting models are run through
``onnxsim.simplify(..., extra_optimizers=["rewrite_tensor_scatter_to_scatter_elements"])``
and equivalence-checked numerically against the original ``TensorScatter``
node (onnxsim's own ``check_ok``, backed here by onnx's reference evaluator
since onnxruntime doesn't have a ``TensorScatter`` kernel yet), mirroring
``tests/test_gatherelements_to_gather.py``'s ``_simplify_and_check`` idiom.
"""

import collections

import numpy as np
from onnx import parser

import onnxsim


def _simplify_and_check(model, input_data, check_n=3):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_tensor_scatter_to_scatter_elements"],
    )
    assert check_ok, "rewritten graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "TensorScatter" not in op_types, op_types
    assert "ScatterElements" in op_types, op_types
    return sim_model, op_types


def _simplify_and_assert_declined(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_tensor_scatter_to_scatter_elements"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "TensorScatter" in op_types, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# Default axis (-2, normalizing to 1 for this rank-3 model) and default mode
# ("linear"), with `write_indices` supplied. `write_indices + sequence_length
# <= max_sequence_length` for every batch item (1+2<=6, 3+2<=6), as required
# by "linear" mode.
# --------------------------------------------------------------------------- #


def test_default_axis_linear_mode_with_write_indices():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[2,6,3] past_cache, float[2,2,3] update, int64[2] write_indices)
      => (float[2,6,3] Y)
    {
      Y = TensorScatter (past_cache, update, write_indices)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(0)
    data = {
        "past_cache": rng.randn(2, 6, 3).astype(np.float32),
        "update": rng.randn(2, 2, 3).astype(np.float32),
        "write_indices": np.array([1, 3], dtype=np.int64),
    }
    _simplify_and_check(model, data)


# --------------------------------------------------------------------------- #
# `write_indices` omitted entirely -- defaults to all-zero per the op's own
# doc, so this also exercises the pass's "no write_indices input" branch
# (the rewrite must not try to read a 3rd input that doesn't exist).
# --------------------------------------------------------------------------- #


def test_missing_write_indices_defaults_to_zero():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[2,5,3] past_cache, float[2,2,3] update) => (float[2,5,3] Y)
    {
      Y = TensorScatter (past_cache, update)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(1)
    data = {
        "past_cache": rng.randn(2, 5, 3).astype(np.float32),
        "update": rng.randn(2, 2, 3).astype(np.float32),
    }
    _simplify_and_check(model, data)


# --------------------------------------------------------------------------- #
# "circular" mode: write_indices chosen so the write wraps around
# max_sequence_length for at least one batch item (batch 0: [3,4] -> wraps to
# [3,0]; batch 1: [0,1], no wraparound) -- exercises the pass's `Mod` step.
# --------------------------------------------------------------------------- #


def test_circular_mode_wraps_around():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[2,4,3] past_cache, float[2,2,3] update, int64[2] write_indices)
      => (float[2,4,3] Y)
    {
      Y = TensorScatter <mode = "circular"> (past_cache, update, write_indices)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(2)
    data = {
        "past_cache": rng.randn(2, 4, 3).astype(np.float32),
        "update": rng.randn(2, 2, 3).astype(np.float32),
        "write_indices": np.array([3, 0], dtype=np.int64),
    }
    _simplify_and_check(model, data)


# --------------------------------------------------------------------------- #
# Minimal rank-2 case: axis must be explicit (default -2 normalizes to 0,
# which is invalid -- the batch axis can never be `axis`), so axis=1 (== -1)
# is spelled out. No leading dims between batch and axis at all.
# --------------------------------------------------------------------------- #


def test_rank2_explicit_axis():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[3,5] past_cache, float[3,2] update, int64[3] write_indices)
      => (float[3,5] Y)
    {
      Y = TensorScatter <axis = 1> (past_cache, update, write_indices)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(3)
    data = {
        "past_cache": rng.randn(3, 5).astype(np.float32),
        "update": rng.randn(3, 2).astype(np.float32),
        "write_indices": np.array([0, 1, 2], dtype=np.int64),
    }
    _simplify_and_check(model, data)


# --------------------------------------------------------------------------- #
# Negative axis (-1, the last dim) on a rank-3 model, with a non-trivial
# "leading" dim (dim 1, size 3) between the batch axis and `axis`. Per the
# op's own pseudocode, `write_indices` depends only on `prefix_idx[0]` (the
# batch coordinate), so every position along dim 1 must be scattered
# identically regardless of its own coordinate -- this is what exercises the
# rewrite's broadcast (rather than a per-position) construction of the index
# tensor.
# --------------------------------------------------------------------------- #


def test_negative_axis_with_leading_dim():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[2,3,4] past_cache, float[2,3,2] update, int64[2] write_indices)
      => (float[2,3,4] Y)
    {
      Y = TensorScatter <axis = -1> (past_cache, update, write_indices)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(4)
    data = {
        "past_cache": rng.randn(2, 3, 4).astype(np.float32),
        "update": rng.randn(2, 3, 2).astype(np.float32),
        "write_indices": np.array([0, 2], dtype=np.int64),
    }
    _simplify_and_check(model, data)


# --------------------------------------------------------------------------- #
# Fully dynamic (symbolic) shapes: the pass needs only *rank* to be known
# statically (to build the reshape/broadcast literals), not any concrete
# dimension size -- everything else is read at runtime via Shape/Gather. This
# checks the rewrite actually fires (and stays correct) without any static
# dims at all.
# --------------------------------------------------------------------------- #


def test_fully_dynamic_shapes_still_rewrites():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[batch,max_seq,dim] past_cache, float[batch,seq,dim] update,
            int64[batch] write_indices) => (float[batch,max_seq,dim] Y)
    {
      Y = TensorScatter (past_cache, update, write_indices)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(5)
    data = {
        "past_cache": rng.randn(2, 6, 3).astype(np.float32),
        "update": rng.randn(2, 2, 3).astype(np.float32),
        "write_indices": np.array([1, 3], dtype=np.int64),
    }
    _simplify_and_check(model, data)


# --------------------------------------------------------------------------- #
# Rank not statically known at all -- the predicate must decline (it has no
# rank to build the reshape/broadcast literals against). A plain unranked
# graph input doesn't work: ONNX's own checker requires every main-graph
# input/output to declare a `shape` field (confirmed empirically -- see
# `tests/test_formal_verify_fuse_consecutive_unsqueezes.py`'s own note on
# this exact restriction). Instead, `past_cache`/`update` are produced by
# `Squeeze`ing an extra trailing size-1 dim off statically-shaped graph
# inputs using an `axes` value that is itself a genuine runtime graph
# input (not derivable from any constant, unlike an `Add` of two
# initializers -- onnxsim's own shape inference does partial constant-data
# propagation and resolves that case anyway) -- no shape inference can
# determine which axis a truly runtime-valued `Squeeze` removes, so the
# squeezed output's rank is genuinely unresolvable at graph-simplification
# time even though its runtime value never varies for this test's fixed
# `input_data`.
# --------------------------------------------------------------------------- #


def test_unknown_rank_declines():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 24]
    >
    agraph (float[2,6,3,1] past_cache_raw, float[2,2,3,1] update_raw,
            int64[2] write_indices, int64[1] axes) => (float[2,6,3] Y)
    {
      past_cache = Squeeze (past_cache_raw, axes)
      update = Squeeze (update_raw, axes)
      Y = TensorScatter (past_cache, update, write_indices)
    }
    """
    model = parser.parse_model(body)
    rng = np.random.RandomState(6)
    data = {
        "past_cache_raw": rng.randn(2, 6, 3, 1).astype(np.float32),
        "update_raw": rng.randn(2, 2, 3, 1).astype(np.float32),
        "write_indices": np.array([1, 3], dtype=np.int64),
        "axes": np.array([3], dtype=np.int64),
    }
    _simplify_and_assert_declined(model, data)
