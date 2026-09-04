"""Tests for the opt-in ``rewrite_gatherelements_to_gather`` pass.

Every model is a single ``GatherElements`` node, built with the ONNX text
format parser (``onnx.parser``) per CLAUDE.md's convention for this repo's
tests. A constant, axis-invariant ``indices`` is spelled directly as a text
tensor literal (an initializer with no matching graph input, so it is
``is_constant_initializer``-true) -- CLAUDE.md's "small, fixed/deterministic
constants are fine as text literals" case. Rewriting models are run through
``onnxsim.simplify(..., extra_optimizers=["rewrite_gatherelements_to_gather"])``,
equivalence-checked against the original ``GatherElements`` node using fixed
``input_data`` for ``data``, mirroring
``tests/test_gridsample_to_gather.py``'s ``_simplify_and_check`` idiom.
"""

import collections

import numpy as np
from onnx import parser

import onnxsim


def _simplify_and_check(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gatherelements_to_gather"],
    )
    assert check_ok, "rewritten graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "GatherElements" not in op_types, op_types
    assert "Gather" in op_types, op_types
    return sim_model, op_types


def _simplify_and_assert_declined(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gatherelements_to_gather"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "GatherElements" in op_types, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# Constant, axis-invariant indices (axis=0): every row of `indices` is
# constant across the non-axis (column) dimension, so the elementwise
# GatherElements is really a broadcast of a length-2 index vector -- should
# rewrite to plain Gather.
# --------------------------------------------------------------------------- #


def test_axis_invariant_int64_rewrites():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 17]
    >
    agraph (float[3,4] data) => (float[2,4] Y)
    <int64[2,4] indices = {0,0,0,0, 2,2,2,2}>
    {
      Y = GatherElements <axis=0> (data, indices)
    }
    """
    model = parser.parse_model(body)
    data = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    _simplify_and_check(model, {"data": data})


# --------------------------------------------------------------------------- #
# Same but with int32 indices -- GatherElements' Tind constraint allows
# either int32 or int64.
# --------------------------------------------------------------------------- #


def test_axis_invariant_int32_rewrites():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 17]
    >
    agraph (float[3,4] data) => (float[2,4] Y)
    <int32[2,4] indices = {1,1,1,1, 0,0,0,0}>
    {
      Y = GatherElements <axis=0> (data, indices)
    }
    """
    model = parser.parse_model(body)
    data = np.random.RandomState(1).randn(3, 4).astype(np.float32)
    _simplify_and_check(model, {"data": data})


# --------------------------------------------------------------------------- #
# Negative axis, carried through unchanged: axis=-1 (== axis 1 for rank 2).
# Invariance now means "constant down each column" (fixed non-axis
# coordinate is axis 0, the row).
# --------------------------------------------------------------------------- #


def test_axis_invariant_negative_axis_rewrites():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 17]
    >
    agraph (float[3,4] data) => (float[3,2] Y)
    <int64[3,2] indices = {1,3, 1,3, 1,3}>
    {
      Y = GatherElements <axis=-1> (data, indices)
    }
    """
    model = parser.parse_model(body)
    data = np.random.RandomState(2).randn(3, 4).astype(np.float32)
    _simplify_and_check(model, {"data": data})


# --------------------------------------------------------------------------- #
# Constant indices that IS NOT axis-invariant (varies along the non-axis
# dimension): genuinely elementwise, cannot reduce to plain Gather -- must
# decline, GatherElements survives.
# --------------------------------------------------------------------------- #


def test_non_axis_invariant_declines():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 17]
    >
    agraph (float[3,4] data) => (float[2,4] Y)
    <int64[2,4] indices = {0,1,2,0, 1,0,2,1}>
    {
      Y = GatherElements <axis=0> (data, indices)
    }
    """
    model = parser.parse_model(body)
    data = np.random.RandomState(3).randn(3, 4).astype(np.float32)
    _simplify_and_assert_declined(model, {"data": data})


# --------------------------------------------------------------------------- #
# Dynamic (non-constant, graph-input) indices -- most real GatherElements
# uses look like this; the pass must not attempt to inspect their values and
# must decline unconditionally.
# --------------------------------------------------------------------------- #


def test_dynamic_indices_declines():
    body = """
    <
      ir_version: 10,
      opset_import: ["": 17]
    >
    agraph (float[3,4] data, int64[2,4] indices) => (float[2,4] Y)
    {
      Y = GatherElements <axis=0> (data, indices)
    }
    """
    model = parser.parse_model(body)
    data = np.random.RandomState(4).randn(3, 4).astype(np.float32)
    # Axis-invariant values on purpose -- even though the *values* happen to
    # satisfy the invariance this pass looks for, `indices` being a runtime
    # graph input (not a constant) must still make the pass decline: proving
    # invariance requires known values, not just fortunate ones.
    indices = np.array([[0, 0, 0, 0], [2, 2, 2, 2]], dtype=np.int64)
    _simplify_and_assert_declined(model, {"data": data, "indices": indices})
