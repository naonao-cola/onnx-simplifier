"""Tests for the opt-in ``rewrite_gathernd_to_gather`` pass.

Every model is a single ``GatherND`` node, built with the ONNX text format
parser (``onnx.parser``) per CLAUDE.md's convention for this repo's tests.
Rewriting models are run through ``onnxsim.simplify(...,
extra_optimizers=["rewrite_gathernd_to_gather"])``, which numerically
equivalence-checks the rewritten graph against the original ``GatherND`` node
(via onnxruntime, or the onnx reference evaluator when onnxruntime is not
installed) using fixed, concrete ``input_data`` for both ``data`` and
``indices`` -- see ``tests/test_gridsample_to_gather.py``'s
``_simplify_and_check`` helper, which this mirrors.
"""

import collections

import numpy as np
from onnx import parser

import onnxsim


def _model(data_shape, indices_shape, out_shape, batch_dims, opset=17, ir_version=10):
    body = f"""
    <
      ir_version: {ir_version},
      opset_import: ["": {opset}]
    >
    agraph (float{data_shape} data, int64{indices_shape} indices) => (float{out_shape} Y)
    {{
      Y = GatherND <batch_dims={batch_dims}> (data, indices)
    }}
    """
    return parser.parse_model(body)


def _simplify_and_check(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gathernd_to_gather"],
    )
    assert check_ok, "rewritten graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "GatherND" not in op_types, op_types
    assert "Gather" in op_types, op_types
    return sim_model, op_types


def _simplify_and_assert_declined(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gathernd_to_gather"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "GatherND" in op_types, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# batch_dims=0, k=1 -- the trivial case: degenerates to
# Gather(data, Squeeze(indices, -1), axis=0).
# --------------------------------------------------------------------------- #


def test_trivial_batch_dims0_k1():
    rng = np.random.RandomState(0)
    data = rng.randn(5, 3).astype(np.float32)
    indices = rng.randint(0, 5, size=(4, 1)).astype(np.int64)
    model = _model("[5,3]", "[4,1]", "[4,3]", batch_dims=0)
    _simplify_and_check(model, {"data": data, "indices": indices})


# --------------------------------------------------------------------------- #
# batch_dims=1, k=1
# --------------------------------------------------------------------------- #


def test_batch_dims1_k1():
    rng = np.random.RandomState(1)
    data = rng.randn(2, 5, 3).astype(np.float32)
    indices = rng.randint(0, 5, size=(2, 4, 1)).astype(np.int64)
    model = _model("[2,5,3]", "[2,4,1]", "[2,4,3]", batch_dims=1)
    _simplify_and_check(model, {"data": data, "indices": indices})


# --------------------------------------------------------------------------- #
# batch_dims=1, k=2 -- mirrors what rewrite_gridsample_to_gather emits:
# data transposed to (N,H,W,C), indices (N,Hout,Wout,2) of (iy,ix) pairs.
# --------------------------------------------------------------------------- #


def test_batch_dims1_k2_gridsample_like():
    rng = np.random.RandomState(2)
    N, H, W, C = 2, 5, 7, 3
    Hout, Wout = 4, 6
    data = rng.randn(N, H, W, C).astype(np.float32)
    iy = rng.randint(0, H, size=(N, Hout, Wout, 1)).astype(np.int64)
    ix = rng.randint(0, W, size=(N, Hout, Wout, 1)).astype(np.int64)
    indices = np.concatenate([iy, ix], axis=-1)
    model = _model(
        f"[{N},{H},{W},{C}]",
        f"[{N},{Hout},{Wout},2]",
        f"[{N},{Hout},{Wout},{C}]",
        batch_dims=1,
    )
    _simplify_and_check(model, {"data": data, "indices": indices})


# --------------------------------------------------------------------------- #
# Negative indices -- GatherND's own spec permits
# -data_shape[i] <= indices[...,i] <= data_shape[i]-1; each jointly-indexed
# column must be normalized to non-negative before being combined via
# integer strides, so this exercises that normalization path directly.
# --------------------------------------------------------------------------- #


def test_negative_indices():
    rng = np.random.RandomState(3)
    N, H, W, C = 2, 5, 7, 3
    Hout, Wout = 4, 6
    data = rng.randn(N, H, W, C).astype(np.float32)
    iy = rng.randint(-H, H, size=(N, Hout, Wout, 1)).astype(np.int64)
    ix = rng.randint(-W, W, size=(N, Hout, Wout, 1)).astype(np.int64)
    indices = np.concatenate([iy, ix], axis=-1)
    model = _model(
        f"[{N},{H},{W},{C}]",
        f"[{N},{Hout},{Wout},2]",
        f"[{N},{Hout},{Wout},{C}]",
        batch_dims=1,
    )
    _simplify_and_check(model, {"data": data, "indices": indices})


# --------------------------------------------------------------------------- #
# Dynamic (symbolic) batch dimension N -- the pass must derive B from
# Shape(data) at runtime, never assume a static batch size.
# --------------------------------------------------------------------------- #


def test_dynamic_batch_dim():
    rng = np.random.RandomState(4)
    N, H, W, C = 3, 4, 6, 2
    Hout, Wout = 3, 5
    data = rng.randn(N, H, W, C).astype(np.float32)
    iy = rng.randint(0, H, size=(N, Hout, Wout, 1)).astype(np.int64)
    ix = rng.randint(0, W, size=(N, Hout, Wout, 1)).astype(np.int64)
    indices = np.concatenate([iy, ix], axis=-1)
    model = _model(
        f"[N,{H},{W},{C}]",
        f"[N,{Hout},{Wout},2]",
        f"[N,{Hout},{Wout},{C}]",
        batch_dims=1,
    )
    _simplify_and_check(model, {"data": data, "indices": indices})


# --------------------------------------------------------------------------- #
# Declined: indices' last-dim size (k) not statically known.
# --------------------------------------------------------------------------- #


def test_declined_unknown_k():
    data = np.random.RandomState(5).randn(5, 3).astype(np.float32)
    indices = np.array([[0], [1], [2], [3]], dtype=np.int64)
    model = _model("[5,3]", "[4,K]", "[4,3]", batch_dims=0)
    _simplify_and_assert_declined(model, {"data": data, "indices": indices})


# --------------------------------------------------------------------------- #
# Declined: one of the jointly-indexed (flattened) axes of `data` has an
# unknown (symbolic) size, even though k itself is statically known.
# --------------------------------------------------------------------------- #


def test_declined_unknown_flattened_axis():
    data = np.random.RandomState(6).randn(5, 3).astype(np.float32)
    indices = np.array([[0], [1], [2], [3]], dtype=np.int64)
    model = _model("[D,3]", "[4,1]", "[4,3]", batch_dims=0)
    _simplify_and_assert_declined(model, {"data": data, "indices": indices})
