"""Tests for ``onnxsim.quantize_weight_only_icquant`` -- see
``onnxsim/icquant.py`` for the technique: per-group outlier-aware
block-wise INT4 quantization, like ``onnxsim.spqr``, but communicating
each group's chosen outlier positions via a combinadic ("index coding")
rank instead of a dense bitmask or an explicit index list.
"""

import itertools
import math

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.icquant import _combinadic_rank, _combinadic_unrank

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


# --- Combinadic rank/unrank -------------------------------------------------


@pytest.mark.parametrize("n", [4, 5, 6, 8])
def test_combinadic_rank_unrank_is_a_bijection(n):
    for k in range(1, n + 1):
        combos = list(itertools.combinations(range(n), k))
        ranks = [_combinadic_rank(c, n) for c in combos]
        # Every C(n, k) combination gets a distinct rank covering
        # [0, C(n, k)) exactly -- no rank is skipped or reused.
        assert sorted(ranks) == list(range(math.comb(n, k)))
        for combo, rank in zip(combos, ranks):
            assert _combinadic_unrank(rank, k, n) == list(combo)


def test_combinadic_k1_rank_is_the_index_itself():
    # A 1-of-n subset's combinadic rank degenerates to the chosen index --
    # a sanity check on the general bijection's base case.
    for i in range(10):
        assert _combinadic_rank([i], 10) == i
        assert _combinadic_unrank(i, 1, 10) == [i]


# --- Metadata bit cost -------------------------------------------------------


def test_icquant_metadata_bits_matches_paper_claim():
    # group_size=32, k=1: C(32, 1) = 32 -> 5 bits/group = 0.15625 bits/element.
    stats = onnxsim.icquant_metadata_bits(group_size=32, num_outliers=1)
    assert stats["combinadic_bits"] == 5
    assert stats["combinadic_bits_per_element"] == pytest.approx(5 / 32)
    assert stats["bitmask_bits_per_element"] == pytest.approx(1.0)

    # group_size=32, k=2: C(32, 2) = 496 -> ceil(log2(496)) = 9 bits/group
    # = 0.28125 bits/element, matching the paper's own reported ~0.3
    # bits/element overhead (vs. a naive ~1 bit/element bitmask).
    stats2 = onnxsim.icquant_metadata_bits(group_size=32, num_outliers=2)
    assert math.comb(32, 2) == 496
    assert stats2["combinadic_bits"] == 9
    assert stats2["combinadic_bits_per_element"] == pytest.approx(9 / 32)
    assert stats2["combinadic_bits_per_element"] < 0.3
    # Both naive alternatives cost strictly more per element than the
    # combinadic encoding -- ICQuant's own point.
    assert stats2["combinadic_bits"] < stats2["bitmask_bits"]
    assert stats2["combinadic_bits"] < stats2["index_list_bits"]


def test_icquant_metadata_bits_zero_outliers_is_free():
    stats = onnxsim.icquant_metadata_bits(group_size=32, num_outliers=0)
    assert stats["combinadic_bits"] == 0
    assert stats["index_list_bits"] == 0


# --- End-to-end quantization -------------------------------------------------
#
# quantize_weight_only_icquant now delegates to the verified C++ port
# (apply_icquant_cpp), which hardcodes group_size=32/num_outliers=1 and
# folds the round trip directly into a replacement float32 initializer
# instead of building a real INT4/DequantizeLinear/ScatterND graph rewrite
# -- see onnxsim/icquant.py's own docstring. The detailed algorithmic
# properties (outlier exact reconstruction, grid size, error reduction vs.
# a naive single-scale fit, opset-independence) are already covered end to
# end against apply_icquant_cpp directly in tests/test_icquant_cpp.py; the
# tests below only exercise the thin wrapper itself: parameter validation
# and basic delegation sanity.


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_icquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.quantize_weight_only_icquant(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == (32, 8)
    assert new_w.dtype == np.float32

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_icquant_declines_when_k_not_divisible_by_group_size():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 32
    q = onnxsim.quantize_weight_only_icquant(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_icquant_rejects_non_default_group_size():
    model = _matmul_model(K=32, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_icquant(model, group_size=8)


def test_icquant_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_icquant(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_icquant_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_icquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_icquant_rejects_negative_num_outliers():
    model = _matmul_model(K=32, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_icquant(model, num_outliers=-1)
