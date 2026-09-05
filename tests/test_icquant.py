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


def test_icquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.quantize_weight_only_icquant(model, group_size=8, num_outliers=1)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert "DequantizeLinear" in op_types
    assert "ScatterND" in op_types

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_icquant_outlier_positions_reconstruct_exactly_via_numpy():
    # Verify reconstruction directly against the emitted initializers with
    # numpy (a tight *relative* tolerance), rather than round-tripping
    # through onnxruntime -- onnxruntime's MatMul reduction order isn't
    # bit-exact across CPU architectures, so it isn't the right tool to
    # confirm codes reconstruct the original weight exactly.
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 0.1
    weight[0, 0] = 50.0  # row 0 (of W.T's [N, K] view: N=col 0, K=row 0)
    model = _matmul_model(K=32, N=8, weight=weight)
    q = onnxsim.quantize_weight_only_icquant(model, group_size=8, num_outliers=1)
    onnx.checker.check_model(q)

    init_map = {t.name: onnx.numpy_helper.to_array(t) for t in q.graph.initializer}
    codes_name = next(n for n in init_map if n.endswith("_icquant_codes"))
    scale_name = next(n for n in init_map if n.endswith("_icquant_scale"))
    idx_name = next(n for n in init_map if n.endswith("_icquant_outlier_indices"))
    val_name = next(n for n in init_map if n.endswith("_icquant_outlier_values"))

    codes = init_map[codes_name].astype(np.float64)  # [K, N]
    scale = init_map[scale_name].astype(np.float64)  # [K/group_size, N]
    dequant = codes * np.repeat(scale, 8, axis=0)  # [K, N]

    indices = init_map[idx_name]  # [num_outliers, 2] as [k_pos, n_pos]
    values = init_map[val_name].astype(np.float64)
    dequant[indices[:, 0], indices[:, 1]] = values

    assert np.any((indices[:, 0] == 0) & (indices[:, 1] == 0))
    reconstructed_outlier = dequant[0, 0]
    assert reconstructed_outlier == pytest.approx(50.0, rel=1e-6)

    # Every other element quantized to within its own group's scale/2.
    non_outlier_mask = np.ones_like(dequant, dtype=bool)
    non_outlier_mask[indices[:, 0], indices[:, 1]] = False
    w_kn = weight.astype(np.float64)  # already [K, N]
    err = np.abs(w_kn - dequant)[non_outlier_mask]
    scale_full = np.repeat(scale, 8, axis=0)[non_outlier_mask]
    assert np.all(err <= scale_full / 2 + 1e-9)


def test_icquant_reduces_max_error_vs_zero_outliers():
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 0.1
    weight[0, 0] = 40.0
    weight[5, 3] = -35.0

    model = _matmul_model(K=32, N=8, weight=weight)
    q_plain = onnxsim.quantize_weight_only_icquant(model, group_size=8, num_outliers=0)
    q_ic = onnxsim.quantize_weight_only_icquant(model, group_size=8, num_outliers=1)

    probe = np.eye(32, dtype=np.float32)
    (plain_y,) = _run(q_plain, {"X": probe})
    (ic_y,) = _run(q_ic, {"X": probe})

    w64 = weight.astype(np.float64)
    plain_err = np.abs(w64 - plain_y.astype(np.float64))
    ic_err = np.abs(w64 - ic_y.astype(np.float64))
    assert ic_err.max() < plain_err.max()


def test_icquant_declines_when_k_not_divisible_by_group_size():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 8
    q = onnxsim.quantize_weight_only_icquant(model, group_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_icquant_declines_when_num_outliers_too_large():
    model = _matmul_model(K=32, N=8, seed=9)
    q = onnxsim.quantize_weight_only_icquant(
        model, group_size=8, num_outliers=8
    )  # num_outliers must be < group_size
    assert q.SerializeToString() == model.SerializeToString()


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


def test_icquant_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.quantize_weight_only_icquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_icquant_zero_outliers_skips_scatternd():
    model = _matmul_model(K=32, N=8, seed=10)
    q = onnxsim.quantize_weight_only_icquant(model, group_size=8, num_outliers=0)
    onnx.checker.check_model(q)
    op_types = [n.op_type for n in q.graph.node]
    assert "ScatterND" not in op_types
    assert "DequantizeLinear" in op_types


def test_icquant_rejects_negative_num_outliers():
    model = _matmul_model(K=32, N=8, seed=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_icquant(model, num_outliers=-1)
