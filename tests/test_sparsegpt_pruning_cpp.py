"""Tests for ``onnxsim.apply_sparsegpt_pruning_cpp`` -- the C++-backed port
of ``onnxsim.apply_sparsegpt_pruning`` (SparseGPT, Frantar & Alistarh, 2023;
see ``onnxsim/structured_pruning_entry.cpp``'s own "SparseGPT (unstructured /
N:M) pruning" section and ``ApplySparseGptPruning``). Like
``test_structured_wanda_pruning_cpp.py``, this runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor.

Unlike every other C++-ported pruning pass tested elsewhere in this test
suite (all purely structural: they drop whole rows/columns and leave every
surviving entry byte-for-byte unchanged), SparseGPT RECOMPUTES every
surviving entry's own value via a sequential, Hessian-error-compensating
update -- so "reaches the target sparsity" is nowhere near enough to prove
this port correct. The tests below instead compare this port's actual
output, entry for entry, against the pure-Python ``onnxsim.apply_sparsegpt_
pruning`` reference (same calibration data, same parameters) -- both
implementations solve for the mathematically unique Cholesky factor of the
same damped Hessian, so a correct port should match to (and, empirically,
essentially at) floating-point precision, not merely a loose tolerance.

Scope: this port matches plain ``MatMul``/vanilla-``Gemm`` (not
``com.microsoft::FusedGemm``/``GemmFastGelu``) and ``com.microsoft::
Attention``'s merged QKV weight, FLOAT32/FLOAT16/BFLOAT16 (widened from an
earlier FLOAT32-only scope -- see ``IsSupportedFloatDtype``/
``ReadTensorAsF64``/``WriteF64TensorAs`` and, for the Attention QKV weight
specifically, the SparseGPT-local ``MatchAttentionProducerAnyFloat``), and
does NOT match ``Conv`` at all -- see ``structured_pruning_entry.h``'s own
``ApplySparseGptPruning`` declaration comment for the full scope decision
and rationale (Conv remains the one open gap, so
``onnxsim.apply_sparsegpt_pruning`` itself is NOT aliased to this port). The
Conv-declined test below confirms that gap is handled by leaving the layer
alone, not by crashing or by silently mishandling the tensor; the
FLOAT16/BFLOAT16 section further down confirms the newly-closed dtype gap
matches the pure-Python reference exactly.
"""

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=32, N=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return (
        _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{N}] Y)
            {{
              Y = MatMul(X, W)
            }}
            """,
            initializer=[_f32(weight, "W")],
        ),
        weight,
    )


def _weight(model, index=0):
    return onnx.numpy_helper.to_array(model.graph.initializer[index])


def _assert_bytewise_close(actual, expected, rtol=1e-5, atol=1e-6):
    # SparseGPT recomputes every kept entry too, so a correct port should
    # match the Python reference at essentially floating-point precision --
    # this module's own docstring explains why a loose tolerance would not
    # actually prove the port correct.
    np.testing.assert_allclose(
        actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
    )


# --- Core: matches the pure-Python reference exactly ----------------------


def test_sparsegpt_pruning_cpp_matmul_unstructured_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=50)
    rng = np.random.default_rng(150)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=12
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    # Real recomputation happened, not a same-shape no-op.
    assert not np.array_equal(_weight(actual), _weight(model))


def test_sparsegpt_pruning_cpp_matmul_reaches_roughly_the_target_sparsity():
    K, N = 64, 16
    model, _w = _matmul_model(K=K, N=N, seed=58)
    rng = np.random.default_rng(158)
    x_cal = rng.standard_normal((96, K)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.1)
    # Value-only rewrite -- shape is never touched.
    assert _weight(pruned).shape == _weight(model).shape


def test_sparsegpt_pruning_cpp_gemm_transb_matches_python_reference():
    # transB=1 Gemm -- exercises the weight_transposed=True branch of both
    # the Python reference and this C++ port's own w <-> w_nk reshape.
    K, N = 24, 6
    rng = np.random.default_rng(51)
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.4  # [N, K], transB layout
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(bias, "B")],
    )
    x_cal = rng.standard_normal((40, K)).astype(np.float32)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4, proc_block_size=10
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4, proc_block_size=10
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    assert not np.array_equal(_weight(actual), w)


def test_sparsegpt_pruning_cpp_nm_pattern_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=52)
    rng = np.random.default_rng(152)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], n=2, m=4, proc_block_size=12
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))

    # Exactly 2 of every 4 consecutive columns survive, per output row --
    # the actual N:M structural guarantee, not merely "matches Python".
    w_nk = _weight(actual).T  # [N, K]
    for row in w_nk:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            if len(group) == 4:
                assert np.count_nonzero(group) == 2


def test_sparsegpt_pruning_cpp_attention_merged_qkv_matches_python_reference():
    hidden = 16
    nq = nk = nv = 8
    total_n = nq + nk + nv
    num_heads = 2
    rng = np.random.default_rng(53)
    w_qkv = rng.standard_normal((hidden, total_n)).astype(np.float32) * 0.3
    bias = rng.standard_normal((total_n,)).astype(np.float32) * 0.05
    model = _model(
        f"""
        g (float[batch,seq,{hidden}] X) => (float[batch,seq,{nv}] Y)
        {{
          Y, present = com.microsoft.Attention <num_heads={num_heads}>(X, Wqkv, Bqkv)
        }}
        """,
        initializer=[_f32(w_qkv, "Wqkv"), _f32(bias, "Bqkv")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    rng2 = np.random.default_rng(153)
    x_cal = rng2.standard_normal((3, 5, hidden)).astype(np.float32)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    # The merged weight was actually pruned, and the bias (never touched by
    # SparseGPT -- see this port's own scope) is untouched.
    assert not np.array_equal(_weight(actual), w_qkv)
    np.testing.assert_array_equal(_weight(actual, index=1), bias)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_sparsegpt_pruning_cpp_multiple_layers_sharing_one_input_matches_python_reference():
    # Mirrors the shape com.microsoft::GroupQueryAttention's own separate
    # Q/K/V projections take: three independent MatMul weights, all reading
    # the SAME upstream activation -- pruning.py's own docstring explains
    # why these need no special-casing at all (ordinary MatMul/Gemm nodes,
    # ranked and pruned exactly like any other layer). Each gets its own H
    # (built from the one shared probe) and is pruned completely
    # independently -- exactly what this test checks, entry for entry
    # against the Python reference for all three weights at once.
    hidden = 20
    rng = np.random.default_rng(54)
    wq = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    wk = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    wv = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[batch,{hidden}] X) => (float[batch,{hidden}] Q, float[batch,{hidden}] K, float[batch,{hidden}] V)
        {{
          Q = MatMul(X, Wq)
          K = MatMul(X, Wk)
          V = MatMul(X, Wv)
        }}
        """,
        initializer=[_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv")],
    )
    rng2 = np.random.default_rng(154)
    x_cal = rng2.standard_normal((40, hidden)).astype(np.float32)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    onnx.checker.check_model(actual)
    for i in range(3):
        _assert_bytewise_close(_weight(actual, i), _weight(expected, i))
        assert not np.array_equal(
            _weight(actual, i), _weight(model, i)
        )  # each was actually pruned


# --- No-op / declined-input behavior ---------------------------------------


def test_sparsegpt_pruning_cpp_zero_sparsity_is_a_noop():
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=55)
    rng = np.random.default_rng(155)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_weight(pruned), w)


def test_sparsegpt_pruning_cpp_no_calibration_batches_leaves_layer_untouched():
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=56)
    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    np.testing.assert_array_equal(_weight(pruned), w)


def test_sparsegpt_pruning_cpp_only_layers_with_observed_activations_are_pruned():
    # Two independent MatMul layers, each fed by its OWN graph input --
    # calibration_data supplies real data for "X1" but only an all-zero
    # batch for "X2" (still required: ModelExecutor::Run needs every
    # positional graph input filled, so it can't simply be omitted). A
    # dead (all-zero) probe activation makes its Hessian's own diagonal
    # exactly zero everywhere, so every column of that layer is "dead"
    # (see InverseHessianCholesky's own dead-channel handling) -- but the
    # layer is still very much observed and processed (dead columns get a
    # fixed diagonal of 1.0 specifically so they CAN still be pruned/
    # compensated, not skipped), so both layers end up pruned. This checks
    # the two are pruned completely independently of one another: W1 (real
    # signal) reconstructs the layer well; W2 (dead input) is provably
    # equivalent to magnitude-only pruning on all-zero-Hessian columns.
    K1, N1 = 16, 4
    K2, N2 = 12, 3
    rng = np.random.default_rng(59)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((K2, N2)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K1}] X1, float[batch,{K2}] X2) => (float[batch,{N1}] Y1, float[batch,{N2}] Y2)
        {{
          Y1 = MatMul(X1, W1)
          Y2 = MatMul(X2, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    rng2 = np.random.default_rng(159)
    x1_cal = rng2.standard_normal((40, K1)).astype(np.float32)
    x2_cal = np.zeros((40, K2), dtype=np.float32)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X1": x1_cal, "X2": x2_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X1": x1_cal, "X2": x2_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual, 0), _weight(expected, 0))
    _assert_bytewise_close(_weight(actual, 1), _weight(expected, 1))
    assert not np.array_equal(_weight(actual, 0), w1)


def test_sparsegpt_pruning_cpp_requires_n_and_m_together():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], m=4)


def test_sparsegpt_pruning_cpp_sparsity_out_of_range_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], sparsity=1.5)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], sparsity=-0.1)


def test_sparsegpt_pruning_cpp_bad_nm_relationship_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        # n > m is never valid (n must be <= m).
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], n=5, m=4)


# --- Deliberately out-of-scope inputs: declined, not mishandled ------------


def test_sparsegpt_pruning_cpp_conv_is_left_completely_untouched():
    # This port does NOT match Conv at all (see structured_pruning_entry.h's
    # own ApplySparseGptPruning declaration comment) -- confirm the weight
    # is left byte-identical, not crashed on and not silently (incorrectly)
    # pruned as if it were a plain 2-D MatMul weight.
    Cin, Cout = 3, 4
    rng = np.random.default_rng(60)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          Y = Conv<kernel_shape=[3,3]>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(160)
    x_cal = rng2.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    np.testing.assert_array_equal(_weight(pruned), w)


# --- FLOAT16/BFLOAT16 weight matching ---------------------------------------
#
# FLOAT16/BFLOAT16 weights (both for the plain MatMul/vanilla-Gemm candidate
# path and for com.microsoft::Attention's own merged QKV weight, via the
# SparseGPT-local MatchAttentionProducerAnyFloat matcher) are IN scope for
# this C++ port -- IsSupportedFloatDtype/ReadTensorAsF64/WriteF64TensorAs,
# the same widening this file's own module docstring's "Scope" paragraph
# above already documents. 2-D Conv remains the one open gap (see the
# previous test).
#
# BFLOAT16 gets no analogous real-onnxruntime-execution test here: this
# environment's onnxruntime has no CPU kernel for ANY op on a BFLOAT16
# tensor at all (confirmed the same way test_pruning.py's own "BFLOAT16 has
# no onnxruntime CPU execution support" section comment documents for the
# pure-Python reference -- a plain BFLOAT16 MatMul model raises
# NOT_IMPLEMENTED the moment a session is created), and this test file's own
# module docstring requires a real onnxruntime-backed executor throughout
# (never a fake/mock one) -- so a BFLOAT16 calibration run would fail on
# this environment's onnxruntime limitation alone, before this port's own
# widened matching/reading/writing code ever runs. There is accordingly no
# environment in which this file could exercise BFLOAT16 end to end; FLOAT16
# (which onnxruntime CAN execute) is tested below instead.


def test_sparsegpt_pruning_cpp_fp16_matmul_matches_python_reference():
    K, N = 16, 4
    rng = np.random.default_rng(61)
    w = (rng.standard_normal((K, N)).astype(np.float32) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[onnx.numpy_helper.from_array(w, "W")],
    )
    rng2 = np.random.default_rng(161)
    x_cal = rng2.standard_normal((32, K)).astype(np.float16)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    w_expected = _weight(expected)
    w_actual = _weight(actual)
    assert w_actual.dtype == np.float16
    assert w_expected.dtype == np.float16
    _assert_bytewise_close(w_actual, w_expected)
    # Real recomputation happened, not a same-shape no-op.
    assert not np.array_equal(w_actual, w)


def test_sparsegpt_pruning_cpp_fp16_attention_merged_qkv_matches_python_reference():
    # Exercises MatchAttentionProducerAnyFloat specifically -- the
    # SparseGPT-local, dtype-widened duplicate of the shared (still
    # FLOAT32-only) MatchAttentionProducer used elsewhere in this file.
    hidden = 16
    nq = nk = nv = 8
    total_n = nq + nk + nv
    num_heads = 2
    rng = np.random.default_rng(62)
    w_qkv = (rng.standard_normal((hidden, total_n)).astype(np.float32) * 0.3).astype(
        np.float16
    )
    bias = (rng.standard_normal((total_n,)).astype(np.float32) * 0.05).astype(
        np.float16
    )
    model = _model(
        f"""
        g (float16[batch,seq,{hidden}] X) => (float16[batch,seq,{nv}] Y)
        {{
          Y, present = com.microsoft.Attention <num_heads={num_heads}>(X, Wqkv, Bqkv)
        }}
        """,
        initializer=[
            onnx.numpy_helper.from_array(w_qkv, "Wqkv"),
            onnx.numpy_helper.from_array(bias, "Bqkv"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    rng2 = np.random.default_rng(162)
    x_cal = rng2.standard_normal((3, 5, hidden)).astype(np.float16)

    expected = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    w_expected = _weight(expected)
    w_actual = _weight(actual)
    assert w_actual.dtype == np.float16
    _assert_bytewise_close(w_actual, w_expected)
    assert not np.array_equal(w_actual, w_qkv)
    # Bias (never touched by SparseGPT) is untouched, dtype included.
    np.testing.assert_array_equal(_weight(actual, index=1), bias)


# --- End-to-end reconstruction quality --------------------------------------


def test_sparsegpt_pruning_cpp_reconstructs_better_than_a_same_mask_style_baseline():
    # The actual point of the technique, checked end to end through the C++
    # port specifically (mirrors test_pruning.py's own identically-named
    # pure-Python test): given comparable calibration signal, SparseGPT's
    # Hessian-compensated result should reconstruct the layer's output at
    # least as well, on that same calibration data, as simply zeroing the
    # same-shaped lowest-magnitude entries with no compensation at all.
    K, N = 48, 12
    rng = np.random.default_rng(62)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model, _w = _matmul_model(K=K, N=N, seed=62)
    assert np.array_equal(_weight(model), w)
    x_cal = rng.standard_normal((512, K)).astype(np.float32)  # well-conditioned H

    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    w_sparsegpt = _weight(pruned).astype(np.float64)

    w64 = w.astype(np.float64)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64)

    x64 = x_cal.astype(np.float64)
    y_orig = x64 @ w64
    err_sparsegpt = np.sum((y_orig - x64 @ w_sparsegpt) ** 2)
    err_naive = np.sum((y_orig - x64 @ w_naive) ** 2)
    assert err_sparsegpt <= err_naive
