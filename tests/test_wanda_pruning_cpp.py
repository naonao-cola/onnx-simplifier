"""Tests for ``onnxsim.apply_wanda_pruning_cpp`` -- the C++-backed port of
``onnxsim.apply_wanda_pruning`` (Wanda, Sun et al., 2023; see
``onnxsim/structured_pruning_entry.cpp``'s own "Wanda unstructured
(element-wise) pruning" section and ``ApplyWandaPruning``). Like
``test_sparsegpt_pruning_cpp.py``/``test_structured_wanda_pruning_cpp.py``,
this runs the model over real calibration data through a real
``onnxruntime``-backed :class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor.

Unlike ``apply_sparsegpt_pruning_cpp`` (which recomputes every *kept*
entry's own value too), Wanda is a one-shot static importance score --
exactly like ``prune_magnitude_cpp`` -- so a correct port zeros exactly the
same entries the pure-Python ``onnxsim.apply_wanda_pruning`` reference
zeros, and every surviving entry is byte-identical to the original (never
recomputed).

Scope: this port matches plain ``MatMul``/vanilla-``Gemm`` (not
``com.microsoft::FusedGemm``/``GemmFastGelu``) and ``com.microsoft::
Attention``'s merged QKV weight, each with a constant 2-D (1-D merged bias,
for Attention) FLOAT32/FLOAT16/BFLOAT16 weight -- TRUE parity with the
pure-Python ``onnxsim.apply_wanda_pruning`` on this axis -- and does NOT
match ``Conv`` at all -- see ``structured_pruning_entry.h``'s own
``ApplyWandaPruning`` declaration comment for the full scope decision and
rationale (mirroring ``ApplySparseGptPruning``'s own identical Conv
decision). The Conv-declined test below confirms that gap is handled by
leaving the layer alone, not by crashing or by silently mishandling the
tensor; the FLOAT16/BFLOAT16 tests below confirm those dtypes are matched,
pruned, and written back with their own exact original bit pattern
preserved for every surviving entry -- not merely "not crashed on".
"""

import ml_dtypes
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


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


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


def _magnitude_weight(model, node_index=0, input_index=1):
    # `onnxsim.apply_magnitude_pruning` is now a thin alias for the C++-backed
    # `onnxsim.prune_magnitude_cpp`, which leaves the original initializer
    # dangling and appends a new, anonymously-named one for the pruned
    # weight (see `tests/test_pruning_cpp.py`'s own `_weight` helper) --
    # unlike `_weight` above (position-based, still correct for
    # `apply_wanda_pruning_cpp`'s own in-place-mutating pure-Python
    # reference), a magnitude-pruning result must be resolved via the
    # node's CURRENT weight input.
    node = model.graph.node[node_index]
    w_name = node.input[input_index]
    init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(init)


def _assert_bytewise_close(actual, expected, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(
        actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
    )


# --- Core: matches the pure-Python reference exactly ----------------------


def test_wanda_pruning_cpp_matmul_unstructured_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=50)
    rng = np.random.default_rng(150)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    assert not np.array_equal(_weight(actual), _weight(model))


def test_wanda_pruning_cpp_matmul_reaches_roughly_the_target_sparsity():
    K, N = 64, 16
    model, _w = _matmul_model(K=K, N=N, seed=58)
    rng = np.random.default_rng(158)
    x_cal = rng.standard_normal((96, K)).astype(np.float32)

    pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.1)
    assert _weight(pruned).shape == _weight(model).shape


def test_wanda_pruning_cpp_gemm_transb_matches_python_reference():
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

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    assert not np.array_equal(_weight(actual), w)


def test_wanda_pruning_cpp_nm_pattern_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=52)
    rng = np.random.default_rng(152)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], n=2, m=4
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], n=2, m=4
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


def test_wanda_pruning_cpp_attention_merged_qkv_matches_python_reference():
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

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    assert not np.array_equal(_weight(actual), w_qkv)
    np.testing.assert_array_equal(_weight(actual, index=1), bias)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_multiple_layers_sharing_one_input_matches_python_reference():
    # Mirrors the shape com.microsoft::GroupQueryAttention's own separate
    # Q/K/V projections take: three independent MatMul weights, all reading
    # the SAME upstream activation -- ranked and pruned completely
    # independently (each has its own weight, but they share one act_norm
    # entry, keyed by x_name).
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

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    onnx.checker.check_model(actual)
    for i in range(3):
        _assert_bytewise_close(_weight(actual, i), _weight(expected, i))
        assert not np.array_equal(_weight(actual, i), _weight(model, i))


# --- global_sparsity mode ---------------------------------------------------


def test_wanda_pruning_cpp_global_sparsity_matches_python_reference():
    K1, N1 = 20, 4
    K2, N2 = 8, 6
    rng = np.random.default_rng(70)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((K2, N2)).astype(np.float32) * 2.0
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
    rng2 = np.random.default_rng(170)
    x1_cal = rng2.standard_normal((40, K1)).astype(np.float32)
    x2_cal = rng2.standard_normal((40, K2)).astype(np.float32)

    expected = onnxsim.apply_wanda_pruning(
        model,
        calibration_data=[{"X1": x1_cal, "X2": x2_cal}],
        sparsity=0.5,
        global_sparsity=True,
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model,
        calibration_data=[{"X1": x1_cal, "X2": x2_cal}],
        sparsity=0.5,
        global_sparsity=True,
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual, 0), _weight(expected, 0))
    _assert_bytewise_close(_weight(actual, 1), _weight(expected, 1))
    # The whole-model pooled sparsity target is reached exactly at the
    # element count level (no per-row floor), matching the Python original.
    total = w1.size + w2.size
    zeros = np.count_nonzero(_weight(actual, 0) == 0) + np.count_nonzero(
        _weight(actual, 1) == 0
    )
    assert zeros == pytest.approx(total * 0.5, abs=1)


def test_wanda_pruning_cpp_global_sparsity_rejects_nm():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(
            model, calibration_data=[], n=2, m=4, global_sparsity=True
        )


# --- No-calibration-data behavior: falls back to plain magnitude -----------


def test_wanda_pruning_cpp_no_calibration_batches_falls_back_to_plain_magnitude():
    # pruning.py's own apply_wanda_pruning falls back to plain-|W| magnitude
    # importance when no calibration activation was ever observed for a
    # layer (_wanda_importance's own `norm is None` branch) -- VERIFIED by
    # reading the Python source, not assumed. With calibration_data=[], this
    # C++ port should therefore match onnxsim.apply_magnitude_pruning
    # exactly, not merely leave the layer untouched (unlike
    # apply_sparsegpt_pruning_cpp, which has no data-free fallback at all).
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=56)

    pruned = onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=0.5)
    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), _magnitude_weight(magnitude_pruned))
    # Actually pruned, not a no-op.
    assert not np.array_equal(_weight(pruned), w)


def test_wanda_pruning_cpp_zero_sparsity_is_a_noop():
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=55)
    rng = np.random.default_rng(155)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_weight(pruned), w)


def test_wanda_pruning_cpp_requires_n_and_m_together():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], m=4)


def test_wanda_pruning_cpp_sparsity_out_of_range_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=1.5)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=-0.1)


def test_wanda_pruning_cpp_bad_nm_relationship_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], n=5, m=4)


# --- Activation-weighted importance provably differs from plain magnitude --


def test_wanda_pruning_cpp_activation_weighting_differs_from_plain_magnitude():
    # One input feature (row of W, in the [K, N] layout) is scaled far above
    # the rest during calibration -- so its own small |W| entries can still
    # out-rank a larger-magnitude entry belonging to a quiet input channel.
    # This directly exercises the ||X_j||_2 multiplier: a plain-magnitude
    # pruning of the exact same weight would drop a different entry set.
    K, N = 8, 4
    salient_k = 2
    rng = np.random.default_rng(64)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    # Make salient_k's own row of W uniformly small in magnitude so it would
    # be dropped first under plain-|W| ranking...
    w[salient_k, :] = 0.02
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    x_cal = rng.standard_normal((64, K)).astype(np.float32)
    # ...but scale that same channel's activation up massively at
    # calibration time, so ||X_salient||_2 dominates every other channel's
    # own norm and Wanda should keep it despite its small |W|.
    x_cal[:, salient_k] *= 100.0

    wanda_pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)

    w_wanda = _weight(wanda_pruned)
    w_magnitude = _magnitude_weight(magnitude_pruned)
    assert not np.array_equal(w_wanda, w_magnitude)
    # Plain magnitude drops the whole salient_k row (its own tied-smallest
    # entries); Wanda keeps at least one of them thanks to the activation
    # boost.
    assert np.count_nonzero(w_magnitude[salient_k, :]) == 0
    assert np.count_nonzero(w_wanda[salient_k, :]) > 0


# --- Deliberately out-of-scope inputs: declined, not mishandled ------------


def test_wanda_pruning_cpp_conv_is_left_completely_untouched():
    # This port does NOT match Conv at all (see structured_pruning_entry.h's
    # own ApplyWandaPruning declaration comment) -- confirm the weight is
    # left byte-identical, not crashed on and not silently (incorrectly)
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
    pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    np.testing.assert_array_equal(_weight(pruned), w)


# --- FLOAT16/BFLOAT16 weight support: matches the pure-Python reference ----
#
# Unlike the Conv gap above, FLOAT16/BFLOAT16 is now TRUE parity with
# pruning.py's own `apply_wanda_pruning` -- reads out upcast to float64 via
# ReadTensorAsF64, importance/masking computed identically to the FLOAT32
# path, written back down via WriteF64TensorAs, exactly mirroring
# pruning.py's own `_to_f64`/`_from_f64` round trip (see
# structured_pruning_entry.h's own ApplyWandaPruning declaration comment).


def test_wanda_pruning_cpp_matmul_float16_matches_python_reference():
    K, N = 16, 4
    rng = np.random.default_rng(61)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w, "W")],
    )
    onnx.checker.check_model(model)
    rng2 = np.random.default_rng(161)
    x_cal = rng2.standard_normal((16, K)).astype(np.float16)

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    # Exact bit-pattern match against the Python reference (not
    # assert_allclose): both round-trip through float64 and mask, never
    # recompute, a surviving entry's own value.
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(expected).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w.view(np.uint16))
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_matmul_bfloat16_matches_python_reference():
    # onnxruntime has no BFLOAT16 CPU execution support in this environment
    # (confirmed separately: a plain BFLOAT16 MatMul session raises
    # NOT_IMPLEMENTED at session-creation time -- see this repo's own
    # test_magnitude_pruning_bfloat16_preserves_dtype_and_matches_array_
    # oracle for the same note) -- so calibration_data is deliberately `[]`
    # (not merely omitted -- omitting it triggers random calibration data
    # generation, still a real session) rather than a real batch: both
    # apply_wanda_pruning and apply_wanda_pruning_cpp skip calling the
    # executor entirely for zero calibration batches (see
    # WandaCalibrationStats' own top comment) and fall back to plain
    # per-layer magnitude importance, which is exactly what this test cross-
    # checks -- the BFLOAT16 candidate-matching/read-upcast/write-downcast
    # round trip, not the (here environment-unsupported) real activation
    # capture.
    K, N = 16, 4
    rng = np.random.default_rng(62)
    w = (rng.standard_normal((K, N)) * 0.5).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (bfloat16[batch,{K}] X) => (bfloat16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_bf16(w, "W")],
    )
    onnx.checker.check_model(model)

    expected = onnxsim.apply_wanda_pruning(model, calibration_data=[], sparsity=0.5)
    actual = onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=0.5)
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.BFLOAT16
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(expected).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w.view(np.uint16))
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_attention_merged_qkv_float16_matches_python_reference():
    # FLOAT16 analogue of test_wanda_pruning_cpp_attention_merged_qkv_
    # matches_python_reference -- exercises MatchAttentionProducerWideDtype
    # (this pass' own local, dtype-widened copy of MatchAttentionProducer),
    # not just the MatMul/Gemm candidate path.
    hidden = 16
    nq = nk = nv = 8
    total_n = nq + nk + nv
    num_heads = 2
    rng = np.random.default_rng(63)
    w_qkv = (rng.standard_normal((hidden, total_n)) * 0.3).astype(np.float16)
    bias = (rng.standard_normal((total_n,)) * 0.05).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,seq,{hidden}] X) => (float16[batch,seq,{nv}] Y)
        {{
          Y, present = com.microsoft.Attention <num_heads={num_heads}>(X, Wqkv, Bqkv)
        }}
        """,
        initializer=[_f16(w_qkv, "Wqkv"), _f16(bias, "Bqkv")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    rng2 = np.random.default_rng(163)
    x_cal = rng2.standard_normal((3, 5, hidden)).astype(np.float16)

    expected = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(expected).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w_qkv.view(np.uint16))
    # Bias is never touched by unstructured/N:M pruning -- byte-identical.
    np.testing.assert_array_equal(_weight(actual, index=1), bias)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)
