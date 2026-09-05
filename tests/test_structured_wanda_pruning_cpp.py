"""Tests for ``onnxsim.apply_structured_wanda_pruning_cpp`` -- the C++-backed
port of ``onnxsim.apply_structured_wanda_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s "Wanda calibration" section and
``ApplyStructuredWandaPruning``). This is onnxsim's first calibration-driven
(not purely graph-structural) C++ pruning entry point: it runs the model over
real calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``, the same executor
:func:`onnxsim.simplify` itself uses) to capture per-channel activation
norms, exactly proving the DLPack executor boundary works end to end for a
non-constant-folding caller -- never a fake/mock executor.

Same chain-finding scope as ``onnxsim.apply_structured_pruning_cpp`` (see
that port's own module docstring), minus the additional quantized-weight
chain families it also matches -- this Wanda port has no quantized-weight
counterpart, mirroring the pure-Python ``onnxsim.apply_structured_wanda_pruning``
exactly (see ``structured_pruning_entry.h``'s own ``ApplyStructuredWandaPruning``
declaration comment).

``onnxsim.apply_structured_wanda_pruning`` is now a thin alias for this port
(see that function's own docstring) -- TRUE full parity with the pure-Python
implementation was verified (the FULL ``tests/test_pruning.py`` suite, not
merely this file's own hand-picked cases) before that alias was made, closing
two real, previously-confirmed scope gaps:
(1) this port's own ``FindConvChains``/``MatchConvProducer`` used to match
only a plain ``Conv`` node, never ``ConvTranspose``, while the pure-Python
``_match_conv_producer``/``_match_conv_transpose_producer`` matched both --
closed by ``MatchConvTransposeProducer``/``MatchConvTransposeConsumer`` and
``WalkToConvConsumer``'s own ``allow_conv_transpose_consumer`` parameter (see
this file's own "ConvTranspose producer/consumer roles" section below); and
(2) a ``Concat``-merged branch feeding a *grouped* Conv consumer used to be
declined outright rather than admitted when block-aligned, combined with a
real calibrated (Wanda) activation norm producing a different keep set than
the pure-Python reference (``test_structured_wanda_pruning_conv_concat_
admits_block_aligned_grouped_conv_consumer`` in ``test_pruning.py`` used to
fail against this port) -- closed by ``ConcatBranchesAlignToConsumerGroup``
(see this file's own ``test_structured_wanda_pruning_cpp_conv_concat_admits_
block_aligned_grouped_conv_consumer`` below, and the identical fix's own
plain-structured-pruning counterpart,
``test_cpp_structured_pruning_conv_concat_admits_block_aligned_grouped_conv_
consumer`` in ``test_structured_pruning_cpp.py`` -- the underlying matcher
(``FindConvConcatChains``) is shared by both entry points).

Every ``..._matches_python_reference_...`` test below that cross-checks this
port's own output against ``onnxsim.apply_structured_wanda_pruning`` is,
consequently, now comparing the C++ port against a thin wrapper around
itself for that specific assertion alone -- each such test also carries an
independent, from-scratch numpy oracle (not read back from either
implementation under test) precisely so it stays a real regression check
rather than a tautology once the alias is in place; see each test's own
comment for its own oracle. The FULL ``tests/test_pruning.py`` suite's own
concrete-oracle tests (never a cross-implementation comparison) are this
port's primary regression backstop going forward, since they now exercise
this port directly through the aliased pure-Python entry point.
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _plain_keep_indices(w1, keep_count):
    # Plain (weight-magnitude-only) importance oracle -- the same criterion
    # onnxsim.apply_structured_pruning_cpp itself ranks by.
    importance = np.linalg.norm(w1.T.astype(np.float64), axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _mlp_model(K, H, Out, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    return (
        _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = MatMul(X, W1)
              a = Relu(h)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
        ),
        w1,
        w2,
    )


def _conv_pair_model(w1, w2, spatial=10):
    Cin, C2 = w1.shape[1], w2.shape[0]
    out_spatial = spatial - 4
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )


# --- Basic chain: activation-weighted result differs from plain magnitude
# pruning on the same weights, and matches a hand-computed numpy oracle -----


def test_structured_wanda_pruning_cpp_matmul_matches_oracle_and_differs_from_plain():
    # One input feature is scaled far above the rest during calibration (and
    # at eval time) -- the same "engineer a salient channel" technique
    # test_pruning.py's own structured Wanda Conv regression tests use --
    # so the resulting keep set is actually activation-driven, verified
    # below to differ from plain L2-norm-only ranking rather than merely
    # reproducing it by coincidence.
    K, H, Out = 8, 16, 4
    salient_input = 2
    rng = np.random.default_rng(50)
    w1 = (rng.standard_normal((K, H)).astype(np.float32)) * 0.3
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    rng_cal = np.random.default_rng(51)
    x_cal = rng_cal.standard_normal((32, K)).astype(np.float32)
    x_cal[:, salient_input] *= 30.0
    calibration_data = [{"X": x_cal}]

    h_cal = np.maximum(x_cal.astype(np.float64) @ w1.astype(np.float64), 0.0)
    act_norm = np.sqrt(np.mean(np.square(h_cal), axis=0))
    importance = np.linalg.norm(w1.T.astype(np.float64), axis=1) * np.maximum(
        act_norm, 1e-8
    )
    keep = np.sort(np.argsort(-importance)[: H // 2])
    plain_keep = _plain_keep_indices(w1, H // 2)
    assert not np.array_equal(keep, plain_keep)  # the activation term matters here

    pruned = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    rng_x = np.random.default_rng(52)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    x[:, salient_input] *= 30.0
    (y,) = _run(pruned, {"X": x})
    h = np.maximum(x @ w1[:, keep], 0.0)
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)

    # Also confirm the pruned model actually kept the oracle's own channel
    # set, not merely one that happens to produce a numerically close
    # output -- i.e. that plain magnitude pruning on the identical weights
    # would have kept a different, worse set.
    plain_pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    (y_plain,) = _run(plain_pruned, {"X": x})
    assert not np.allclose(y_plain, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_wanda_pruning_cpp_conv_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(60)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2)

    rng_cal = np.random.default_rng(61)
    x_cal = rng_cal.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    # Compute the real probe activation (right where the chain feeds its
    # consumer, i.e. post-Relu) via a probe model exactly like
    # bias_correction.py's own _add_probe_outputs.
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )
    _, a_cal_real = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(
        np.mean(np.square(a_cal_real.astype(np.float64)), axis=(0, 2, 3))
    )
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])

    pruned = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _conv_pair_model(w1[keep], w2[:, keep])
    rng_x = np.random.default_rng(62)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- No-calibration-data fallback -------------------------------------------


def test_structured_wanda_pruning_cpp_empty_calibration_data_matches_plain():
    # An empty (but present) calibration_data means no activation was ever
    # observed for any probe point, so every matched chain falls back to
    # apply_structured_pruning_cpp's own plain ||W_row||_2 ranking --
    # exactly byte-identical output.
    model, _, _ = _mlp_model(K=8, H=16, Out=4, seed=70)

    wanda_empty = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    plain = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert wanda_empty.SerializeToString() == plain.SerializeToString()


def test_structured_wanda_pruning_cpp_conv_empty_calibration_data_matches_plain():
    Cin, C1, C2 = 3, 12, 6
    rng = np.random.default_rng(71)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2)

    wanda_empty = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    plain = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert wanda_empty.SerializeToString() == plain.SerializeToString()


# --- Cross-check against the pure-Python reference --------------------------


def test_structured_wanda_pruning_cpp_matches_python_reference_matmul():
    K, H, Out = 8, 20, 5
    model, w1, w2 = _mlp_model(K=K, H=H, Out=Out, seed=80)
    rng_cal = np.random.default_rng(81)
    x_cal = rng_cal.standard_normal((24, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)

    cpp_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    py_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_py.graph.initializer
    }
    assert set(cpp_inits) == set(py_inits)
    for name in cpp_inits:
        np.testing.assert_array_equal(cpp_inits[name], py_inits[name])

    # Independent oracle -- kept as a real (not cross-implementation-only)
    # regression check: computed from scratch in numpy, not read back from
    # either implementation under test, so this stays meaningful even once
    # apply_structured_wanda_pruning becomes a thin alias for this C++ port.
    h_cal = np.maximum(x_cal.astype(np.float64) @ w1.astype(np.float64), 0.0)
    act_norm = np.sqrt(np.mean(np.square(h_cal), axis=0))
    importance = np.linalg.norm(w1.T.astype(np.float64), axis=1) * np.maximum(
        act_norm, 1e-8
    )
    keep_count = H - round(H * 0.5)
    keep = np.sort(np.argsort(-importance)[:keep_count])
    np.testing.assert_array_equal(cpp_inits["W1"], w1[:, keep])
    np.testing.assert_array_equal(cpp_inits["W2"], w2[keep, :])


def test_structured_wanda_pruning_cpp_matches_python_reference_multi_batch():
    # Multiple calibration batches, accumulated sum-of-squares across all of
    # them -- exercises WandaCalibrationStats' own per-batch accumulation
    # loop against pruning.py's own identical `for batch in calibration_data`
    # loop.
    K, H, Out = 6, 14, 3
    model, w1, w2 = _mlp_model(K=K, H=H, Out=Out, seed=90)
    rng_cal = np.random.default_rng(91)
    batches = [rng_cal.standard_normal((8, K)).astype(np.float32) for _ in range(4)]
    calibration_data = [{"X": b} for b in batches]

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # Independent oracle -- sum-of-squares accumulated across every batch
    # (see the matmul reference test's own comment for why this stays
    # meaningful post-alias).
    sq_sum = np.zeros(H, dtype=np.float64)
    n_samples = 0
    for b in batches:
        h_cal = np.maximum(b.astype(np.float64) @ w1.astype(np.float64), 0.0)
        sq_sum += np.sum(np.square(h_cal), axis=0)
        n_samples += h_cal.shape[0]
    act_norm = np.sqrt(sq_sum / n_samples)
    importance = np.linalg.norm(w1.T.astype(np.float64), axis=1) * np.maximum(
        act_norm, 1e-8
    )
    keep_count = H - round(H * 0.6)
    keep = np.sort(np.argsort(-importance)[:keep_count])
    cpp_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(cpp_inits["W1"], w1[:, keep])
    np.testing.assert_array_equal(cpp_inits["W2"], w2[keep, :])


def test_structured_wanda_pruning_cpp_matches_python_reference_conv():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(100)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(101)
    x_cal = rng_cal.standard_normal((3, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, a_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # Independent oracle -- see the matmul reference test's own comment for
    # why this stays meaningful post-alias.
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])
    cpp_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(cpp_inits["W1"], w1[keep])
    np.testing.assert_array_equal(cpp_inits["W2"], w2[:, keep])


def test_structured_wanda_pruning_cpp_matches_python_reference_concat_branch():
    # A Concat-merged branch: each branch's own probe point is where it
    # feeds the Concat node itself, not the shared downstream consumer --
    # exercises WandaCalibrationStats' own ConcatChain/branch.operand_name
    # probe wiring (and ApplyConcatChains' own act_norm threading) against
    # pruning.py's own `_wanda_branch_importance`.
    Cin, C1a, C1b, Out = 5, 6, 10, 4
    rng = np.random.default_rng(110)
    wa = rng.standard_normal((Cin, C1a)).astype(np.float32)
    wb = rng.standard_normal((Cin, C1b)).astype(np.float32)
    w2 = rng.standard_normal((C1a + C1b, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{Cin}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, Wa)
          b = MatMul(X, Wb)
          m = Concat<axis=-1>(a, b)
          Y = MatMul(m, W2)
        }}
        """,
        initializer=[_f32(wa, "Wa"), _f32(wb, "Wb"), _f32(w2, "W2")],
    )

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(111)
    x_cal = rng_cal.standard_normal((16, Cin)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, a_cal, b_cal = _run(probe_model, {"X": x_cal})
    norm_a = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=0))
    norm_b = np.sqrt(np.mean(np.square(b_cal.astype(np.float64)), axis=0))

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # Sanity: the branches were actually pruned (not silently left
    # untouched), so this test really exercises the Concat-branch path.
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["Wa"].dims)[1] < C1a
    assert list(inits["Wb"].dims)[1] < C1b

    # Independent oracle -- each branch ranked/pruned to its own independent
    # keep set (see the matmul reference test's own comment for why this
    # stays meaningful post-alias).
    imp_a = np.linalg.norm(wa.T.astype(np.float64), axis=1) * np.maximum(norm_a, 1e-8)
    imp_b = np.linalg.norm(wb.T.astype(np.float64), axis=1) * np.maximum(norm_b, 1e-8)
    keep_a = np.sort(np.argsort(-imp_a)[: C1a - round(C1a * 0.4)])
    keep_b = np.sort(np.argsort(-imp_b)[: C1b - round(C1b * 0.4)])
    global_keep = np.concatenate([keep_a, keep_b + C1a])
    cpp_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(cpp_inits["Wa"], wa[:, keep_a])
    np.testing.assert_array_equal(cpp_inits["Wb"], wb[:, keep_b])
    np.testing.assert_array_equal(cpp_inits["W2"], w2[global_keep, :])


# --- Error handling ----------------------------------------------------------


def test_structured_wanda_pruning_cpp_missing_calibration_input_raises():
    model, _, _ = _mlp_model(K=8, H=16, Out=4, seed=120)
    bad_batch = {"NotX": np.zeros((2, 8), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_structured_wanda_pruning_cpp(
            model, calibration_data=[bad_batch], sparsity=0.5
        )


# --- Default (auto-generated) calibration data ------------------------------


def test_structured_wanda_pruning_cpp_default_calibration_data_runs():
    # calibration_data=None generates random calibration batches via
    # onnxsim.generate_random_calibration_data, matching the pure-Python
    # apply_structured_wanda_pruning's own default -- just confirms the
    # whole path runs end to end and produces a valid, actually-pruned
    # model, not a specific oracle (random data has no fixed oracle here).
    model, _, _ = _mlp_model(K=8, H=16, Out=4, seed=130)
    pruned = onnxsim.apply_structured_wanda_pruning_cpp(
        model, num_samples=4, seed=5, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 8]
    assert list(inits["W2"].dims) == [8, 4]


# --- importance_norm ("l1" vs "l2") and global_sparsity ---------------------
#
# Driven through the *empty-calibration-data* fallback path (see this file's
# own module docstring/`test_structured_wanda_pruning_cpp_*` tests above for
# the "no matching activation -> falls back to plain weight-only ranking"
# behavior): isolates the *weight*-magnitude term's own L1-vs-L2 switch (and
# global_sparsity's own pooled ranking) from the activation-norm term, while
# still exercising the real Wanda entry point/binding end to end.


def test_structured_wanda_pruning_cpp_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    # Same "concentrated" vs. "spread" adversarial column layout as
    # test_structured_pruning_cpp.py's own importance_norm test.
    K, H, Out = 16, 4, 3
    w1 = np.zeros((K, H), dtype=np.float32)
    w1[0, 0] = 8.0  # "concentrated"
    w1[:, 1] = 1.0  # "spread"
    w1[2, 2] = 1000.0  # "filler_high"
    w1[3, 3] = 0.001  # "filler_low"
    rng = np.random.default_rng(90)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    onnx.checker.check_model(model)

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_structured_wanda_pruning(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    kept_l1 = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5, importance_norm="l1"
    )
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


def _two_scale_mlp_model(K=8, H=16, Out=4, big_scale=50.0, small_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    w1_big = (rng.standard_normal((K, H)) * big_scale).astype(np.float32)
    w2_big = rng.standard_normal((H, Out)).astype(np.float32)
    w1_small = (rng.standard_normal((K, H)) * small_scale).astype(np.float32)
    w2_small = rng.standard_normal((H, Out)).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Ybig, float[batch,{Out}] Ysmall)
        {{
          hbig = MatMul(X, W1big)
          abig = Relu(hbig)
          Ybig = MatMul(abig, W2big)
          hsmall = MatMul(X, W1small)
          asmall = Relu(hsmall)
          Ysmall = MatMul(asmall, W2small)
        }}
        """,
        initializer=[
            _f32(w1_big, "W1big"),
            _f32(w2_big, "W2big"),
            _f32(w1_small, "W1small"),
            _f32(w2_small, "W2small"),
        ],
    )


def _oracle_slice_grouped_consumer_conv(w2, keep, group, n_channels):
    # A from-scratch reimplementation of _slice_grouped_consumer_conv_weight/
    # SliceGroupedConsumerConvWeight (see test_pruning.py's own identical
    # helper), kept independent of both implementations under test here so
    # this stays a real check on the algorithm rather than the algorithm
    # checking itself.
    out_channels = w2.shape[0]
    out_per_group = out_channels // group
    block = n_channels // group
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        parts.append(w2[gi * out_per_group : (gi + 1) * out_per_group][:, local_keep])
    return np.concatenate(parts, axis=0)


def test_structured_wanda_pruning_cpp_conv_concat_admits_block_aligned_grouped_conv_consumer():
    # The regression this module's own docstring used to call out (gap (2)):
    # a Concat-merged branch feeding a *grouped* Conv consumer, combined
    # with a real calibrated (Wanda) activation norm -- exercises
    # ConcatBranchesAlignToConsumerGroup's own block-alignment admission and
    # ApplyConcatChains' own per-block branch selection against both the
    # pure-Python reference (which has always supported this shape via
    # _concat_branches_align_to_consumer_group/_apply_concat_chains) and an
    # independent from-scratch oracle (so this stays meaningful even once
    # the pure-Python entry point becomes a thin alias for this C++ port).
    # Mirrors test_pruning.py's own
    # test_structured_wanda_pruning_conv_concat_admits_block_aligned_grouped_conv_consumer.
    Cin, Ca, Cb, Cout, group = 3, 4, 4, 8, 2
    rng = np.random.default_rng(230)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (Ca + Cb) // group, 1, 1)).astype(np.float32)

    def _build(wa_, wb_, wout_):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
            {{
              ha = Conv<kernel_shape=[3,3]>(X, WA)
              hb = Conv<kernel_shape=[3,3]>(X, WB)
              merged = Concat<axis=1>(ha, hb)
              Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
            }}
            """,
            initializer=[_f32(wa_, "WA"), _f32(wb_, "WB"), _f32(wout_, "WOUT")],
        )

    model = _build(wa, wb, wout)
    onnx.checker.check_model(model)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("ha", onnx.TensorProto.FLOAT, None)
    )
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("hb", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(231)
    x_cal = rng_cal.standard_normal((4, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, ha_cal, hb_cal = _run(probe_model, {"X": x_cal})
    norm_a = np.sqrt(np.mean(np.square(ha_cal.astype(np.float64)), axis=(0, 2, 3)))
    norm_b = np.sqrt(np.mean(np.square(hb_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    imp_a = np.linalg.norm(wa.reshape(Ca, -1).astype(np.float64), axis=1) * np.maximum(
        norm_a, 1e-8
    )
    imp_b = np.linalg.norm(wb.reshape(Cb, -1).astype(np.float64), axis=1) * np.maximum(
        norm_b, 1e-8
    )
    # Each branch is exactly one block wide here, so its own per-block top-k
    # is an ordinary whole-branch top-k over this Wanda-weighted importance.
    keep_a = np.sort(np.argsort(-imp_a)[:2])
    keep_b = np.sort(np.argsort(-imp_b)[:2])
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    wout_sliced = _oracle_slice_grouped_consumer_conv(wout, global_keep, group, Ca + Cb)
    oracle = _build(wa[keep_a], wb[keep_b], wout_sliced)

    rng_x = np.random.default_rng(232)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_cpp, y_oracle, rtol=1e-5, atol=1e-5)


# --- ConvTranspose producer/consumer roles -----------------------------------
#
# Mirrors test_pruning.py's own "Conv1d / Conv3d / ConvTranspose structural
# pruning" section (test_structured_wanda_pruning_conv_transpose_producer_
# matches_oracle_exactly and the plain-structured-pruning ConvTranspose
# consumer/grouped-consumer/grouped-producer-declined tests, upgraded here to
# the calibrated Wanda entry point) -- gap (1) this module's own docstring
# used to call out: MatchConvProducer/MatchConvTransposeProducer and
# WalkToConvConsumer's own `allow_conv_transpose_consumer` branch, exercised
# through the real calibration path end to end and cross-checked against the
# pure-Python reference (which shares this exact chain-finding machinery with
# apply_structured_pruning).


def _conv_transpose_then_conv_model(w_ct, w2, spatial=10):
    # ConvTranspose *producer* -> Conv consumer: w_ct is [Cin, M, kH, kW]
    # (ConvTranspose's own reversed layout), M -- axis 1 -- is what gets
    # pruned. w2 is an ordinary Conv's [Cout, M, kH, kW] weight, consuming M
    # on its own (ordinary) axis 1.
    Cin = w_ct.shape[0]
    Cout = w2.shape[0]
    mid_spatial = spatial + 2  # ConvTranspose, unit stride, valid: in - 1 + k
    out_spatial = mid_spatial - 2  # following Conv, valid 3x3
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          h = ConvTranspose<kernel_shape=[3,3]>(X, WCT)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w_ct, "WCT"), _f32(w2, "W2")],
    )


def _conv_then_conv_transpose_model(w1, w_ct, spatial=10):
    # Conv producer -> ConvTranspose *consumer*: w1 is an ordinary Conv's
    # [C1, Cin, kH, kW] weight (C1, axis 0, pruned as always); w_ct is
    # ConvTranspose's own reversed [C1, Cout, kH, kW] weight, consuming C1 on
    # its own axis 0 -- the reverse of an ordinary Conv consumer's axis 1.
    Cin = w1.shape[1]
    Cout = w_ct.shape[1]
    mid_spatial = spatial - 2  # producer Conv, valid 3x3
    out_spatial = mid_spatial + 2  # ConvTranspose, unit stride, valid
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          Y = ConvTranspose<kernel_shape=[3,3]>(a, WCT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w_ct, "WCT")],
    )


def test_structured_wanda_pruning_cpp_conv_transpose_producer_matches_python_reference():
    # ConvTranspose acting as a *producer*: its own output channels (M, axis
    # 1 of its [Cin, M, kH, kW] weight -- the reverse of an ordinary Conv
    # producer's axis 0) must be ranked (moved to the front before the
    # ||W_row||_2 view, see MoveAxis1To0Flat) and sliced off axis 1, not
    # axis 0.
    Cin, M, Cout = 5, 8, 6
    rng = np.random.default_rng(344)
    w_ct = rng.standard_normal((Cin, M, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((Cout, M, 3, 3)).astype(np.float32)
    model = _conv_transpose_then_conv_model(w_ct, w2)
    onnx.checker.check_model(model)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(345)
    x_cal = rng_cal.standard_normal((3, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, a_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    keep_count = M - round(M * 0.5)
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["WCT"].dims) == [Cin, keep_count, 3, 3]
    assert list(inits["W2"].dims) == [Cout, keep_count, 3, 3]

    # Independent oracle -- ConvTranspose's own "N,K" importance view moves
    # its output-channel axis (1) to the front before ranking, mirroring
    # MoveAxis1To0Flat/_producer_weight_nk's own convention -- checked here
    # against a from-scratch numpy computation, not either implementation
    # under test, so this stays meaningful even once the pure-Python entry
    # point becomes a thin alias for this C++ port.
    w_ct_nk = np.moveaxis(w_ct, 1, 0).reshape(M, -1).astype(np.float64)
    importance = np.linalg.norm(w_ct_nk, axis=1) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[:keep_count])
    oracle = _conv_transpose_then_conv_model(w_ct[:, keep], w2[:, keep])
    rng_x = np.random.default_rng(3450)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_cpp, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_cpp_conv_transpose_consumer_matches_python_reference():
    # ConvTranspose acting as a *consumer*: its own input channels (axis 0 of
    # its [C1, Cout, kH, kW] weight -- the reverse of an ordinary Conv
    # consumer's axis 1) must be sliced to match the upstream Conv
    # producer's own kept output channels.
    Cin, C1, Cout = 3, 8, 6
    rng = np.random.default_rng(346)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w_ct = rng.standard_normal((C1, Cout, 3, 3)).astype(np.float32)
    model = _conv_then_conv_transpose_model(w1, w_ct)
    onnx.checker.check_model(model)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(347)
    x_cal = rng_cal.standard_normal((3, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, a_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    keep_count = C1 - round(C1 * 0.5)
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["W1"].dims) == [keep_count, Cin, 3, 3]
    assert list(inits["WCT"].dims) == [keep_count, Cout, 3, 3]

    # Independent oracle -- the producer's (ordinary Conv) own ||W_row||_2
    # ranking, weighted by the real calibration activation captured right
    # where it feeds the ConvTranspose consumer -- checked against a
    # from-scratch numpy computation, not either implementation under test.
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[:keep_count])
    oracle = _conv_then_conv_transpose_model(w1[keep], w_ct[keep])
    rng_x = np.random.default_rng(3470)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_cpp, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_cpp_grouped_conv_transpose_consumer_matches_python_reference():
    # A grouped (group > 1) ConvTranspose consumer is matched for any group
    # (unlike the producer side, restricted to group == 1) -- its own
    # input-channel axis (0) already spans the FULL in_channels regardless
    # of group, so pruning it is the same flat `w[keep, ...]` slice as the
    # group == 1 case, with the shared per-group-block top-k choosing a
    # uniform keep count per block.
    Cin, C1, Cout, group = 3, 8, 6, 2
    rng = np.random.default_rng(348)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w_ct = rng.standard_normal((C1, Cout // group, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},10,10] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          Y = ConvTranspose<kernel_shape=[3,3], group={group}>(a, WCT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w_ct, "WCT")],
    )
    onnx.checker.check_model(model)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(349)
    x_cal = rng_cal.standard_normal((3, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, a_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    block = C1 // group
    per_group_keep = max(1, round(block * 0.5))
    keep_count = per_group_keep * group
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["W1"].dims) == [keep_count, Cin, 3, 3]
    assert list(inits["WCT"].dims) == [keep_count, Cout // group, 3, 3]
    ct_node = next(n for n in pruned_cpp.graph.node if "WCT" in n.input)
    group_attr = next(a for a in ct_node.attribute if a.name == "group")
    assert group_attr.i == group  # group itself is unchanged by consumer pruning

    # Independent oracle -- one independent Wanda-weighted top-k per
    # group-sized block of the producer's own output channels, checked
    # against a from-scratch numpy computation, not either implementation
    # under test.
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(
        np.concatenate(
            [
                np.argsort(-importance[g * block : (g + 1) * block])[:per_group_keep]
                + g * block
                for g in range(group)
            ]
        )
    )
    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},10,10] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          Y = ConvTranspose<kernel_shape=[3,3], group={group}>(a, WCT)
        }}
        """,
        initializer=[_f32(w1[keep], "W1"), _f32(w_ct[keep], "WCT")],
    )
    rng_x = np.random.default_rng(3490)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_cpp, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_cpp_grouped_conv_transpose_producer_left_untouched():
    # The mirror image of the plain-structured-pruning
    # test_structured_pruning_grouped_conv_transpose_producer_is_left_untouched:
    # a grouped (group > 1) ConvTranspose is never matched as a *producer*
    # (MatchConvTransposeProducer's own `group == 1` restriction), so a
    # chain that would otherwise look matchable is left completely untouched
    # rather than guessed at -- cross-checked against the pure-Python
    # reference, which declines it for the identical reason.
    Cin, M, group, Cout = 4, 8, 2, 6
    rng = np.random.default_rng(350)
    w_ct = rng.standard_normal((Cin, M // group, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((Cout, M, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
        {{
          h = ConvTranspose<kernel_shape=[3,3], group={group}>(X, WCT)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w_ct, "WCT"), _f32(w2, "W2")],
    )
    onnx.checker.check_model(model)

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(inits["WCT"], w_ct)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_wanda_pruning_cpp_global_sparsity_matches_python_reference_and_redistributes():
    K, H, Out = 8, 16, 4
    sparsity = 0.5
    model = _two_scale_mlp_model(
        K=K, H=H, Out=Out, big_scale=50.0, small_scale=0.5, seed=7
    )
    onnx.checker.check_model(model)

    local_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=sparsity
    )
    global_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=sparsity, global_sparsity=True
    )
    global_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=[], sparsity=sparsity, global_sparsity=True
    )
    onnx.checker.check_model(global_cpp)
    assert global_cpp.SerializeToString() == global_py.SerializeToString()

    inits_local = {t.name: t for t in local_cpp.graph.initializer}
    inits_global = {t.name: t for t in global_cpp.graph.initializer}
    assert inits_local["W1big"].dims[1] == H // 2
    assert inits_local["W1small"].dims[1] == H // 2
    big_kept = inits_global["W1big"].dims[1]
    small_kept = inits_global["W1small"].dims[1]
    assert big_kept > H // 2 > small_kept
