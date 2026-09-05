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

NOT an alias for the pure-Python ``onnxsim.apply_structured_wanda_pruning``,
despite matching it exactly on every ordinary MatMul/Conv/gated/residual/
Concat-branch/split-gated case this file's own tests below cover, plus
``importance_norm``/``global_sparsity`` (see this file's own coverage of
both): two confirmed, real scope gaps were found running the FULL
``tests/test_pruning.py`` suite through this port (not merely this file's
own hand-picked cases) --
(1) this port's own ``FindConvChains``/``MatchConvProducer`` only ever
matches a plain ``Conv`` node (``node.op_type() != "Conv"`` outright
declines), never ``ConvTranspose``, while the pure-Python
``_match_conv_producer``/``_match_conv_transpose_producer`` matches both;
and
(2) a ``Concat``-merged branch feeding a *grouped* Conv consumer, combined
with a real calibrated (Wanda) activation norm, produces a different keep
set than the pure-Python reference (``test_structured_wanda_pruning_conv_
concat_admits_block_aligned_grouped_conv_consumer`` in ``test_pruning.py``
fails against this port).
Both are pre-existing, narrower-than-Python C++-port scope decisions (not
introduced by this round's ``importance_norm``/``global_sparsity`` work),
so ``onnxsim.apply_structured_wanda_pruning`` stays a genuine, separate
pure-Python implementation rather than an alias -- see that function's own
docstring for the same note.
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
    model, _, _ = _mlp_model(K=8, H=20, Out=5, seed=80)
    rng_cal = np.random.default_rng(81)
    x_cal = rng_cal.standard_normal((24, 8)).astype(np.float32)
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


def test_structured_wanda_pruning_cpp_matches_python_reference_multi_batch():
    # Multiple calibration batches, accumulated sum-of-squares across all of
    # them -- exercises WandaCalibrationStats' own per-batch accumulation
    # loop against pruning.py's own identical `for batch in calibration_data`
    # loop.
    model, _, _ = _mlp_model(K=6, H=14, Out=3, seed=90)
    rng_cal = np.random.default_rng(91)
    calibration_data = [
        {"X": rng_cal.standard_normal((8, 6)).astype(np.float32)} for _ in range(4)
    ]

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_structured_wanda_pruning_cpp_matches_python_reference_conv():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(100)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2)

    rng_cal = np.random.default_rng(101)
    x_cal = rng_cal.standard_normal((3, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_structured_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


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

    rng_cal = np.random.default_rng(111)
    x_cal = rng_cal.standard_normal((16, Cin)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

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
