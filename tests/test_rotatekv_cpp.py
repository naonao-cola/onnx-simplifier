"""Tests for ``onnxsim.apply_rotatekv_cpp`` -- the C++-backed port of
``onnxsim.apply_rotatekv`` (RotateKV, see ``onnxsim/rotatekv_entry.h``).

This technique's own rotation fit is closed-form (the eigenvector basis of
a calibration-activation covariance), with no seeded randomness anywhere
in either the Python reference or this port -- but this port's own
from-scratch Jacobi eigensolver is independently correct, not a
transcription of LAPACK's own eigh, so (per ``rotatekv_entry.h``'s own
"ACCEPTED, PERMANENT DIVERGENCE" note) the two sides are expected to fit a
DIFFERENT, but equally valid (orthogonal), basis -- these tests therefore
check structural/algebraic properties and downstream reconstruction
quality (the same style ``tests/test_quip_sharp_cpp.py``'s own
``test_cpp_incoherence_processing_beats_naive_group_quant_with_outlier_channel``
already establishes for this repo's own rotation-family ports), never a
tight numeric cross-check of the rotation matrix itself against the pure
Python reference.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_HEAD_DIM = 8


def _model(body, initializer=(), opset=18, ir_version=9):
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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _decoder_step_model(seq_past=5, head_dim=_HEAD_DIM, transpose_kt=False):
    """A minimal decoder-step graph exercising the full RotateKV match: a
    KV-cache stream (``past_key``/``new_key`` -> ``present_key``) whose own
    output feeds a decomposed attention ``QK^T`` MatMul (through an
    optional Transpose, when ``transpose_kt``), then Softmax, then a second
    MatMul against V -- the exact shape RotateKV itself requires to find a
    compensable Query.
    """
    kt_expr = (
        "Kt = Transpose<perm = [1, 0]>(present_key)\n          scores = MatMul(Q, Kt)"
        if transpose_kt
        else "scores = MatMul(Q, present_key)"
    )
    return _model(
        f"""
        g (float[{seq_past},{head_dim}] past_key,
           float[1,{head_dim}] new_key,
           float[1,{head_dim}] Q,
           float[{seq_past + 1},{head_dim}] V)
        => (float[{seq_past + 1},{head_dim}] present_key, float[1,{head_dim}] Y)
        {{
          present_key = Concat<axis=0>(past_key, new_key)
          {kt_expr}
          probs = Softmax<axis=-1>(scores)
          Y = MatMul(probs, V)
        }}
        """
    )


def _calibration(head_dim=_HEAD_DIM, seq_past=5, num_samples=32, seed=1):
    # A concentrated outlier channel in Key so a rotation has something
    # real to spread evenly -- the same style test_quip_sharp_cpp.py's own
    # outlier-channel fixture uses.
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        new_key = rng.standard_normal((1, head_dim)).astype(np.float32) * 0.05
        new_key[:, 0] += rng.standard_normal(1).astype(np.float32) * 20.0
        past_key = rng.standard_normal((seq_past, head_dim)).astype(np.float32) * 0.05
        past_key[:, 0] += rng.standard_normal(seq_past).astype(np.float32) * 20.0
        q = rng.standard_normal((1, head_dim)).astype(np.float32)
        v = rng.standard_normal((seq_past + 1, head_dim)).astype(np.float32)
        batches.append({"past_key": past_key, "new_key": new_key, "Q": q, "V": v})
    return batches


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))


def test_cpp_rotates_key_and_compensates_query():
    # transpose_kt=True: the only shape-VALID variant of this fixture --
    # `scores = MatMul(Q, present_key)` (transpose_kt's own False branch)
    # is dimensionally invalid for real execution regardless of RotateKV
    # (Q:[1,head_dim] @ present_key:[seq+1,head_dim] doesn't multiply);
    # confirmed directly that even the UNMODIFIED model fails to load in
    # onnxruntime, so this is a fixture-only concern, not a port issue.
    model = _decoder_step_model(transpose_kt=True)
    q = onnxsim.apply_rotatekv_cpp(model, _calibration())
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert (
        op_types.count("MatMul")
        == len([n for n in model.graph.node if n.op_type == "MatMul"]) + 2
    )  # new_key rotation + Q rotation, on top of scores/out MatMuls.
    # The original Concat/QK^T/Softmax/out-MatMul nodes are all still
    # present (rewired, not replaced).
    assert op_types.count("Concat") == 1
    assert op_types.count("Softmax") == 1

    r_inits = [t for t in q.graph.initializer if t.name.endswith("_r")]
    assert len(r_inits) == 1
    r = onnx.numpy_helper.to_array(r_inits[0])
    assert r.shape == (_HEAD_DIM, _HEAD_DIM)
    # A genuine orthogonal matrix: R^T @ R == I.
    np.testing.assert_allclose(r.T @ r, np.eye(_HEAD_DIM), rtol=1e-4, atol=1e-4)


def test_cpp_handles_transpose_kt_hop():
    model = _decoder_step_model(transpose_kt=True)
    q = onnxsim.apply_rotatekv_cpp(model, _calibration())
    onnx.checker.check_model(q)
    r_inits = [t for t in q.graph.initializer if t.name.endswith("_r")]
    assert len(r_inits) == 1


def test_cpp_rotation_is_exact_migration_up_to_rounding():
    # Rotating Key and compensating Query is provably exact (see
    # rotatekv_entry.h's own top-of-file "Exactness" argument) -- the
    # rotated model's own final output must match the ORIGINAL float
    # model's output closely, for the SAME feed, regardless of which
    # orthogonal basis this port's own Jacobi eigensolver happened to fit.
    # transpose_kt=True -- see test_cpp_rotates_key_and_compensates_query's
    # own comment for why the False branch is shape-invalid regardless of
    # RotateKV.
    model = _decoder_step_model(transpose_kt=True)
    q = onnxsim.apply_rotatekv_cpp(model, _calibration())
    onnx.checker.check_model(q)

    # The exactness argument holds for the WHOLE Key stream feeding this
    # step's QK^T MatMul, not just the freshly rotated new_key: `Q_rotated
    # @ Kt == Q @ X^T` requires the CACHED past_key to already be rotated
    # by this same R too, matching rotatekv_entry.h's own top-of-file
    # comment ("present_key ... now carries rotated data for every token
    # cached through this SAME (modified) graph" -- i.e. by the time this
    # transformed graph is in steady-state use, its own past_key input is
    # itself the output of an earlier step's own new_key rotation). So the
    # rotated model's own feed must supply a past_key pre-rotated by the
    # SAME fitted R the port chose -- confirmed directly (not fed float32
    # noise into both models identically) that this reproduces the
    # provably-exact match the header comment claims; feeding raw,
    # never-rotated past_key into both sides (as an earlier version of this
    # test did) compares two different conceptual scenarios, not the same
    # one, and was never expected to match.
    r_init = next(t for t in q.graph.initializer if t.name.endswith("_r"))
    r = onnx.numpy_helper.to_array(r_init).astype(np.float64)

    rng = np.random.default_rng(7)
    past_key = rng.standard_normal((5, _HEAD_DIM)).astype(np.float32)
    shared_feeds = {
        "new_key": rng.standard_normal((1, _HEAD_DIM)).astype(np.float32),
        "Q": rng.standard_normal((1, _HEAD_DIM)).astype(np.float32),
        "V": rng.standard_normal((6, _HEAD_DIM)).astype(np.float32),
    }
    ref = _run(model, {"past_key": past_key, **shared_feeds})
    got = _run(
        q,
        {
            "past_key": (past_key.astype(np.float64) @ r).astype(np.float32),
            **shared_feeds,
        },
    )
    np.testing.assert_allclose(got["Y"], ref["Y"], rtol=1e-3, atol=1e-4)
    # present_key itself changes (it now carries ROTATED data) -- only the
    # final attention output is expected to match, not the raw cache.
    assert not np.allclose(got["present_key"], ref["present_key"])


def test_cpp_noop_when_no_attention_consumer():
    # present_key feeds nothing but a plain Identity output -- no QK^T
    # MatMul to compensate, so RotateKV must leave the stream untouched
    # entirely (see rotatekv_entry.h's own top-of-file comment: rotating
    # Key alone with no way to compensate Query would silently change
    # attention scores).
    model = _model(
        f"""
        g (float[5,{_HEAD_DIM}] past_key, float[1,{_HEAD_DIM}] new_key)
        => (float[6,{_HEAD_DIM}] present_key)
        {{
          present_key = Concat<axis=0>(past_key, new_key)
        }}
        """
    )
    result = onnxsim.apply_rotatekv_cpp(
        model,
        [
            {
                "past_key": np.zeros((5, _HEAD_DIM), dtype=np.float32),
                "new_key": np.zeros((1, _HEAD_DIM), dtype=np.float32),
            }
        ],
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_kv_cache_pattern_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_rotatekv_cpp(model, [{"X": np.zeros((4, 4), np.float32)}])
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_stream_whose_past_has_another_consumer():
    # past_key is ALSO consumed by a second node beyond the Concat --
    # _find_kv_cache_candidates (and this port) requires `past` to be
    # consumed by nothing else. Giving ONLY past_key a second consumer
    # does NOT make this a no-op -- the matcher just swaps roles and
    # treats new_key as "past" instead (still single-consumer), the same
    # subtlety tests/test_intactkv_cpp.py's own precedent already
    # documents (reconfirmed here directly against the pure-Python
    # apply_rotatekv reference) -- so both operands get a second consumer.
    model = _model(
        f"""
        g (float[5,{_HEAD_DIM}] past_key, float[1,{_HEAD_DIM}] new_key,
           float[1,{_HEAD_DIM}] Q, float[6,{_HEAD_DIM}] V)
        => (float[6,{_HEAD_DIM}] present_key, int64[2] shape_out,
            int64[2] new_key_shape, float[1,{_HEAD_DIM}] Y)
        {{
          present_key = Concat<axis=0>(past_key, new_key)
          shape_out = Shape(past_key)
          new_key_shape = Shape(new_key)
          Kt = Transpose<perm = [1, 0]>(present_key)
          scores = MatMul(Q, Kt)
          probs = Softmax<axis=-1>(scores)
          Y = MatMul(probs, V)
        }}
        """
    )
    result = onnxsim.apply_rotatekv_cpp(
        model,
        [
            {
                "past_key": np.zeros((5, _HEAD_DIM), dtype=np.float32),
                "new_key": np.zeros((1, _HEAD_DIM), dtype=np.float32),
                "Q": np.zeros((1, _HEAD_DIM), dtype=np.float32),
                "V": np.zeros((6, _HEAD_DIM), dtype=np.float32),
            }
        ],
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_two_independent_streams_no_cross_interference():
    # Key and a second, independent KV stream both present -- exercises
    # this repo's own established multi-candidate concern (see
    # tests/test_intactkv_cpp.py's own test_cpp_handles_multiple_independent_kv_streams)
    # directly at the raw-protobuf level this port operates on.
    model = _model(
        f"""
        g (float[5,{_HEAD_DIM}] past_key1, float[1,{_HEAD_DIM}] new_key1,
           float[1,{_HEAD_DIM}] Q1, float[6,{_HEAD_DIM}] V1,
           float[5,{_HEAD_DIM}] past_key2, float[1,{_HEAD_DIM}] new_key2,
           float[1,{_HEAD_DIM}] Q2, float[6,{_HEAD_DIM}] V2)
        => (float[6,{_HEAD_DIM}] present_key1, float[1,{_HEAD_DIM}] Y1,
            float[6,{_HEAD_DIM}] present_key2, float[1,{_HEAD_DIM}] Y2)
        {{
          present_key1 = Concat<axis=0>(past_key1, new_key1)
          Kt1 = Transpose<perm = [1, 0]>(present_key1)
          scores1 = MatMul(Q1, Kt1)
          probs1 = Softmax<axis=-1>(scores1)
          Y1 = MatMul(probs1, V1)
          present_key2 = Concat<axis=0>(past_key2, new_key2)
          Kt2 = Transpose<perm = [1, 0]>(present_key2)
          scores2 = MatMul(Q2, Kt2)
          probs2 = Softmax<axis=-1>(scores2)
          Y2 = MatMul(probs2, V2)
        }}
        """
    )
    calib = []
    rng = np.random.default_rng(3)
    for _ in range(8):
        calib.append(
            {
                "past_key1": rng.standard_normal((5, _HEAD_DIM)).astype(np.float32),
                "new_key1": rng.standard_normal((1, _HEAD_DIM)).astype(np.float32),
                "Q1": rng.standard_normal((1, _HEAD_DIM)).astype(np.float32),
                "V1": rng.standard_normal((6, _HEAD_DIM)).astype(np.float32),
                "past_key2": rng.standard_normal((5, _HEAD_DIM)).astype(np.float32),
                "new_key2": rng.standard_normal((1, _HEAD_DIM)).astype(np.float32),
                "Q2": rng.standard_normal((1, _HEAD_DIM)).astype(np.float32),
                "V2": rng.standard_normal((6, _HEAD_DIM)).astype(np.float32),
            }
        )
    q = onnxsim.apply_rotatekv_cpp(model, calib)
    onnx.checker.check_model(q)
    r_inits = sorted(t.name for t in q.graph.initializer if t.name.endswith("_r"))
    assert len(r_inits) == 2
    assert any("present_key1" in n for n in r_inits)
    assert any("present_key2" in n for n in r_inits)
