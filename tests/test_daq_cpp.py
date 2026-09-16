"""Tests for ``onnxsim.apply_daq_cpp`` -- the C++-backed port of
``onnxsim.apply_daq`` (see ``onnxsim/daq_entry.h``). Unlike the
calibration-driven two-model ports (GPTQ, Qronos), DAQ is data-free: it
never runs either model and needs no ``ModelExecutor``/``onnxruntime`` at
all, so these tests never import ``onnxruntime`` and just compare
protobuf-level output directly.

Unlike a plain per-block max/min-derived scale (every GGUF-family port's
own shape), DAQ's own scale comes from an *argmax* over 18 grid
candidates, scored by a whole-tensor cosine-similarity/sign-preservation
reduction. That argmax is genuinely sensitive to floating-point
summation order whenever two candidates' scores are close: numpy's own
``np.dot``/``np.linalg.norm`` use pairwise (tree) summation for larger
arrays, while this port's own C++ kernel sums sequentially, so the two
can round differently in the last few bits and -- on rare, ordinary
(non-adversarial) inputs -- tip the argmax to a different candidate
multiplier entirely, not just a last-ulp difference in the final
reconstructed weight. So, unlike test_qronos_cpp.py's exact-agreement
contract (whose kernels have no such argmax-over-a-reduction step), this
file checks the reconstructed weight with a loose ``allclose`` tolerance
rather than bit-for-bit equality -- matching this repo's own established
contract for a port whose underlying scheme has a documented,
accepted floating-point-order sensitivity (see gguf_q2_k.h's own
"ACCEPTED, PERMANENT DIVERGENCE" precedent, and daq_entry.h's own).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.daq import apply_daq
from onnxsim.deepseek_fp8 import _fp8_round_trip
from onnxsim.onnx_simplifier import apply_daq_cpp


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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _two_models(K=64, N=16, seed=0, delta_scale=0.05):
    rng = np.random.default_rng(seed)
    w_base = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    delta = (rng.standard_normal((K, N)) * delta_scale).astype(np.float32)
    w_post = (w_base + delta).astype(np.float32)

    base_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w_base, "W")],
    )
    post_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w_post, "W")],
    )
    return base_model, post_model, w_base, w_post


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def _assert_close_parity(base_model, post_model, **kwargs):
    # Loose tolerance, not bit-for-bit equality -- see this module's own
    # docstring on why DAQ's own argmax-over-a-reduction scale search is
    # floating-point-order sensitive in a way the closed-form GGUF-family
    # ports are not.
    py = apply_daq(base_model, post_model, **kwargs)
    cpp = apply_daq_cpp(base_model, post_model, **kwargs)
    onnx.checker.check_model(cpp)
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        assert list(a.dims) == list(b.dims), a.name
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(a),
            onnx.numpy_helper.to_array(b),
            rtol=1e-2,
            atol=1e-3,
            err_msg=a.name,
        )
    return cpp


def test_daq_cpp_matches_python_cosine_metric():
    base_model, post_model, _w_base, _w_post = _two_models(seed=0)
    _assert_close_parity(base_model, post_model, metric="cosine")


def test_daq_cpp_matches_python_sign_preservation_metric():
    base_model, post_model, _w_base, _w_post = _two_models(seed=1)
    _assert_close_parity(base_model, post_model, metric="sign_preservation")


def test_daq_cpp_matches_python_across_shapes_and_seeds():
    for K, N, seed in [(128, 32, 5), (32, 8, 11), (96, 24, 21)]:
        base_model, post_model, _wb, _wp = _two_models(K=K, N=N, seed=seed)
        _assert_close_parity(base_model, post_model)


def test_daq_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 64, 12
    w_base = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    w_post = (w_base + rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    base_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(w_base, "W")],
    )
    post_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(w_post, "W")],
    )
    _assert_close_parity(base_model, post_model)


def test_daq_cpp_preserves_delta_better_than_naive_reconstruction_scale():
    # DAQ's own point: a scale chosen to minimize ||W_post - W_hat|| can
    # wipe out a large fraction of the fine-tuning delta dW, even though
    # it looks fine on raw reconstruction error. Verify the C++ port's own
    # chosen scale preserves dW's cosine similarity at least as well as
    # the naive absmax scale (deepseek_fp8's own choice, i.e. DAQ's own
    # alpha=0/multiplier=1 candidate) would -- not an arbitrary absolute
    # threshold: FP8's own 3-bit mantissa means the achievable cosine
    # similarity for an amplified small-magnitude delta signal is
    # modest (empirically ~0.83-0.87 across seeds here, verified directly
    # against onnxsim.daq's own apply_daq on this exact input), so what
    # actually demonstrates the technique is beating the naive baseline,
    # not clearing a high bar in absolute terms.
    base_model, post_model, w_base, w_post = _two_models(seed=3, delta_scale=0.02)
    cpp = apply_daq_cpp(base_model, post_model, metric="cosine")
    w_hat = _current_weight(cpp).astype(np.float64)
    w_base64 = w_base.astype(np.float64)
    w_post64 = w_post.astype(np.float64)
    delta_w = w_post64 - w_base64
    delta_w_hat = w_hat - w_base64

    def cosine(a, b):
        return float(
            np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b))
        )

    cos_sim = cosine(delta_w, delta_w_hat)

    # The naive (DAQ's own multiplier=1) absmax scale -- reuses
    # onnxsim.deepseek_fp8's own already-verified FP8 round-trip helper.
    scale0 = max(float(np.max(np.abs(w_post64))), 1e-12) / 448.0
    w_hat_naive = _fp8_round_trip(w_post64 / scale0) * scale0
    cos_sim_naive = cosine(delta_w, w_hat_naive - w_base64)
    assert cos_sim >= cos_sim_naive - 1e-9


def test_daq_cpp_skips_layer_with_no_finetuning_delta():
    # base_model and post_model share the exact same weight -- dW is
    # exactly zero, so the layer must be left completely untouched.
    rng = np.random.default_rng(4)
    K, N = 64, 16
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    base_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    post_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    cpp = apply_daq_cpp(base_model, post_model)
    assert cpp.SerializeToString() == post_model.SerializeToString()


def test_daq_cpp_skips_layer_missing_from_base_model():
    rng = np.random.default_rng(5)
    K, N = 64, 16
    w_post = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    base_model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    post_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w_post, "W")],
    )
    cpp = apply_daq_cpp(base_model, post_model)
    assert cpp.SerializeToString() == post_model.SerializeToString()


def test_daq_cpp_respects_skip_names():
    base_model, post_model, _wb, _wp = _two_models(seed=6)
    cpp = apply_daq_cpp(base_model, post_model, skip_names={"W"})
    assert cpp.SerializeToString() == post_model.SerializeToString()


def test_daq_cpp_rejects_unknown_metric():
    base_model, post_model, _wb, _wp = _two_models(seed=7)
    with pytest.raises(Exception):
        apply_daq_cpp(base_model, post_model, metric="bogus")


def test_daq_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    cpp = apply_daq_cpp(model, model)
    assert cpp.SerializeToString() == model.SerializeToString()
