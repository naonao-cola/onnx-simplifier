"""Tests for ``onnxsim.apply_ptq4vit_quantization`` (PTQ4ViT's own "twin
uniform quantization", see ``onnxsim/ptq4vit.py``) -- splits a Softmax/GELU
output's value range at a calibration-searched threshold and quantizes each
side with its own independent uniform quantizer.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.ptq4vit import (
    _search_twin_split,
    _single_uniform_quantize_dequantize,
    _twin_quantize_dequantize,
)

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _softmax_model(opset=13):
    # 128 columns (a plausible attention key count) so that a handful of
    # calibration batches gives the search enough samples (>= 2 * n_levels,
    # see onnxsim.ptq4vit._search_twin_split) to find a real split, and so
    # the softmax has enough categories to actually skew like real
    # attention probabilities do (a 2-3 category softmax barely skews at
    # all).
    return _model(
        """
        g (float[16,128] X) => (float[16,128] Y)
        {
          Probs = Softmax<axis=-1>(X)
          Y = Identity(Probs)
        }
        """,
        opset=opset,
    )


def _gelu_model():
    # The standalone Gelu op only has a schema from opset 20 onward.
    return _model(
        """
        g (float[N] X) => (float[N] Y)
        {
          G = Gelu(X)
          Y = Identity(G)
        }
        """,
        opset=20,
        ir_version=9,
    )


def _gelu_decomposed_model():
    # The standard export decomposition: 0.5 * x * (1 + erf(x / sqrt(2))),
    # the same shape onnxsim.ibert_gelu's own test builds.
    return _model(
        """
        g (float[N] X) => (float[N] Y)
        {
          Sqrt2 = Constant<value = float[1] {1.4142135}>()
          Half = Constant<value = float[1] {0.5}>()
          One = Constant<value = float[1] {1.0}>()
          Scaled = Div(X, Sqrt2)
          Erfed = Erf(Scaled)
          Shifted = Add(Erfed, One)
          Weighted = Mul(X, Shifted)
          Gelu = Mul(Weighted, Half)
          Y = Identity(Gelu)
        }
        """
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _softmax_calibration(seed=0, batches=8):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal((16, 128)).astype(np.float32)} for _ in range(batches)
    ]


def _gelu_calibration(seed=0, batches=8, n=64):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal(n).astype(np.float32) * 3.0} for _ in range(batches)
    ]


# --- graph surgery ---------------------------------------------------------


def test_ptq4vit_wraps_softmax_output():
    model = _softmax_model()
    q = onnxsim.apply_ptq4vit_quantization(
        model, calibration_data=_softmax_calibration()
    )
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("Softmax") == 1  # left in place, not rewritten
    assert "Where" in op_types
    assert "Less" in op_types
    assert "Round" in op_types

    identity = next(n for n in q.graph.node if n.op_type == "Identity")
    where_node = next(n for n in q.graph.node if n.op_type == "Where")
    assert identity.input[0] == where_node.output[0]


def test_ptq4vit_wraps_standalone_gelu_output():
    model = _gelu_model()
    q = onnxsim.apply_ptq4vit_quantization(model, calibration_data=_gelu_calibration())
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("Gelu") == 1
    assert "Where" in op_types

    identity = next(n for n in q.graph.node if n.op_type == "Identity")
    where_node = next(n for n in q.graph.node if n.op_type == "Where")
    assert identity.input[0] == where_node.output[0]


def test_ptq4vit_wraps_decomposed_erf_gelu_output():
    model = _gelu_decomposed_model()
    q = onnxsim.apply_ptq4vit_quantization(model, calibration_data=_gelu_calibration())
    onnx.checker.check_model(q)

    # The Erf-decomposed GELU's own math is untouched (unlike
    # onnxsim.ibert_gelu, which rewrites Erf itself) -- only its final
    # output gets wrapped.
    assert any(n.op_type == "Erf" for n in q.graph.node)
    op_types = [n.op_type for n in q.graph.node]
    assert "Where" in op_types

    identity = next(n for n in q.graph.node if n.op_type == "Identity")
    where_node = next(n for n in q.graph.node if n.op_type == "Where")
    assert identity.input[0] == where_node.output[0]


def test_ptq4vit_noop_without_softmax_or_gelu():
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    q = onnxsim.apply_ptq4vit_quantization(
        model, calibration_data=_softmax_calibration()
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_ptq4vit_declines_below_opset11():
    model = _softmax_model(opset=10)
    q = onnxsim.apply_ptq4vit_quantization(
        model, calibration_data=_softmax_calibration()
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_ptq4vit_leaves_graph_output_untouched():
    # A matched node whose output is itself a graph output is left alone --
    # rewiring it would need renaming the output ValueInfoProto, which this
    # module deliberately doesn't attempt (see its own docstring).
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Probs)
        {
          Probs = Softmax<axis=-1>(X)
        }
        """
    )
    q = onnxsim.apply_ptq4vit_quantization(
        model, calibration_data=_softmax_calibration()
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_ptq4vit_defaults_to_random_calibration_when_omitted():
    model = _softmax_model()
    q = onnxsim.apply_ptq4vit_quantization(model)
    onnx.checker.check_model(q)
    assert any(n.op_type == "Where" for n in q.graph.node)


# --- numeric correctness ----------------------------------------------------


def test_ptq4vit_graph_matches_numpy_reconstruction_exactly():
    """The inserted ONNX graph must compute exactly
    :func:`onnxsim.ptq4vit._twin_quantize_dequantize` -- verified directly
    against numpy (not a loose onnxruntime-vs-numpy tolerance), since these
    are all elementwise ops with no reduction whose order could vary across
    CPU architectures.
    """
    model = _softmax_model()
    calib = _softmax_calibration()
    q = onnxsim.apply_ptq4vit_quantization(model, calibration_data=calib)

    split_init = next(t for t in q.graph.initializer if t.name.endswith("_split"))
    split = float(onnx.numpy_helper.to_array(split_init))

    rng = np.random.default_rng(3)
    feeds = {"X": rng.standard_normal((16, 128)).astype(np.float32)}
    (float_probs,) = _run(model, feeds)
    (quantized_y,) = _run(q, feeds)

    expected = _twin_quantize_dequantize(
        float_probs.astype(np.float64), 0.0, split, 1.0, n_levels=256
    )
    np.testing.assert_allclose(quantized_y, expected, rtol=1e-5, atol=1e-6)


def test_ptq4vit_search_finds_a_split_between_lo_and_hi():
    rng = np.random.default_rng(1)
    values = rng.beta(0.5, 8.0, size=50_000)
    split = _search_twin_split(values, lo=0.0, hi=1.0, n_levels=256)
    assert split is not None
    assert 0.0 < split < 1.0


def test_ptq4vit_search_returns_none_on_tiny_input():
    values = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    assert _search_twin_split(values, lo=0.0, hi=1.0, n_levels=256) is None


# --- the empirical claim this module rests on -------------------------------
#
# PTQ4ViT's own headline claim is that twin uniform quantization reduces
# reconstruction error relative to an ordinary single-scale uniform
# quantizer, on distributions shaped like real post-Softmax/post-GELU
# activations. This is verified directly here (not assumed by construction,
# per this module's own docstring and the numeric-honesty precedent
# onnxsim.ibert_gelu's own docstring documents) against synthetic data
# shaped the same way real ViT activations are: a Beta distribution skewed
# toward 0 with a thin tail near 1 for Softmax, and a bimodal mixture (a
# small negative cluster plus a much wider positive spread) for GELU.


def test_twin_quantization_beats_single_uniform_quantizer_on_softmax_shaped_data():
    rng = np.random.default_rng(0)
    values = rng.beta(0.5, 8.0, size=200_000)  # concentrated near 0, thin tail near 1
    n_levels = 256

    split = _search_twin_split(values, lo=0.0, hi=1.0, n_levels=n_levels)
    assert split is not None

    # The comparison that matters is against a single quantizer at the
    # *equal total bit budget* (2 * n_levels levels -- one extra bit spent
    # uniformly on every level, vs. twin's one extra bit spent on a side
    # selector): comparing against a single quantizer at the *same*
    # n_levels would always favor twin quantization trivially (it then
    # has roughly double the raw level count), which would not actually
    # show the split is exploiting this distribution's concentration
    # rather than just spending more bits. See
    # test_twin_quantization_does_not_always_win_on_a_flat_distribution
    # for the negative control this comparison makes possible.
    twin = _twin_quantize_dequantize(values, 0.0, split, 1.0, n_levels)
    single_equal_budget = _single_uniform_quantize_dequantize(
        values, 0.0, 1.0, 2 * n_levels
    )

    mse_twin = float(np.mean((values - twin) ** 2))
    mse_single_equal_budget = float(np.mean((values - single_equal_budget) ** 2))

    assert mse_twin < mse_single_equal_budget / 2.5


def test_twin_quantization_beats_single_uniform_quantizer_on_gelu_shaped_data():
    rng = np.random.default_rng(0)
    neg = rng.uniform(-0.17, 0.0, size=50_000)  # GELU's small negative dip
    pos = np.abs(rng.normal(2.0, 2.0, size=150_000))  # wide positive spread
    values = np.concatenate([neg, pos])
    lo, hi = float(values.min()), float(values.max())
    n_levels = 256

    split = _search_twin_split(values, lo=lo, hi=hi, n_levels=n_levels)
    assert split is not None

    twin = _twin_quantize_dequantize(values, lo, split, hi, n_levels)
    single_equal_budget = _single_uniform_quantize_dequantize(
        values, lo, hi, 2 * n_levels
    )

    mse_twin = float(np.mean((values - twin) ** 2))
    mse_single_equal_budget = float(np.mean((values - single_equal_budget) ** 2))

    assert mse_twin < mse_single_equal_budget / 1.2


def test_twin_quantization_does_not_always_win_on_a_flat_distribution():
    # Honesty check (this module's own docstring says explicitly that twin
    # quantization is not a free win "by construction"): at an equal total
    # bit budget, a uniform (flat) distribution has no concentration for a
    # split to exploit, so the search finds no split that clears the
    # equal-budget bar -- unlike the skewed/bimodal distributions above,
    # where it clears that same bar by a wide margin.
    rng = np.random.default_rng(2)
    values = rng.uniform(0.0, 1.0, size=50_000)
    split = _search_twin_split(values, lo=0.0, hi=1.0, n_levels=256)
    assert split is None
