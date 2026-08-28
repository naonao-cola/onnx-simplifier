"""Tests for ``onnxsim.apply_autoround`` (see ``onnxsim/autoround.py``) --
jointly optimizes rounding *and* the per-block clip range/scale for every
INT4-quantized MatMul/Gemm layer, on top of what
:mod:`onnxsim.adaround`'s rounding-only optimization can reach.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _matmul_model(weight, K, N, batch, opset=21):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )


def _decode_int4(model):
    """Decodes the DequantizeLinear(Wq, Ws)-fed MatMul/Gemm's weight back
    to a dense float array, straight from the initializer bytes -- kept
    independent of autoround.py's own math. Resolves Wq/Ws by the
    DequantizeLinear node's actual inputs rather than by scanning for "the"
    INT4/FLOAT initializer: quantize_weight_only_int4 leaves the original
    float weight initializer in the graph too (now unused, since the
    consuming node was rewired to the dequantized tensor), so a bare
    "first FLOAT initializer" scan can silently grab that dead tensor
    instead of the real (much smaller) per-block scale.
    """
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    scale_full = scale_full[tuple(slicer)]
    return codes * scale_full


def _scale_tensor(model):
    """The real per-block scale initializer, resolved via the
    DequantizeLinear node's own input -- see ``_decode_int4``'s docstring
    for why a bare "the FLOAT initializer" scan is wrong here."""
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    return next(t for t in model.graph.initializer if t.name == dq_node.input[1])


def _outlier_block_weight(
    K=32, N=4, seed=0, main_std=0.1, n_outliers=1, outlier_mag=3.0
):
    """One 32-wide (single-block) weight matrix per output channel: most
    elements are small (``main_std``) and one or a few are a moderate
    outlier (``outlier_mag``). A fixed, abs-max-derived scale is set by the
    outlier(s), leaving the small elements under-resolved -- AdaRound's
    rounding-only optimization cannot fix that (it can only nudge floor
    vs. ceil at that same fixed scale), but shrinking the block's clip
    range gives the small elements more of the quantization grid, at the
    cost of clipping the outlier(s) harder -- the situation AutoRound's
    joint clip optimization targets. (An outlier so large it dominates the
    scale *and* still sits exactly on a quantization level, as a single
    ``8.0``-with-``scale=8/7`` outlier would, leaves no room to improve --
    shrinking scale would only add clipping error there for no offsetting
    gain. ``outlier_mag=3.0`` with ``main_std=0.1`` avoids that: the
    outlier isn't already exactly represented, so there is real headroom.)
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((K, N)) * main_std
    idx = rng.choice(K, size=n_outliers, replace=False)
    signs = rng.choice([-1.0, 1.0], size=(n_outliers, N))
    w[idx, :] = signs * outlier_mag
    return w.astype(np.float32)


def test_autoround_beats_adaround_reconstruction_on_outlier_block():
    K, N, batch = 32, 4, 64
    weight = _outlier_block_weight(K=K, N=N, seed=10)
    float_model = _matmul_model(weight, K, N, batch)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)

    rng = np.random.default_rng(11)
    x = rng.standard_normal((batch, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    w_float = weight.astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=200
    )
    w_ada = _decode_int4(adaround_model)
    ada_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_ada)

    autoround_model = onnxsim.apply_autoround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=200
    )
    w_auto = _decode_int4(autoround_model)
    auto_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_auto)

    # Empirically ~2.9% (see onnxsim/autoround.py's derivation) -- 0.98
    # leaves comfortable margin without being loose enough to pass by luck.
    assert auto_err < ada_err * 0.98


def test_autoround_never_worse_than_adaround_across_seeds():
    """The joint rounding+clip search is non-convex and, unprotected, can
    converge to a worse local optimum than AdaRound's own decoupled
    (fixed-scale) search reaches -- seed 70 below is one such case. This is
    exactly why :func:`onnxsim.apply_autoround` always compares against
    AdaRound's own result and keeps whichever is actually better (see
    ``onnxsim/autoround.py``'s docstring); this test is the regression
    guard for that safety net, not just for typical-case improvement.
    """
    K, N, batch = 32, 4, 64
    for seed in (10, 40, 70):
        weight = _outlier_block_weight(K=K, N=N, seed=seed)
        float_model = _matmul_model(weight, K, N, batch)
        quant_model = onnxsim.quantize_weight_only_int4(float_model)
        x = (
            np.random.default_rng(seed + 1)
            .standard_normal((batch, K))
            .astype(np.float32)
        )
        calibration_data = [{"X": x}]

        w_float = weight.astype(np.float64)
        y_float = x.astype(np.float64) @ w_float

        adaround_model = onnxsim.apply_adaround(
            float_model,
            quant_model,
            calibration_data=calibration_data,
            num_iterations=200,
        )
        ada_err = np.linalg.norm(
            y_float - x.astype(np.float64) @ _decode_int4(adaround_model)
        )

        autoround_model = onnxsim.apply_autoround(
            float_model,
            quant_model,
            calibration_data=calibration_data,
            num_iterations=200,
        )
        auto_err = np.linalg.norm(
            y_float - x.astype(np.float64) @ _decode_int4(autoround_model)
        )

        assert auto_err <= ada_err + 1e-6, (
            f"seed={seed}: autoround regressed vs. adaround"
        )


def test_autoround_changes_scale_unlike_adaround():
    K, N, batch = 32, 4, 64
    weight = _outlier_block_weight(K=K, N=N, seed=10)
    float_model = _matmul_model(weight, K, N, batch)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)

    rng = np.random.default_rng(11)
    calibration_data = [{"X": rng.standard_normal((batch, K)).astype(np.float32)}]

    before_scale = onnx.numpy_helper.to_array(_scale_tensor(quant_model))

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=50
    )
    after_ada_scale = onnx.numpy_helper.to_array(_scale_tensor(adaround_model))
    np.testing.assert_array_equal(before_scale, after_ada_scale)

    autoround_model = onnxsim.apply_autoround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=200
    )
    after_auto_scale = onnx.numpy_helper.to_array(_scale_tensor(autoround_model))
    assert not np.allclose(before_scale, after_auto_scale)


def test_autoround_codes_stay_in_range_and_output_finite():
    K, N, batch = 64, 16, 16
    rng = np.random.default_rng(5)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    float_model = _matmul_model(weight, K, N, batch)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)

    x = rng.standard_normal((batch, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    autoround_model = onnxsim.apply_autoround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=150
    )
    onnx.checker.check_model(autoround_model)

    wq = next(
        t
        for t in autoround_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
    )
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    assert np.all(codes >= -7) and np.all(codes <= 7)

    (float_y,) = _run(float_model, {"X": x})
    (auto_y,) = _run(autoround_model, {"X": x})
    assert np.all(np.isfinite(auto_y))
    assert np.linalg.norm(float_y - auto_y) / max(np.linalg.norm(float_y), 1e-6) < 0.5


def test_autoround_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_autoround(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
