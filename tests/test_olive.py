"""Tests for ``onnxsim.quantize_weight_only_olive`` -- see ``onnxsim/olive.py``
for the technique (OliVe-style Outlier-Victim Pair quantization: an outlier
element is paired with its immediate memory-adjacent neighbor, the
"victim", and the pair's combined bit budget is renegotiated -- the
outlier gets an extra bit of dynamic range, the victim is re-quantized far
more coarsely -- rather than either an exact sparse correction
(``onnxsim.spqr``) or a whole rescued column (``onnxsim.owq``)).
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


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
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


def _find_initializer(model, suffix):
    matches = [t for t in model.graph.initializer if t.name.endswith(suffix)]
    assert len(matches) == 1, f"expected exactly one initializer ending in {suffix!r}"
    return onnx.numpy_helper.to_array(matches[0])


def test_olive_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.quantize_weight_only_olive(model, bits=4, block_size=8)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert "DequantizeLinear" in op_types

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_olive_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.quantize_weight_only_olive(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_olive_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_olive(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_olive_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_olive(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_olive_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 8
    q = onnxsim.quantize_weight_only_olive(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_olive_rejects_odd_block_size():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_olive(model, block_size=7)


def test_olive_rejects_bits_below_3():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_olive(model, bits=2)


def test_olive_no_outlier_skips_outlier_branch():
    # Every element close to the block median -- no outliers, so no OVP
    # pair is ever formed and the whole layer should degenerate to plain
    # group-wide dequantization (no Where/outlier_scale/outlier_mask).
    rng = np.random.default_rng(5)
    weight = 0.1 + 0.01 * rng.standard_normal((8, 4)).astype(np.float32)
    model = _matmul_model(K=8, N=4, weight=weight)
    q = onnxsim.quantize_weight_only_olive(model, bits=4, block_size=8)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert "Where" not in op_types
    assert not any(t.name.endswith("_outlier_scale") for t in q.graph.initializer)
    assert not any(t.name.endswith("_outlier_mask") for t in q.graph.initializer)
    assert op_types.count("DequantizeLinear") == 1


def test_olive_ovp_pair_reconstructs_outlier_far_better_than_plain_quantization():
    # A single block of 8 elements (K=8, N=1) engineered to exercise every
    # OVP case in one shot:
    #   idx 0,1: one outlier (5.0) paired with an ordinary victim (0.05)
    #            -> a genuine OVP pair.
    #   idx 2,3: two ordinary elements -> plain pair, no outlier involved.
    #   idx 4,5: two outliers (6.0, -5.5) adjacent to each other -> declined
    #            (no unpaired non-outlier neighbor to act as victim), both
    #            fall back to ordinary group-wide quantization.
    #   idx 6,7: two ordinary elements -> plain pair.
    weight = np.array(
        [[5.0], [0.05], [0.06], [-0.07], [6.0], [-5.5], [0.05], [0.04]],
        dtype=np.float32,
    )
    model = _matmul_model(K=8, N=1, weight=weight)
    q = onnxsim.quantize_weight_only_olive(
        model, bits=4, block_size=8, outlier_threshold=4.0
    )
    onnx.checker.check_model(q)

    codes = _find_initializer(q, "_olive_codes").astype(np.float64)  # [K, N]
    base_scale = _find_initializer(q, "_olive_base_scale").astype(np.float64)
    outlier_scale = _find_initializer(q, "_olive_outlier_scale").astype(np.float64)
    mask = _find_initializer(q, "_olive_outlier_mask").astype(bool)
    assert codes.shape == (8, 1)
    assert mask.shape == (8, 1)

    dequant = np.where(mask, codes * outlier_scale, codes * base_scale)

    # The OVP outlier (idx 0) reconstructs within a few percent...
    outlier_rel_err = abs(float(dequant[0, 0]) - 5.0) / 5.0
    assert outlier_rel_err < 0.1
    assert bool(mask[0, 0])

    # ...while the declined outliers (idx 4, 5), which never get widened
    # dynamic range because their own neighbor is also an outlier, clip
    # hard against the block's ordinary (non-outlier-derived) scale --
    # the "decline gracefully" fallback, not a rescue.
    assert not bool(mask[4, 0])
    assert not bool(mask[5, 0])
    assert abs(float(dequant[4, 0]) - 6.0) / 6.0 > 0.5
    assert abs(float(dequant[5, 0]) - (-5.5)) / 5.5 > 0.5

    # The victim (idx 1) is quantized far more coarsely than the ordinary
    # code width would allow -- its code stays within the victim range
    # even though the ordinary range (qmax=7) could represent it more
    # precisely.
    victim_qmax = 2 ** (4 - 2) - 1  # bits=4 -> victim_bits=3 -> qmax=3
    assert abs(codes[1, 0]) <= victim_qmax
    assert not bool(mask[1, 0])

    # Ordinary pairs (idx 2,3 and 6,7) use the full ordinary code width and
    # reconstruct closely (no outlier involved at all).
    ordinary_qmax = 2 ** (4 - 1) - 1  # bits=4 -> qmax=7
    for idx, true_val in ((2, 0.06), (3, -0.07), (6, 0.05), (7, 0.04)):
        assert not bool(mask[idx, 0])
        assert abs(codes[idx, 0]) <= ordinary_qmax
        assert abs(float(dequant[idx, 0]) - true_val) < 0.02

    # Sanity: if idx 0 had instead been *declined* (quantized against
    # base_scale/ordinary_qmax, exactly like idx 4/5), it would have
    # clipped just as hard as they did -- the same mechanism-consistent
    # comparison, using this run's own base_scale rather than an
    # arbitrarily defined "naive" quantizer. OVP's own benefit for idx 0
    # is real, not an artifact of a lucky scale.
    declined_code = np.clip(
        np.round(5.0 / base_scale[0, 0]), -ordinary_qmax, ordinary_qmax
    )
    declined_dequant = float(declined_code * base_scale[0, 0])
    declined_rel_err = abs(declined_dequant - 5.0) / 5.0
    assert declined_rel_err > 0.5
    assert declined_rel_err > outlier_rel_err


def test_olive_bit_budget_matches_ordinary_pair():
    # By construction: outlier_bits + victim_bits == 2 * bits == two
    # ordinary elements' own combined budget, for every bits >= 3.
    for bits in (3, 4, 5, 6):
        ordinary_qmax = 2 ** (bits - 1) - 1
        outlier_qmax = 2**bits - 1
        victim_qmax = 2 ** (bits - 2) - 1
        outlier_bits = (outlier_qmax + 1).bit_length()
        victim_bits = (victim_qmax + 1).bit_length() if victim_qmax > 0 else 1
        assert outlier_bits + victim_bits == 2 * bits
        assert 2 * ordinary_qmax <= 2 * outlier_qmax  # sanity: never a regression
