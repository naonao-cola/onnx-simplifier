"""Tests for ``onnxsim.apply_rptq_reorder`` (RPTQ, see ``onnxsim/rptq.py``)
-- clusters a MatMul/Gemm layer's input channels by their own calibration
range and reorders them (plus the weight's matching rows) so same-cluster
channels sit contiguously, an exact pre-quantization identity
(``Gather(X, perm, axis=-1) @ Gather(W, perm, axis=0) == X @ W``) meant to
run ahead of a separate per-cluster-aware W8A8 quantizer.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.rptq import RptqLayerInfo

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _matmul_model(K=64, N=16, weight=None, seed=0):
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
        initializer=[_f32(weight, "W")],
    )


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _clustered_calibration(K=60, num_samples=64, num_clans=3, seed=1):
    # RPTQ's own motivating scenario: distinct groups of channels sharing a
    # similar magnitude within the group, but wildly different across
    # groups -- exactly what a per-tensor quantization range wastes
    # resolution on.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    clan_of = np.arange(K) % num_clans
    for clan in range(num_clans):
        scale = 10.0**clan  # 1x, 10x, 100x, ...
        x[:, clan_of == clan] *= scale
    return x, clan_of


def test_rptq_output_matches_float_almost_exactly():
    # Like SmoothQuant, RPTQ's own reorder is exact algebraic identity, not
    # an approximation -- so the tolerance is far tighter than any lossy
    # INT4/INT8 quantization scheme's.
    model = _matmul_model(K=60, N=16, seed=0)
    x, _ = _clustered_calibration(K=60, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    rptq_model, layer_info = onnxsim.apply_rptq_reorder(
        model, calibration_data=calibration_data, num_clusters=3
    )
    onnx.checker.check_model(rptq_model)
    assert any(n.op_type == "Gather" for n in rptq_model.graph.node)
    assert "X" in layer_info
    assert isinstance(layer_info["X"], RptqLayerInfo)

    float_out = _run(model, {"X": x})
    rptq_out = _run(rptq_model, {"X": x}, output_names=["Y"])
    assert np.all(np.isfinite(rptq_out["Y"]))
    assert _rel_l2(float_out["Y"], rptq_out["Y"]) < 1e-4


def test_rptq_clusters_similar_range_channels_together():
    # Channels from the same "clan" (same magnitude scale) should be
    # reordered to sit contiguously, and the resulting cluster boundaries
    # should partition the permuted axis into exactly num_clusters groups
    # that don't mix clans.
    K, num_clans = 60, 3
    model = _matmul_model(K=K, N=8, seed=2)
    x, clan_of = _clustered_calibration(
        K=K, num_samples=64, num_clans=num_clans, seed=3
    )
    calibration_data = [{"X": x}]

    _, layer_info = onnxsim.apply_rptq_reorder(
        model, calibration_data=calibration_data, num_clusters=num_clans
    )
    info = layer_info["X"]
    assert len(info.cluster_bounds) == num_clans

    perm = info.permutation
    assert sorted(perm.tolist()) == list(range(K))
    # Originally-interleaved channels (index % num_clans) must land in
    # contiguous, non-interleaved runs after reordering.
    permuted_clans = clan_of[perm]
    for start, end in info.cluster_bounds:
        segment = permuted_clans[start:end]
        assert np.all(segment == segment[0]), (
            "cluster segment mixes channels from different magnitude clans"
        )
    # Every original channel that was in the same clan should have ended up
    # in the same cluster segment as every other channel from that clan.
    seen_clan_per_segment = [permuted_clans[start] for start, _ in info.cluster_bounds]
    assert len(set(seen_clan_per_segment)) == num_clans


def test_rptq_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    x, _ = _clustered_calibration(K=K, num_samples=32, seed=9)
    calibration_data = [{"X": x}]

    rptq_model, layer_info = onnxsim.apply_rptq_reorder(
        model, calibration_data=calibration_data, num_clusters=4
    )
    onnx.checker.check_model(rptq_model)
    assert "X" in layer_info

    float_out = _run(model, {"X": x})
    rptq_out = _run(rptq_model, {"X": x}, output_names=["Y"])
    assert _rel_l2(float_out["Y"], rptq_out["Y"]) < 1e-4


def test_rptq_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result, layer_info = onnxsim.apply_rptq_reorder(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
    assert layer_info == {}


def test_rptq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result, layer_info = onnxsim.apply_rptq_reorder(
        model, calibration_data=[{"X": np.zeros((4, 64), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
    assert layer_info == {}


def test_rptq_skips_non_2d_activation():
    # A 3-D activation (e.g. [batch, seq, hidden], typical of an
    # ONNX-exported transformer) isn't a plain 2-D tensor -- matches this
    # module's own documented scope, same as onnxsim.apply_smoothquant.
    K, N = 16, 8
    rng = np.random.default_rng(10)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    x = rng.standard_normal((2, 3, K)).astype(np.float32)
    result, layer_info = onnxsim.apply_rptq_reorder(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()
    assert layer_info == {}
