"""Tests for ``onnxsim.apply_rptq_reorder_cpp`` -- the C++-backed port of
RPTQ's own reorder step (see ``onnxsim/rptq_entry.h``). Like
``tests/test_spqr_cpp.py``, this runs the model over real calibration data
through a real ``onnxruntime``-backed executor -- never a fake/mock
executor.

``onnxsim.rptq.apply_rptq_reorder`` (the pure-Python entry point) is now a
thin wrapper around this same C++ function -- see ``onnxsim/rptq.py`` and
``tests/test_rptq.py`` (which exercises the identical public contract
through that wrapper). There is no longer a separate pure-Python
implementation to compare against for parity; this file instead verifies
the C++ port's own properties directly: the permutation is always a valid
bijection, cluster boundaries partition it correctly on well-separated
calibration data, the reorder is numerically lossless, and edge cases
(missing calibration input, no matching layer, empty calibration data)
behave as documented in ``rptq_entry.h``.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_rptq_reorder_cpp
from onnxsim.rptq import RptqLayerInfo

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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
        weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _clustered_calibration(K=60, num_samples=64, num_clans=3, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    clan_of = np.arange(K) % num_clans
    for clan in range(num_clans):
        x[:, clan_of == clan] *= 10.0**clan
    return x, clan_of


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


def test_cpp_permutation_is_valid_bijection_and_reconstruction_exact():
    K, N = 60, 16
    model = _matmul_model(K=K, N=N, seed=0)
    x, _ = _clustered_calibration(K=K, num_samples=64, seed=1)

    out, layer_info = apply_rptq_reorder_cpp(
        model, calibration_data=[{"X": x}], num_clusters=3
    )
    onnx.checker.check_model(out)
    assert "X" in layer_info
    info = layer_info["X"]
    assert isinstance(info, RptqLayerInfo)
    assert sorted(info.permutation.tolist()) == list(range(K))

    float_out = _run(model, {"X": x})
    reordered_out = _run(out, {"X": x}, output_names=["Y"])
    assert np.all(np.isfinite(reordered_out["Y"]))
    assert _rel_l2(float_out["Y"], reordered_out["Y"]) < 1e-4


def test_cpp_cluster_bounds_partition_well_separated_clans():
    K, num_clans = 60, 3
    model = _matmul_model(K=K, N=8, seed=2)
    x, clan_of = _clustered_calibration(
        K=K, num_samples=64, num_clans=num_clans, seed=3
    )

    _, layer_info = apply_rptq_reorder_cpp(
        model, calibration_data=[{"X": x}], num_clusters=num_clans
    )
    info = layer_info["X"]
    assert len(info.cluster_bounds) == num_clans

    perm = info.permutation
    permuted_clans = clan_of[perm]
    for start, end in info.cluster_bounds:
        segment = permuted_clans[start:end]
        assert np.all(segment == segment[0])
    seen = {permuted_clans[start] for start, _ in info.cluster_bounds}
    assert len(seen) == num_clans


def test_cpp_gemm_transb_matches_and_gathers():
    rng = np.random.default_rng(8)
    K, N = 48, 12
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
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

    out, layer_info = apply_rptq_reorder_cpp(
        model, calibration_data=[{"X": x}], num_clusters=4
    )
    onnx.checker.check_model(out)
    assert "X" in layer_info
    assert any(n.op_type == "Gather" for n in out.graph.node)

    float_out = _run(model, {"X": x})
    reordered_out = _run(out, {"X": x}, output_names=["Y"])
    assert _rel_l2(float_out["Y"], reordered_out["Y"]) < 1e-4


def test_cpp_num_clusters_larger_than_channel_count():
    # num_clusters clamped to the channel count (K), same as k = min(
    # num_clusters, n) in the k-means routine -- every channel its own
    # singleton cluster, still a valid, still-lossless permutation.
    K, N = 5, 3
    model = _matmul_model(K=K, N=N, seed=11)
    x, _ = _clustered_calibration(K=K, num_samples=8, num_clans=1, seed=12)
    out, layer_info = apply_rptq_reorder_cpp(
        model, calibration_data=[{"X": x}], num_clusters=64
    )
    onnx.checker.check_model(out)
    info = layer_info["X"]
    assert sorted(info.permutation.tolist()) == list(range(K))
    assert len(info.cluster_bounds) <= K


def test_cpp_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    out, layer_info = apply_rptq_reorder_cpp(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert out.SerializeToString() == model.SerializeToString()
    assert layer_info == {}


def test_cpp_empty_calibration_data_leaves_model_unchanged():
    model = _matmul_model(K=16, N=4, seed=13)
    out, layer_info = apply_rptq_reorder_cpp(model, calibration_data=[])
    assert out.SerializeToString() == model.SerializeToString()
    assert layer_info == {}


def test_cpp_missing_graph_input_raises_value_error():
    model = _matmul_model(K=16, N=4, seed=14)
    with pytest.raises(ValueError):
        apply_rptq_reorder_cpp(
            model, calibration_data=[{"WrongName": np.zeros((2, 16), dtype=np.float32)}]
        )
