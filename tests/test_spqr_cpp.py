"""Tests for ``onnxsim.apply_spqr_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_spqr`` (SpQR, see ``onnxsim/spqr_entry.h``).
Like ``tests/test_llm_int8_cpp.py``/``tests/test_gptq_cpp.py``, this runs
the model over real calibration data through a real ``onnxruntime``-backed
executor -- never a fake/mock executor.

Unlike those two files' own exact (bit-for-bit) node-list parity checks,
this port has one documented, accepted divergence from the pure-Python
reference: outlier-position SELECTION ORDER (see ``spqr_entry.h``'s own
"ACCEPTED, PERMANENT DIVERGENCE" note -- a full deterministic sort here,
vs. numpy's own unspecified ``argpartition`` order there). Both sides
select the exact same SET of outlier positions and compute numerically
identical reconstructed weights; only the outlier_indices/outlier_values
initializers' own row order can differ. So these tests check the outlier
position SET (not raw initializer row order) and the reconstructed
numeric output (via onnxruntime), which is expected to match tightly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_spqr_cpp
from onnxsim.spqr import quantize_weight_only_spqr

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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


def _matmul_model(K=32, N=8, seed=0, opset=21):
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
        opset=opset,
    )


def _calibration(K=32, num_samples=16, seed=1, outlier_col=3, outlier_scale=25.0):
    # A wide-magnitude activation column concentrates sensitivity on the
    # weight rows sharing that column (h_k dominated by that one
    # channel), giving a real, non-trivial outlier set to exercise the
    # sparse-correction machinery, not just the plain block-quantize path.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, outlier_col] *= outlier_scale
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _outlier_position_set(model, block_size, n):
    ws_t = next(t for t in model.graph.initializer if t.name.endswith("_scale"))
    k = ws_t.dims[0] * block_size
    assert (ws_t.dims[0], ws_t.dims[1]) == (k // block_size, n)
    indices_t = next(
        (t for t in model.graph.initializer if t.name.endswith("_outlier_indices")),
        None,
    )
    if indices_t is None:
        return set()
    idx = onnx.numpy_helper.to_array(indices_t)
    return {(int(row[0]), int(row[1])) for row in idx}  # {(k_pos, n_pos)}


def test_cpp_matches_python_outlier_set_and_output():
    K, N, block_size = 32, 8, 16
    model = _matmul_model(K=K, N=N, seed=0)
    calib = _calibration(K=K)

    py = quantize_weight_only_spqr(
        model, block_size=block_size, outlier_fraction=0.05, calibration_data=calib
    )
    cpp = apply_spqr_cpp(
        model, block_size=block_size, outlier_fraction=0.05, calibration_data=calib
    )
    onnx.checker.check_model(py)
    onnx.checker.check_model(cpp)

    py_outliers = _outlier_position_set(py, block_size, N)
    cpp_outliers = _outlier_position_set(cpp, block_size, N)
    assert cpp_outliers == py_outliers
    assert len(cpp_outliers) > 0  # a real outlier set, not the empty-case path

    rng = np.random.default_rng(9)
    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(cpp_y, py_y, rtol=1e-4, atol=1e-5)


def test_cpp_reconstructs_outlier_positions_exactly():
    # Every outlier position must reconstruct to the ORIGINAL float weight
    # exactly (up to float32 rounding), regardless of how its block-quantized
    # value happened to round -- the whole point of the sparse correction.
    K, N, block_size = 32, 8, 16
    model = _matmul_model(K=K, N=N, seed=2)
    calib = _calibration(K=K, seed=3)
    cpp = apply_spqr_cpp(
        model, block_size=block_size, outlier_fraction=0.05, calibration_data=calib
    )

    orig_w = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W")
    ).astype(np.float64)

    rng = np.random.default_rng(10)
    x = rng.standard_normal((4, K)).astype(np.float32)
    x_identity = np.eye(K, dtype=np.float32)
    (recon_full,) = _run(cpp, {"X": np.concatenate([x, x_identity], axis=0)})
    # The identity rows recover Wreconstructed's own columns directly:
    # Y = I @ W == W (each row of I picks one row of W).
    recon_w = recon_full[4:]  # [K, N]

    cpp_outliers = _outlier_position_set(cpp, block_size, N)
    assert len(cpp_outliers) > 0
    for k_pos, n_pos in cpp_outliers:
        np.testing.assert_allclose(
            recon_w[k_pos, n_pos], orig_w[k_pos, n_pos], rtol=1e-3, atol=1e-4
        )


def test_cpp_no_outliers_when_fraction_zero():
    K, N, block_size = 32, 8, 16
    model = _matmul_model(K=K, N=N, seed=4)
    calib = _calibration(K=K, seed=5)
    cpp = apply_spqr_cpp(
        model, block_size=block_size, outlier_fraction=0.0, calibration_data=calib
    )
    onnx.checker.check_model(cpp)
    op_types = [n.op_type for n in cpp.graph.node]
    assert "ScatterND" not in op_types
    assert "ConstantOfShape" not in op_types
    assert "DequantizeLinear" in op_types


def test_cpp_gemm_with_bias_and_transb():
    K, N, block_size = 32, 8, 16
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.4
    bias = rng.standard_normal(N).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    calib = _calibration(K=K, seed=7)
    py = quantize_weight_only_spqr(
        model, block_size=block_size, outlier_fraction=0.05, calibration_data=calib
    )
    cpp = apply_spqr_cpp(
        model, block_size=block_size, outlier_fraction=0.05, calibration_data=calib
    )
    onnx.checker.check_model(cpp)
    assert any(
        n.op_type == "Add" and n.name.endswith("_bias_add_node") for n in cpp.graph.node
    )

    rng2 = np.random.default_rng(11)
    x = rng2.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(cpp_y, py_y, rtol=1e-4, atol=1e-5)


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_spqr_cpp(
        model, calibration_data=[{"X": np.zeros((2, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset21():
    model = _matmul_model(K=32, N=8, seed=8, opset=18)
    result = apply_spqr_cpp(model, calibration_data=_calibration(K=32, seed=8))
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_block_divisible_k():
    K, N = 32 + 3, 8  # not a multiple of block_size (16)
    model = _matmul_model(K=K, N=N, seed=12)
    result = apply_spqr_cpp(
        model, block_size=16, calibration_data=_calibration(K=K, seed=12)
    )
    assert result.SerializeToString() == model.SerializeToString()
