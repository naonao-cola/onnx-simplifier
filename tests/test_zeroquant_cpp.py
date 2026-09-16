"""Tests for ``onnxsim.apply_zeroquant_cpp`` -- the C++-backed port of
``onnxsim.apply_zeroquant`` (ZeroQuant, see ``onnxsim/passes/zeroquant.h``).
Unlike this repo's k-means-family ``*_cpp`` ports, this is a closed-form,
deterministic quantization scheme with no RNG or fitting algorithm
involved, so -- unlike ``tests/test_quarot_cpp.py``'s own tests, which
compare structure/accuracy only because QuaRot's own rotation is a
genuinely different, independently-seeded RNG construction on each side --
these tests check the C++ port's own weight-quantization math and
end-to-end numeric output directly against ``onnxsim.zeroquant``'s pure
-Python reference, expecting a tight match.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.zeroquant import _MAX_SAFE_GROUP_SIZE, _quantize_weight_groupwise_int8

ort = pytest.importorskip("onnxruntime")

_STANDARD_OPS = {
    "MatMul",
    "Gemm",
    "Shape",
    "Gather",
    "Concat",
    "Reshape",
    "Abs",
    "ReduceMax",
    "Max",
    "Div",
    "Round",
    "Clip",
    "Cast",
    "Split",
    "MatMulInteger",
    "Mul",
    "Sum",
    "Add",
    "Slice",
    "Identity",
}


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


def _matmul_model(K=64, N=8, weight=None, seed=0, opset=21):
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
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def test_cpp_replaces_matmul_with_real_matmulinteger():
    model = _matmul_model(K=64, N=8, seed=0)
    q = onnxsim.apply_zeroquant_cpp(model)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert "MatMulInteger" in op_types
    # A real int8 x int8 matmul, not a simulated round-trip: the original
    # MatMul is gone, replaced entirely by the ZeroQuant pipeline.
    assert "MatMul" not in op_types
    assert op_types <= _STANDARD_OPS
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_cpp_single_group_weight_quantization_matches_python_reference():
    K, N, block_size = 64, 8, 64  # one group
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)

    q = onnxsim.apply_zeroquant_cpp(model, block_size=block_size)
    onnx.checker.check_model(q)

    wq_ref, scale_ref = _quantize_weight_groupwise_int8(
        weight.astype(np.float64), block_size, 1e-12
    )

    wq_t = next(
        t
        for t in q.graph.initializer
        if t.data_type == onnx.TensorProto.INT8 and list(t.dims) == [block_size, N]
    )
    ws_t = next(
        t
        for t in q.graph.initializer
        if t.data_type == onnx.TensorProto.FLOAT and list(t.dims) == [N]
    )
    wq_cpp = onnx.numpy_helper.to_array(wq_t)
    ws_cpp = onnx.numpy_helper.to_array(ws_t)

    np.testing.assert_array_equal(wq_cpp, wq_ref)
    np.testing.assert_allclose(ws_cpp, scale_ref[0], rtol=1e-6, atol=1e-9)


def test_cpp_multi_group_weight_quantization_matches_python_reference():
    K, N, block_size = 96, 6, 32  # 3 groups
    num_groups = K // block_size
    rng = np.random.default_rng(2)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)

    q = onnxsim.apply_zeroquant_cpp(model, block_size=block_size)
    onnx.checker.check_model(q)

    mmi_nodes = [n for n in q.graph.node if n.op_type == "MatMulInteger"]
    assert len(mmi_nodes) == num_groups

    initializer_by_name = {t.name: t for t in q.graph.initializer}
    cast_by_input = {n.input[0]: n for n in q.graph.node if n.op_type == "Cast"}
    mul_by_input = {n.input[0]: n for n in q.graph.node if n.op_type == "Mul"}

    wq_ref, scale_ref = _quantize_weight_groupwise_int8(
        weight.astype(np.float64), block_size, 1e-12
    )

    # Each MatMulInteger's own (wq, ws) pair is found by following the exact
    # graph wiring (MatMulInteger -> Cast -> Mul-by-ws) rather than sorting
    # two independently-ordered initializer lists and zipping them, which
    # could silently mispair a group's own weight codes with another
    # group's own scale.
    seen_groups = set()
    for mmi in mmi_nodes:
        wq_t = initializer_by_name[mmi.input[1]]
        wq_cpp = onnx.numpy_helper.to_array(wq_t)
        cast_node = cast_by_input[mmi.output[0]]
        mul_node = mul_by_input[cast_node.output[0]]
        ws_t = initializer_by_name[mul_node.input[1]]
        ws_cpp = onnx.numpy_helper.to_array(ws_t)

        matched_group = None
        for g in range(num_groups):
            if np.array_equal(wq_cpp, wq_ref[g * block_size : (g + 1) * block_size]):
                matched_group = g
                break
        assert matched_group is not None, (
            "no reference group matches this MatMulInteger's own weight codes"
        )
        assert matched_group not in seen_groups
        seen_groups.add(matched_group)
        np.testing.assert_allclose(
            ws_cpp, scale_ref[matched_group], rtol=1e-6, atol=1e-9
        )
    assert seen_groups == set(range(num_groups))


def test_cpp_output_matches_python_reference_and_stays_close_to_float():
    K, N, block_size = 96, 6, 32
    rng = np.random.default_rng(3)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)

    q_cpp = onnxsim.apply_zeroquant_cpp(model, block_size=block_size)
    q_py = onnxsim.apply_zeroquant(model, block_size=block_size)
    onnx.checker.check_model(q_cpp)
    onnx.checker.check_model(q_py)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (cpp_y,) = _run(q_cpp, {"X": x})
    (py_y,) = _run(q_py, {"X": x})

    assert np.all(np.isfinite(cpp_y))
    assert _rel_l2(float_y, cpp_y) < 0.2
    # No RNG on either side: the C++ port and the pure-Python reference
    # apply the exact same closed-form quantization scheme, so their
    # outputs are expected to agree tightly (up to ordinary
    # floating-point summation-order differences), unlike this repo's
    # k-means-family ports.
    np.testing.assert_allclose(cpp_y, py_y, rtol=1e-2, atol=1e-3)


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = 64, 8
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_zeroquant_cpp(model)
    onnx.checker.check_model(q)
    assert any(n.op_type == "Add" for n in q.graph.node)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 64, 8
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
    q = onnxsim.apply_zeroquant_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_handles_leading_batch_and_sequence_dims():
    # X's rank is 3 ([batch, seq, K]), not the flat 2-D shape every other
    # test above uses -- exercises the Shape/Gather/Concat/Reshape
    # flatten-to-2-D prelude and its inverse at the end.
    K, N = 64, 8
    rng = np.random.default_rng(7)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.apply_zeroquant_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((2, 3, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert q_y.shape == (2, 3, N)
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_zeroquant_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_block_divisible_k():
    # K not a multiple of block_size (32) -- zeroquant.py's own encoder
    # skips this layer entirely, and this port matches that exactly.
    model = _matmul_model(K=48, N=8, seed=8)  # 48 is not a multiple of 32
    q = onnxsim.apply_zeroquant_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset18():
    model = _matmul_model(K=64, N=8, seed=9, opset=13)
    q = onnxsim.apply_zeroquant_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_declines_block_size_exceeding_safe_group_size():
    model = _matmul_model(K=64, N=8, seed=10)
    q = onnxsim.apply_zeroquant_cpp(model, block_size=_MAX_SAFE_GROUP_SIZE + 1)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_block_size_and_epsilon_defaults_match_python():
    import inspect

    sig = inspect.signature(onnxsim.apply_zeroquant_cpp)
    assert sig.parameters["block_size"].default == 32
    assert sig.parameters["epsilon"].default == 1e-12
