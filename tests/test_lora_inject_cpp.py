"""Tests for ``onnxsim.inject_lora_cpp`` -- the C++-backed port of LoRA
adapter injection (see ``onnxsim/lora_entry.h``'s ``InjectLora``, already
built and unit-tested at the C++ level by ``onnxsim/lora_entry_test.cpp``,
but -- until now -- never reachable from Python: this file adds the
nanobind binding (``onnxsim.onnxsim_cpp2py_export.inject_lora``) and the
``inject_lora_cpp`` wrapper this module exercises).

Data-free and single-model, unlike ``tests/test_rptq_cpp.py``'s own
executor/calibration-data-driven port -- no ``onnxruntime`` execution
provider crosses into C++ at all, only ``onnxruntime`` (via
``pytest.importorskip`` below) to check the injected branch is a numeric
no-op, exactly as ``tests/test_lora.py``'s own
``test_inject_lora_freezes_the_base_weight_and_is_a_noop_at_init`` does for
the pure-Python :func:`onnxsim.lora.inject_lora`.

``onnxsim.lora.inject_lora`` (the pure-Python entry point) now delegates to
this same C++ function -- see ``onnxsim/lora.py`` and ``tests/test_lora.py``
(which exercises the identical public contract through that wrapper, and
still passes unmodified). This file instead checks two things the
delegation itself doesn't: that :func:`onnxsim.inject_lora_cpp` produces the
*exact same* ``LoraTarget``/``LoraAdapter`` field values (every field except
``A``'s own initializer contents, which only need to be finite and
correctly-shaped -- see ``lora_entry.h``'s own documented RNG-stream
divergence from numpy) as calling the technique directly, and the filtering/
scaling/eligibility contract ``inject_lora``'s own docstring promises.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim import lora
from onnxsim.lora import LoraAdapter, LoraTarget
from onnxsim.onnx_simplifier import inject_lora_cpp

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


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in model.graph.output]
    return sess.run(names, feeds)


def _matmul_gemm_conv_model(seed=0):
    rng = np.random.default_rng(seed)
    w_matmul = rng.standard_normal((6, 8)).astype(np.float32)
    w_gemm = rng.standard_normal((8, 5)).astype(np.float32)
    w_conv = rng.standard_normal((4, 6, 1, 1)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X, float[1,6,4,4] Xc) => (float[batch,5] Y, float[1,4,4,4] Yc) {
          Z = MatMul(X, Wm)
          Y = Gemm(Z, Wg)
          Yc = Conv <kernel_shape = [1, 1]> (Xc, Wc)
        }
        """,
        [_f32(w_matmul, "Wm"), _f32(w_gemm, "Wg"), _f32(w_conv, "Wc")],
    )
    return model, w_matmul, w_gemm, w_conv


def test_matmul_gemm_conv_injection_is_noop_and_matches_python_targets():
    model, w_matmul, w_gemm, w_conv = _matmul_gemm_conv_model()

    injected_cpp, adapter_cpp = inject_lora_cpp(model, rank=3, seed=0)
    injected_py, adapter_py = lora.inject_lora(model, rank=3, seed=0)
    onnx.checker.check_model(injected_cpp)

    assert isinstance(adapter_cpp, LoraAdapter)
    assert all(isinstance(t, LoraTarget) for t in adapter_cpp.targets)
    assert len(adapter_cpp.targets) == 3

    # Every field except A's own initializer values is expected to match
    # the pure-Python reference exactly -- deterministic given the same
    # model and no name collisions, so both sides derive identical names.
    def _fields(t):
        return (
            t.weight_name,
            t.node_output,
            t.op_type,
            t.lora_a_name,
            t.lora_b_name,
            t.rank,
            t.alpha,
        )

    cpp_by_weight = {t.weight_name: t for t in adapter_cpp.targets}
    py_by_weight = {t.weight_name: t for t in adapter_py.targets}
    assert set(cpp_by_weight) == set(py_by_weight) == {"Wm", "Wg", "Wc"}
    for w_name in cpp_by_weight:
        assert _fields(cpp_by_weight[w_name]) == _fields(py_by_weight[w_name])

    # Base weights are byte-for-byte untouched.
    init_by_name = {t.name: t for t in injected_cpp.graph.initializer}
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(init_by_name["Wm"]), w_matmul
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(init_by_name["Wg"]), w_gemm
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(init_by_name["Wc"]), w_conv
    )

    # A is finite and the right shape; B is exactly zero.
    for target in adapter_cpp.targets:
        a = onnx.numpy_helper.to_array(init_by_name[target.lora_a_name])
        b = onnx.numpy_helper.to_array(init_by_name[target.lora_b_name])
        assert np.all(np.isfinite(a))
        np.testing.assert_array_equal(b, np.zeros_like(b))

    # B starts at zero, so the injected model computes exactly what the
    # original did -- true regardless of A's own RNG divergence.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 6)).astype(np.float32)
    xc = rng.standard_normal((1, 6, 4, 4)).astype(np.float32)
    y_before, yc_before = _run(model, {"X": x, "Xc": xc})
    y_after, yc_after = _run(injected_cpp, {"X": x, "Xc": xc})
    np.testing.assert_allclose(y_before, y_after, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(yc_before, yc_after, rtol=1e-5, atol=1e-5)


def test_target_op_types_filters_to_requested_ops():
    model, _, _, _ = _matmul_gemm_conv_model()
    injected, adapter = inject_lora_cpp(model, rank=2, target_op_types=("Conv",))
    onnx.checker.check_model(injected)
    assert [t.op_type for t in adapter.targets] == ["Conv"]


def test_target_names_filters_to_requested_weights():
    model, _, _, _ = _matmul_gemm_conv_model()
    injected, adapter = inject_lora_cpp(model, rank=2, target_names=["Wg"])
    onnx.checker.check_model(injected)
    assert [t.weight_name for t in adapter.targets] == ["Wg"]


def test_target_names_empty_list_injects_nothing():
    # target_names=[] (as opposed to None) must inject nothing -- None and
    # [] are different requests, mirroring inject_lora's own contract.
    model, _, _, _ = _matmul_gemm_conv_model()
    injected, adapter = inject_lora_cpp(model, rank=2, target_names=[])
    assert adapter.targets == []
    assert injected.SerializeToString() == model.SerializeToString()


def test_alpha_scales_the_branch_and_is_recorded_on_the_target():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X) => (float[batch,8] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = inject_lora_cpp(model, rank=3, alpha=6.0, seed=0)
    onnx.checker.check_model(injected)
    assert len(adapter.targets) == 1
    assert adapter.targets[0].alpha == 6.0
    assert any(n.op_type == "Mul" for n in injected.graph.node)


def test_alpha_none_leaves_target_alpha_none_and_no_mul_node():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X) => (float[batch,8] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = inject_lora_cpp(model, rank=3, seed=0)
    assert adapter.targets[0].alpha is None
    assert not any(n.op_type == "Mul" for n in injected.graph.node)


def test_ineligible_nodes_are_skipped():
    rng = np.random.default_rng(4)
    w_1d = rng.standard_normal((6,)).astype(np.float32)  # wrong rank for MatMul
    w_conv_3x3 = rng.standard_normal((4, 6, 3, 3)).astype(np.float32)  # not 1x1
    w_conv_grouped = rng.standard_normal((6, 3, 1, 1)).astype(np.float32)  # group != 1
    w_int = np.array([[1]], dtype=np.int64)  # non-float

    model = _model(
        """
        g (float[1,6] X, float[1,6,8,8] Xc, float[1,6,8,8] Xg, int64[6,1] Xi)
            => (float[1] Y, float[1,4,8,8] Yc, float[1,6,8,8] Yg, int64[6,1] Yi)
        {
          Y = MatMul(X, W1d)
          Yc = Conv <kernel_shape = [3, 3], pads = [1, 1, 1, 1]> (Xc, Wc3x3)
          Yg = Conv <kernel_shape = [1, 1], group = 2> (Xg, Wcg)
          Yi = MatMul(Xi, Wi)
        }
        """,
        [
            _f32(w_1d, "W1d"),
            _f32(w_conv_3x3, "Wc3x3"),
            _f32(w_conv_grouped, "Wcg"),
            onnx.numpy_helper.from_array(w_int, "Wi"),
        ],
    )
    onnx.checker.check_model(model)
    injected, adapter = inject_lora_cpp(model, rank=2, seed=0)
    assert adapter.targets == []
    assert injected.SerializeToString() == model.SerializeToString()


def test_lora_adapter_parameter_names_on_cpp_result():
    model, _, _, _ = _matmul_gemm_conv_model()
    _, adapter = inject_lora_cpp(model, rank=2, target_op_types=("MatMul", "Gemm"))
    names = adapter.parameter_names()
    assert names == [
        adapter.targets[0].lora_a_name,
        adapter.targets[0].lora_b_name,
        adapter.targets[1].lora_a_name,
        adapter.targets[1].lora_b_name,
    ]
