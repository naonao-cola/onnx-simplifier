"""Tests for ``onnxsim.generate_xnnpack_c`` / ``export_xnnpack_c``.

Every model is built via ``onnx.helper`` (not ``onnx.parser``): these models
need real float initializer data -- Conv/Gemm weights permuted at generation
time, whose exact byte-for-byte correctness this file checks against a
NumPy-computed expected permutation -- and the ONNX text format's tensor
literals are ``float_data``, not ``raw_data`` (see CLAUDE.md's own carve-out
for exactly this case).

These are structural/unit tests only: they inspect the generated C *text*
(declared shapes, embedded weight arrays, which XNNPACK entry point a node
lowers to) without compiling or running it -- doing that for real requires an
actual XNNPACK build, which is exercised separately, by hand, against the
pinned commit in ``cmake/build_xnnpack.cmake`` (see the PR description this
test file shipped with for that validation's numeric results, run against a
Conv/depthwise-Conv/GlobalAveragePool/Gemm/Add/MatMul model on a real,
compiled XNNPACK runtime).
"""

import re

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnxsim

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _model(nodes, inputs, outputs, initializers, opset=17, ir_version=8):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir_version
    onnx.checker.check_model(model)
    return model


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _weight(shape, name, seed=0):
    rng = np.random.default_rng(seed)
    return numpy_helper.from_array(
        rng.standard_normal(shape).astype(np.float32) * 0.1, name
    )


def _extract_float_array(source: str, c_name: str):
    """Pulls the float literals out of ``static const float <c_name>[] = {...};``."""
    match = re.search(re.escape(c_name) + r"\[\]\s*=\s*\{(.*?)\};", source, re.DOTALL)
    assert match is not None, f"array '{c_name}' not found in generated source"
    return np.array(
        [
            float(tok.rstrip("f"))
            for tok in match.group(1).replace("\n", " ").split(",")
            if tok.strip()
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Conv: NHWC/filter-layout permutation
# --------------------------------------------------------------------------- #


def test_conv_regular_permutes_input_output_and_filter_to_nhwc():
    conv_w = _weight((4, 3, 3, 3), "w", seed=1)
    conv_b = _weight((4,), "b", seed=2)
    node = helper.make_node("Conv", ["x", "w", "b"], ["y"], pads=[1, 1, 1, 1])
    model = _model(
        [node], [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 4, 8, 8])], [conv_w, conv_b]
    )

    src = onnxsim.generate_xnnpack_c(model, "m")

    assert "xnn_define_convolution_2d(sg, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3, 4," in src
    # Input/output declared NHWC ([1,8,8,3] / [1,8,8,4]), not ONNX's NCHW.
    assert "size_t dims[] = {1, 8, 8, 3};" in src
    assert "size_t dims[] = {1, 8, 8, 4};" in src
    # Filter declared [OC, KH, KW, IC], not ONNX's [OC, IC, KH, KW].
    assert "size_t dims[] = {4, 3, 3, 3};" in src

    expected = numpy_helper.to_array(conv_w).transpose(0, 2, 3, 1).flatten()
    np.testing.assert_allclose(_extract_float_array(src, "g_m_w"), expected, atol=1e-6)


def test_conv_depthwise_uses_dedicated_op_and_filter_layout():
    # groups == Cin, one input channel per group -> depthwise path.
    cin, mult = 4, 2
    conv_w = _weight((cin * mult, 1, 3, 3), "dw", seed=3)
    node = helper.make_node("Conv", ["x", "dw"], ["y"], pads=[1, 1, 1, 1], group=cin)
    model = _model(
        [node], [_vi("x", [1, cin, 8, 8])], [_vi("y", [1, cin * mult, 8, 8])], [conv_w]
    )

    src = onnxsim.generate_xnnpack_c(model, "m")

    assert "xnn_define_depthwise_convolution_2d(" in src
    assert "xnn_define_convolution_2d(" not in src
    # Filter declared [1, KH, KW, Cin*mult].
    assert "size_t dims[] = {1, 3, 3, 8};" in src
    expected = numpy_helper.to_array(conv_w).transpose(1, 2, 3, 0).flatten()
    np.testing.assert_allclose(_extract_float_array(src, "g_m_dw"), expected, atol=1e-6)


def test_conv_grouped_non_depthwise_uses_regular_op_with_group_channels():
    # groups=2, 2 input channels per group -- neither ungrouped nor depthwise.
    conv_w = _weight((4, 2, 3, 3), "w", seed=4)  # Cout=4, Cin/groups=2
    node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1], group=2)
    model = _model([node], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])], [conv_w])

    src = onnxsim.generate_xnnpack_c(model, "m")
    assert "xnn_define_depthwise_convolution_2d(" not in src
    # groups=2, group_input_channels=2, group_output_channels=2.
    assert ", 2, 2, 2, -INFINITY" in src


@pytest.mark.parametrize(
    "auto_pad,expected_pads",
    [
        ("SAME_UPPER", (1, 1, 1, 1)),
        ("SAME_LOWER", (1, 1, 1, 1)),
        ("VALID", (0, 0, 0, 0)),
    ],
)
def test_conv_auto_pad_resolves_to_explicit_padding(auto_pad, expected_pads):
    conv_w = _weight((2, 3, 3, 3), "w", seed=5)
    out_hw = 7 if auto_pad == "VALID" else 7
    node = helper.make_node("Conv", ["x", "w"], ["y"], auto_pad=auto_pad)
    model = _model(
        [node], [_vi("x", [1, 3, 7, 7])], [_vi("y", [1, 2, out_hw, out_hw])], [conv_w]
    )
    src = onnxsim.generate_xnnpack_c(model, "m")
    top, right, bottom, left = expected_pads
    assert (
        f"xnn_define_convolution_2d(sg, {top}, {right}, {bottom}, {left}, 3, 3," in src
    )


def test_conv_bad_auto_pad_raises():
    conv_w = _weight((2, 3, 3, 3), "w", seed=6)
    node = helper.make_node("Conv", ["x", "w"], ["y"], auto_pad="NOTATHING")
    model = _model([node], [_vi("x", [1, 3, 7, 7])], [_vi("y", [1, 2, 5, 5])], [conv_w])
    with pytest.raises(RuntimeError, match="auto_pad"):
        onnxsim.generate_xnnpack_c(model, "m")


# --------------------------------------------------------------------------- #
# Elementwise / MatMul / GlobalAveragePool / Reshape
# --------------------------------------------------------------------------- #


def test_binary_and_unary_ops_carry_nhwc_layout_through():
    relu = helper.make_node("Relu", ["x"], ["r"])
    add = helper.make_node("Add", ["r", "bias"], ["a"])
    sigmoid = helper.make_node("Sigmoid", ["a"], ["y"])
    bias = _weight((1, 3, 1, 1), "bias", seed=7)
    model = _model(
        [relu, add, sigmoid],
        [_vi("x", [1, 3, 4, 4])],
        [_vi("y", [1, 3, 4, 4])],
        [bias],
    )
    src = onnxsim.generate_xnnpack_c(model, "m")
    assert "xnn_unary_clamp" in src  # Relu
    assert "xnn_binary_add" in src
    assert "xnn_unary_sigmoid" in src
    # The [1,3,1,1] bias broadcasts against an NHWC [1,4,4,3] activation, so
    # it must itself be declared NHWC ([1,1,1,3]), not passed through raw.
    assert "size_t dims[] = {1, 1, 1, 3};" in src


def test_matmul_untransposed():
    a_w = _weight((3, 5), "w", seed=8)
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    model = _model([node], [_vi("x", [1, 3])], [_vi("y", [1, 5])], [a_w])
    src = onnxsim.generate_xnnpack_c(model, "m")
    assert "XNN_FLAG_TRANSPOSE_WEIGHTS" in src  # w is [K,N], needs transpose flag


def test_gemm_trans_b_omits_transpose_flag():
    gemm_w = _weight(
        (5, 3), "w", seed=9
    )  # [N, K] -- already the fully_connected default
    node = helper.make_node("Gemm", ["x", "w"], ["y"], transB=1)
    model = _model([node], [_vi("x", [1, 3])], [_vi("y", [1, 5])], [gemm_w])
    src = onnxsim.generate_xnnpack_c(model, "m")
    assert "xnn_define_fully_connected(sg, -INFINITY, INFINITY, " in src
    assert re.search(r"xnn_define_fully_connected\([^;]*,\s*0\)\);", src)


def test_global_average_pool_reduces_to_2d_directly():
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"])
    model = _model([node], [_vi("x", [1, 6, 4, 4])], [_vi("y", [1, 6, 1, 1])], [])
    src = onnxsim.generate_xnnpack_c(model, "m")
    assert "xnn_define_static_reduce(sg, xnn_reduce_mean, 2, (size_t[]){1, 2}," in src
    assert "size_t dims[] = {1, 6};" in src  # declared [N, C], not [N, C, 1, 1]


def test_reshape_after_global_average_pool_is_layout_safe():
    gap = helper.make_node("GlobalAveragePool", ["x"], ["gap"])
    shape_init = numpy_helper.from_array(np.array([1, 6], dtype=np.int64), "shape")
    reshape = helper.make_node("Reshape", ["gap", "shape"], ["y"])
    model = _model(
        [gap, reshape], [_vi("x", [1, 6, 4, 4])], [_vi("y", [1, 6])], [shape_init]
    )
    src = onnxsim.generate_xnnpack_c(model, "m")
    assert "xnn_define_static_reshape" in src


def test_reshape_of_multi_pixel_spatial_map_is_rejected():
    conv_w = _weight((2, 3, 3, 3), "w", seed=10)
    conv = helper.make_node("Conv", ["x", "w"], ["c1"], pads=[1, 1, 1, 1])
    shape_init = numpy_helper.from_array(np.array([1, 128], dtype=np.int64), "shape")
    reshape = helper.make_node("Reshape", ["c1", "shape"], ["y"])
    model = _model(
        [conv, reshape],
        [_vi("x", [1, 3, 8, 8])],
        [_vi("y", [1, 128])],
        [conv_w, shape_init],
    )
    with pytest.raises(RuntimeError, match="not supported in v1"):
        onnxsim.generate_xnnpack_c(model, "m")


# --------------------------------------------------------------------------- #
# Errors / scope boundaries
# --------------------------------------------------------------------------- #


def test_unsupported_op_raises_naming_the_op():
    node = helper.make_node("Identity", ["x"], ["y"])
    model = _model([node], [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 3, 4, 4])], [])
    with pytest.raises(RuntimeError, match="Identity"):
        onnxsim.generate_xnnpack_c(model, "m")


def test_non_fp32_initializer_raises():
    node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    w = numpy_helper.from_array(np.zeros((2, 3, 3, 3), dtype=np.float64), "w")
    model = _model([node], [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 2, 8, 8])], [w])
    with pytest.raises(RuntimeError, match="fp32"):
        onnxsim.generate_xnnpack_c(model, "m")


def test_dynamic_shape_raises():
    node = helper.make_node("Relu", ["x"], ["y"])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3, 4, 4])
    model = _model([node], [x], [y], [])
    with pytest.raises(RuntimeError, match="concrete"):
        onnxsim.generate_xnnpack_c(model, "m")


def test_invalid_function_prefix_raises_value_error():
    node = helper.make_node("Relu", ["x"], ["y"])
    model = _model([node], [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 3, 4, 4])], [])
    with pytest.raises(ValueError, match="function_prefix"):
        onnxsim.generate_xnnpack_c(model, "1bad")


def test_two_prefixes_produce_no_symbol_collisions():
    node = helper.make_node("Relu", ["x"], ["y"])
    model = _model([node], [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 3, 4, 4])], [])
    src_a = onnxsim.generate_xnnpack_c(model, "model_a")
    src_b = onnxsim.generate_xnnpack_c(model, "model_b")
    assert "model_a_create" in src_a and "model_b_create" not in src_a
    assert "model_b_create" in src_b and "model_a_create" not in src_b


def test_export_xnnpack_c_writes_file(tmp_path):
    node = helper.make_node("Relu", ["x"], ["y"])
    model = _model([node], [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 3, 4, 4])], [])
    path = tmp_path / "model.c"
    onnxsim.export_xnnpack_c(model, str(path), "m")
    assert path.read_text().startswith("/* Generated by onnxsim")
