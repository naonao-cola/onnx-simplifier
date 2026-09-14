"""Tests for ``onnxsim.check_webgpu_attention_support`` (the
``onnxsim/webgpu_target.py`` advisory checker) and the
``gemm_fusion_backend="webgpu"`` alias on :func:`onnxsim.simplify`.

Models with a ``com.microsoft::Attention`` node are built directly via the
ONNX text format parser -- the same shape ``fuse_attention.h`` produces (see
``tests/test_dynamic_quantize_attention.py``) -- since these tests are about
inspecting an ``Attention`` node's inputs, not about fusing one.
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim


def _model(body, initializer=(), opset=17, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _attention_model(mask=False, past=False, B=2, S=5, H=32, NH=4, seed=0):
    # Same node shape fuse_attention.h produces: Attention(x, wqkv, bqkv, ...).
    # mask_index/past are left as empty ("") inputs unless requested, matching
    # how the ONNX text format spells an omitted optional input.
    rng = np.random.default_rng(seed)
    wqkv = _f32(rng.standard_normal((H, H * 3)) * 0.1, "wqkv")
    bqkv = _f32(rng.standard_normal(H * 3) * 0.1, "bqkv")
    inputs = ["x", "wqkv", "bqkv"]
    graph_inputs = f"float[{B},{S},{H}] x"
    initializers = [wqkv, bqkv]
    if mask or past:
        inputs.append("mask_index" if mask else "")
        if mask:
            graph_inputs += f", int32[{B}] mask_index"
    if past:
        inputs.append("past")
        graph_inputs += f", float[2,{B},{NH},0,{H // NH}] past"
    attrs = f"num_heads = {NH}, qkv_hidden_sizes = [{H}, {H}, {H}]"
    return _model(
        f"""
        g ({graph_inputs}) => (float[{B},{S},{H}] y)
        {{
          y = com.microsoft.Attention<{attrs}>({", ".join(inputs)})
        }}
        """,
        initializers,
        opset=17,
    )


def test_check_webgpu_attention_support_clean_model():
    # fuse_attention.h's own output shape: no mask, no past -- nothing to flag.
    model = _attention_model()
    assert onnxsim.check_webgpu_attention_support(model) == []


def test_check_webgpu_attention_support_flags_mask_index():
    model = _attention_model(mask=True)
    messages = onnxsim.check_webgpu_attention_support(model)
    assert len(messages) == 1
    assert "mask_index" in messages[0]
    assert "past" not in messages[0].split("wired up")[0]


def test_check_webgpu_attention_support_flags_past():
    model = _attention_model(past=True)
    messages = onnxsim.check_webgpu_attention_support(model)
    assert len(messages) == 1
    assert "past=" in messages[0]


def test_check_webgpu_attention_support_flags_both():
    model = _attention_model(mask=True, past=True)
    messages = onnxsim.check_webgpu_attention_support(model)
    assert len(messages) == 1
    assert "mask_index" in messages[0] and "past=" in messages[0]


def test_check_webgpu_attention_support_ignores_non_attention_nodes():
    # A plain MatMul, and an Attention-*named* node in a different domain,
    # must not be mistaken for com.microsoft::Attention.
    model = _model(
        """
        g (float[2,3] x, float[3,4] w) => (float[2,4] y)
        {
          y = MatMul(x, w)
        }
        """
    )
    assert onnxsim.check_webgpu_attention_support(model) == []


def test_check_webgpu_attention_support_accepts_file_path(tmp_path):
    model = _attention_model(mask=True)
    path = str(tmp_path / "model.onnx")
    onnx.save(model, path)
    messages = onnxsim.check_webgpu_attention_support(path)
    assert len(messages) == 1
    assert "mask_index" in messages[0]


def _conv3d_model(op_type="Conv", via_kernel_shape=True, spatial=3):
    # W has shape (out_channels, in_channels, *kernel_shape); rank - 2 gives
    # the spatial rank either way (kernel_shape attribute or W's own shape).
    out_c, in_c, k = 4, 3, 3
    w_shape = (out_c, in_c) + (k,) * spatial
    w = onnx.numpy_helper.from_array(
        np.random.default_rng(0).standard_normal(w_shape).astype(np.float32), "w"
    )
    x_shape = [1, in_c] + [8] * spatial
    y_shape = [1, out_c] + [6] * spatial
    op = f"{op_type}<kernel_shape = {list((k,) * spatial)}>" if via_kernel_shape else op_type
    return _model(
        f"""
        g (float{x_shape} x) => (float{y_shape} y)
        {{
          y = {op}(x, w)
        }}
        """,
        [w],
    )


def test_check_webgpu_conv3d_support_flags_conv3d_via_kernel_shape():
    model = _conv3d_model("Conv", via_kernel_shape=True)
    messages = onnxsim.check_webgpu_conv3d_support(model)
    assert len(messages) == 1
    assert "Conv" in messages[0] and "spatial rank 3" in messages[0]


def test_check_webgpu_conv3d_support_flags_conv3d_via_initializer_shape():
    # No kernel_shape attribute at all -- rank must come from W's own shape.
    model = _conv3d_model("Conv", via_kernel_shape=False)
    messages = onnxsim.check_webgpu_conv3d_support(model)
    assert len(messages) == 1


def test_check_webgpu_conv3d_support_flags_conv_transpose_3d():
    model = _conv3d_model("ConvTranspose", via_kernel_shape=True)
    messages = onnxsim.check_webgpu_conv3d_support(model)
    assert len(messages) == 1
    assert "ConvTranspose" in messages[0]


def test_check_webgpu_conv3d_support_ignores_conv2d():
    model = _conv3d_model("Conv", via_kernel_shape=True, spatial=2)
    assert onnxsim.check_webgpu_conv3d_support(model) == []


def _resize_model(mode=None, scales=None):
    op = f'Resize<coordinate_transformation_mode = "{mode}">' if mode else "Resize"
    scales_input = ""
    initializer = []
    if scales is not None:
        scales_input = "scales"
        initializer.append(_f32(scales, "scales"))
    return _model(
        f"""
        g (float[1,3,8,8] x) => (float[1,3,4,4] y)
        {{
          y = {op}(x, , {scales_input})
        }}
        """,
        initializer,
        opset=13,
    )


def test_check_webgpu_resize_support_flags_align_corners_downsample():
    model = _resize_model("align_corners", [1.0, 1.0, 0.5, 0.5])
    messages = onnxsim.check_webgpu_resize_support(model)
    assert len(messages) == 1
    assert "align_corners" in messages[0]


def test_check_webgpu_resize_support_ignores_align_corners_upsample():
    # align_corners is only documented as broken for downsampling.
    model = _resize_model("align_corners", [1.0, 1.0, 2.0, 2.0])
    assert onnxsim.check_webgpu_resize_support(model) == []


def test_check_webgpu_resize_support_ignores_default_mode_downsample():
    # half_pixel (the default) isn't the flagged mode, even downsampling.
    model = _resize_model(None, [1.0, 1.0, 0.5, 0.5])
    assert onnxsim.check_webgpu_resize_support(model) == []


def test_check_webgpu_resize_support_ignores_non_constant_scales():
    # scales computed at runtime rather than a constant initializer -- this
    # module doesn't run shape inference, so it can't tell up- from
    # downsampling here and deliberately leaves it unflagged.
    model = _model(
        """
        g (float[1,3,8,8] x, float[4] dyn_scales) => (float[1,3,4,4] y)
        {
          y = Resize<coordinate_transformation_mode = "align_corners">(x, , dyn_scales)
        }
        """,
        opset=13,
    )
    assert onnxsim.check_webgpu_resize_support(model) == []


def test_check_webgpu_support_aggregates_all_checks():
    attention_model = _attention_model(mask=True)
    conv3d_model = _conv3d_model("Conv", via_kernel_shape=True)
    resize_model = _resize_model("align_corners", [1.0, 1.0, 0.5, 0.5])

    assert len(onnxsim.check_webgpu_support(attention_model)) == 1
    assert len(onnxsim.check_webgpu_support(conv3d_model)) == 1
    assert len(onnxsim.check_webgpu_support(resize_model)) == 1
    assert onnxsim.check_webgpu_support(_model(
        """
        g (float[2,3] x, float[3,4] w) => (float[2,4] y)
        {
          y = MatMul(x, w)
        }
        """
    )) == []


def test_gemm_fusion_backend_webgpu_matches_unrestricted():
    # "webgpu" should fuse a FLOAT16 MatMul+Add into Gemm exactly like
    # "unrestricted" does -- unlike the "ort_cpu" default, which restricts the
    # fusion to FLOAT32 operands (see gemm_fusion_backend.h). This exercises
    # the actual C++ pass, so it needs the compiled extension.
    B, K, N = 2, 8, 8
    rng = np.random.default_rng(0)
    w = onnx.numpy_helper.from_array(
        (rng.standard_normal((K, N)) * 0.1).astype(np.float16), "w"
    )
    b = onnx.numpy_helper.from_array(
        (rng.standard_normal(N) * 0.1).astype(np.float16), "b"
    )
    model = _model(
        f"""
        g (float16[{B},{K}] x) => (float16[{B},{N}] y)
        {{
          mm = MatMul(x, w)
          y = Add(mm, b)
        }}
        """,
        [w, b],
        opset=17,
    )

    model_unrestricted, _ = onnxsim.simplify(model, gemm_fusion_backend="unrestricted")
    model_webgpu, _ = onnxsim.simplify(model, gemm_fusion_backend="webgpu")
    model_ort_cpu, _ = onnxsim.simplify(model, gemm_fusion_backend="ort_cpu")

    ops_unrestricted = {n.op_type for n in model_unrestricted.graph.node}
    ops_webgpu = {n.op_type for n in model_webgpu.graph.node}
    ops_ort_cpu = {n.op_type for n in model_ort_cpu.graph.node}

    assert "Gemm" in ops_unrestricted
    assert ops_webgpu == ops_unrestricted
    # The default restricts FLOAT16 fusion, so it should NOT match webgpu's
    # (fused) op set here -- otherwise this test isn't distinguishing anything.
    assert "Gemm" not in ops_ort_cpu
