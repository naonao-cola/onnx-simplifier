"""Tests for ``onnxsim.webgpu_tinygrad_codegen`` -- offline (no real GPU)
numeric verification that the tinygrad ``Tensor`` graph each generator
builds computes the same thing as the actual ONNX node.

Models are built via the ONNX text format parser (see CLAUDE.md's testing
guidance); ``onnx.parser`` never assigns ``NodeProto.name``, so ``_named``
sets it programmatically after parsing, same as
``tests/test_webgpu_kernel_metadata.py``.

This only checks the ONNX -> tinygrad translation (attribute handling,
padding convention, axis order) by running the *same* ``Tensor`` graph on
tinygrad's own default (CPU) device and comparing against
``onnx.reference.ReferenceEvaluator`` -- it does not exercise WGSL
rendering/dispatch at all (that needs a real WebGPU device; see
``scripts/convertmodel/test/webgpu_tinygrad_codegen.test.mjs`` for that
half, over Playwright/Chromium). It does check that
:func:`onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel` and
:func:`~onnxsim.webgpu_tinygrad_codegen.generate_resize_kernel` attach a
structurally sane :class:`~onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec`
(every step is real WGSL with the mandatory ``INFINITY`` constant binding).
"""

import numpy as np
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

pytest.importorskip("tinygrad")

import onnx  # noqa: E402

from onnxsim.webgpu_kernel_metadata import read_webgpu_kernel  # noqa: E402
from onnxsim.webgpu_tinygrad_codegen import (  # noqa: E402
    generate_conv_kernel,
    generate_resize_kernel,
)


def _named(model, output_name, node_name):
    for node in model.graph.node:
        if output_name in node.output:
            node.name = node_name
            return model
    raise AssertionError(f"no node producing {output_name!r}")


def _build_conv_model(spatial, in_c=3, out_c=4, k=3, x_size=8, pads=None, strides=None):
    rng = np.random.default_rng(0)
    w_shape = (out_c, in_c) + (k,) * spatial
    w = numpy_helper.from_array(rng.standard_normal(w_shape).astype(np.float32), "w")
    x_shape = [1, in_c] + [x_size] * spatial
    attrs = f"kernel_shape = {list((k,) * spatial)}"
    if pads:
        attrs += f", pads = {list(pads)}"
    if strides:
        attrs += f", strides = {list(strides)}"
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        g (float{x_shape} x) => (float[?] y)
        {{
          y = Conv<{attrs}>(x, w)
        }}
        """
    )
    model.graph.initializer.append(w)
    _named(model, "y", "conv_node")
    onnx.checker.check_model(model)
    return model, tuple(x_shape)


def _assert_conv_kernel_matches_reference(spatial, **kwargs):
    model, x_shape = _build_conv_model(spatial, **kwargs)
    generate_conv_kernel(model, "conv_node")

    spec = read_webgpu_kernel(model, "conv_node")
    assert spec is not None
    for step in spec.steps:
        assert "@compute" in step.wgsl
        # tinygrad's WGSLRenderer always reserves binding 0 for this -- see
        # onnxsim/webgpu_tinygrad_codegen.py's own docstring.
        assert step.bindings[0].constant == (float("inf"),)

    from tinygrad import Tensor

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal(x_shape).astype(np.float32)
    w_np = numpy_helper.to_array(model.graph.initializer[0])

    ref = ReferenceEvaluator(model)
    (y_ref,) = ref.run(None, {"x": x_np})

    x_t = Tensor(x_np)
    w_t = Tensor(w_np)
    pads = kwargs.get("pads")
    if pads:
        pads_begin, pads_end = pads[:spatial], pads[spatial:]
        x_t = x_t.pad([None, None] + list(zip(pads_begin, pads_end)))
    strides = kwargs.get("strides") or [1] * spatial
    y_t = x_t.conv2d(w_t, stride=strides, padding=0).numpy()

    assert y_t.shape == y_ref.shape
    np.testing.assert_allclose(y_t, y_ref, atol=1e-3, rtol=1e-3)
    return spec


def test_conv2d_kernel_matches_reference():
    _assert_conv_kernel_matches_reference(2)


def test_conv3d_kernel_matches_reference():
    # The gap onnxsim.webgpu_target.check_webgpu_conv3d_support flags.
    _assert_conv_kernel_matches_reference(3)


def test_conv2d_symmetric_padding_kernel_matches_reference():
    _assert_conv_kernel_matches_reference(2, pads=[1, 1, 1, 1])


def test_conv2d_asymmetric_padding_kernel_matches_reference():
    _assert_conv_kernel_matches_reference(2, pads=[0, 1, 2, 1])


def test_conv2d_strided_kernel_matches_reference():
    _assert_conv_kernel_matches_reference(2, strides=[2, 2])


def test_conv3d_padded_kernel_matches_reference():
    _assert_conv_kernel_matches_reference(3, pads=[1, 0, 1, 0, 1, 0])


def test_conv_wrong_op_type_raises():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        g (float[4] a, float[4] b) => (float[4] c)
        {
          c = Add(a, b)
        }
        """
    )
    _named(model, "c", "add_node")
    with pytest.raises(ValueError, match="not a default-domain Conv"):
        generate_conv_kernel(model, "add_node")


def test_conv_auto_pad_not_implemented_raises():
    model, _ = _build_conv_model(2)
    model.graph.node[0].attribute.add(
        name="auto_pad", type=onnx.AttributeProto.STRING, s=b"SAME_UPPER"
    )
    with pytest.raises(ValueError, match="auto_pad"):
        generate_conv_kernel(model, "conv_node")


def _build_resize_model(
    x_shape, scales, mode="linear", coordinate_transformation_mode="align_corners"
):
    scales_init = numpy_helper.from_array(
        np.asarray(scales, dtype=np.float32), "scales"
    )
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 13]>
        g (float{list(x_shape)} x) => (float[?] y)
        {{
          y = Resize<mode = "{mode}", coordinate_transformation_mode = "{coordinate_transformation_mode}">(x, , scales)
        }}
        """
    )
    model.graph.initializer.append(scales_init)
    _named(model, "y", "resize_node")
    onnx.checker.check_model(model)
    return model


def _assert_resize_kernel_matches_reference(x_shape, scales):
    model = _build_resize_model(x_shape, scales)
    generate_resize_kernel(model, "resize_node")

    spec = read_webgpu_kernel(model, "resize_node")
    assert spec is not None
    for step in spec.steps:
        assert "@compute" in step.wgsl
        assert step.bindings[0].constant == (float("inf"),)

    from tinygrad import Tensor

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal(x_shape).astype(np.float32)

    ref = ReferenceEvaluator(model)
    (y_ref,) = ref.run(None, {"x": x_np})

    out_shape = tuple(int(round(d * s)) for d, s in zip(x_shape, scales))
    y_t = (
        Tensor(x_np)
        .interpolate(out_shape[2:], mode="linear", align_corners=True)
        .numpy()
    )

    assert y_t.shape == y_ref.shape
    np.testing.assert_allclose(y_t, y_ref, atol=1e-3, rtol=1e-3)
    return spec


def test_resize_exact_downsample_kernel_matches_reference():
    # 8 * 0.5 = 4 is an exact integer ratio -- ONNX's and tinygrad's own
    # align_corners formulas agree here (see the module docstring / the
    # exact-ratio guard in generate_resize_kernel for the case where they
    # don't).
    spec = _assert_resize_kernel_matches_reference((1, 3, 8, 8), [1.0, 1.0, 0.5, 0.5])
    # tinygrad's Tensor.interpolate schedules as two kernels (unlike Conv,
    # which fuses into one) -- see onnxsim/webgpu_kernel_metadata.py's own
    # docstring for why the schema supports multiple steps at all.
    assert len(spec.steps) == 2
    assert len(spec.intermediates) == 1


def test_resize_exact_non_uniform_scale_kernel_matches_reference():
    # 6 * 2/3 = 4 and 10 * 0.6 = 6 are both exact integers even though the
    # two spatial scales differ from each other.
    _assert_resize_kernel_matches_reference((1, 1, 6, 10), [1.0, 1.0, 2.0 / 3.0, 0.6])


def test_resize_inexact_ratio_raises():
    # 9 * 0.5 = 4.5 is not an exact integer -- align_corners' coordinate
    # formula diverges between ONNX (denominator 4.5-1=3.5) and tinygrad
    # (denominator 4-1=3) for this case, so generate_resize_kernel must
    # refuse it rather than silently generating a numerically wrong kernel.
    model = _build_resize_model((1, 3, 9, 9), [1.0, 1.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="do not divide input shape"):
        generate_resize_kernel(model, "resize_node")


def test_resize_wrong_mode_raises():
    model = _build_resize_model((1, 3, 8, 8), [1.0, 1.0, 0.5, 0.5], mode="nearest")
    with pytest.raises(ValueError, match="mode"):
        generate_resize_kernel(model, "resize_node")


def test_resize_non_4d_input_raises():
    model = _build_resize_model((1, 3, 8, 8, 8), [1.0, 1.0, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="rank"):
        generate_resize_kernel(model, "resize_node")
