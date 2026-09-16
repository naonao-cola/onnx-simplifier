"""Generates WebGPU compute programs for specific ONNX nodes by building the
equivalent computation as a real `tinygrad <https://github.com/tinygrad/tinygrad>`_
``Tensor`` graph, letting tinygrad's own scheduler/compiler lower it to its
UOp IR, and rendering that IR to WGSL with tinygrad's own ``WGSLRenderer`` --
then attaching the result via :mod:`onnxsim.webgpu_kernel_metadata`.

``tinygrad`` is an **optional** dependency: nothing in this module runs (or
is even imported) unless a caller actually calls one of the ``generate_*``
functions below, matching the ``onnxruntime``-is-optional precedent
elsewhere in onnxsim (see ``CLAUDE.md``). Install it with
``pip install tinygrad`` (pinned in ``tests/requirements-tinygrad.txt`` if
this repo has one, else whatever ``pip install tinygrad`` resolves to --
this was built and verified against tinygrad 0.14.0).

## How the lowering actually works, and why no real GPU is needed to do it

tinygrad's ``WEBGPU`` device tag only becomes a *hardware* dependency once
something tries to actually run a kernel through it -- allocating buffers,
compiling shader modules, submitting work -- which needs a real ``dawn``/
``wgpu-native`` shared library tinygrad's own ``ops_webgpu.py`` loads via
ctypes. This sandbox has no such library (``Device["WEBGPU"]`` fails with
"failed to load library webgpu"), and this module never asks it to: it only
uses ``WGSLRenderer`` -- a plain, dependency-free Python class -- directly,
bypassing ``Device["WEBGPU"]`` entirely. Verified concretely against
tinygrad 0.14.0's actual internals (its own test suite is not run here; this
is this module's own from-source verification, redone by hand against the
installed version, since tinygrad's UOp/codegen internals are not a stable
public API and have changed across versions before):

1. Build a ``Tensor`` graph with every leaf created as
   ``Tensor(numpy_array, device="WEBGPU")`` -- the ``device`` string here is
   just a tag threaded through the UOp graph, not a live connection to
   anything; it never triggers ``Device["WEBGPU"]``.
2. ``output.schedule_linear()`` walks the graph and returns one ``Ops.LINEAR``
   UOp for the whole computation (tinygrad 0.14's scheduler; older/newer
   tinygrad versions expose this differently -- e.g. a list of
   ``ScheduleItem`` -- so this is exactly the kind of internal detail that
   can and will drift).
3. Every ``Ops.CALL`` UOp in that graph whose first source is an ``Ops.SINK``
   (as opposed to ``Ops.COPY``, which is a host<->device buffer transfer, not
   a compute kernel) is one real compute kernel to render -- there can be
   more than one; see :func:`_lower_tensor_program`'s own docstring for why.
4. ``tinygrad.codegen.to_program(ast, WGSLRenderer(Target()))`` runs the
   linearize/render passes and returns an ``Ops.PROGRAM`` UOp whose ``.arg``
   is a ``ProgramInfo`` (kernel name, global/local size, buffer slots) and
   whose ``.src`` chain ends in an ``Ops.SOURCE`` UOp holding the actual WGSL
   text. The "compile" step after render (``do_compile``) is a no-op text
   encode for every tinygrad renderer by default (``Compiler.compile``) --
   WGSL has no separate offline compilation step; the browser's own
   ``GPUDevice.createShaderModule`` does that job -- so nothing here ever
   needs a real device.

Every kernel WGSLRenderer emits declares an ``INFINITY`` uniform at binding
0 whether or not the kernel body reads it, and every other declared global
buffer is ``var<storage, read_write>`` regardless of whether tinygrad
considers it a kernel input or output -- see
``scripts/convertmodel/webgpu_kernel_dispatcher.mjs``'s own notes on both,
and ``onnxsim/webgpu_kernel_metadata.py``'s docstring for the resulting
schema (``constant``/``intermediate``/``tensor`` bindings).

## What's covered, and how each is verified

- **``Conv``/``ConvTranspose3d``-shaped gaps** (:func:`generate_conv_kernel`):
  any spatial rank, via ``Tensor.conv2d`` -- despite the name, tinygrad's
  ``conv2d`` is not 2-D-specific; it convolves over every non-batch/channel
  axis of its input; a "3-D conv" is just ``conv2d`` given 5-D tensors. Only
  ``Conv`` (not ``ConvTranspose``) is implemented -- see that function's own
  docstring for why. Asymmetric ``pads`` are applied as an explicit
  ``Tensor.pad`` before a zero-padding ``conv2d`` call, since tinygrad's own
  ``padding`` argument (like PyTorch's) is symmetric per axis, unlike ONNX's.
  ``auto_pad`` other than the default ``"NOTSET"`` is not implemented.
- **``Resize`` align_corners downsampling** (:func:`generate_resize_kernel`):
  ``Tensor.interpolate(..., mode="linear", align_corners=True)``. Only 2-D
  spatial resize (4-D NCHW tensors) with a constant ``scales`` input is
  covered, matching ``onnxsim.webgpu_target.check_webgpu_resize_support``'s
  own scope. Further restricted to scales that divide each spatial dimension
  to an exact integer size: ONNX's own ``align_corners`` formula divides by
  the *exact* (possibly fractional) implied output length
  (``input_size * scale``), while tinygrad's ``Tensor.interpolate`` always
  divides by the integer output size -- the two only agree when
  ``input_size * scale`` is already exact (verified concretely: a 9-pixel
  dimension halved to 4 diverges by ~1.8 absolute between the two, not a
  rounding-level difference), so a non-exact ratio raises ``ValueError``
  rather than silently generating a numerically wrong kernel.
- Attention (``com.microsoft::Attention`` ``mask_index``) is **not**
  implemented here -- the op has several mutually incompatible
  ``mask_index`` shapes (see ``ContribOperators.md``), and getting the wrong
  one silently produces a working-but-wrong kernel rather than an obvious
  failure. Left for a follow-up that picks one shape and cross-checks it
  numerically as rigorously as the two functions below are.

Both implemented generators are checked two ways before being trusted:

1. **Numeric correctness**, offline, no GPU: the same ``Tensor`` graph is
   also run on tinygrad's own default (CPU) device and compared against
   ``onnx.reference.ReferenceEvaluator`` running the *actual* ONNX node --
   see ``tests/test_webgpu_tinygrad_codegen.py``. This checks the ONNX ->
   tinygrad translation (attribute handling, padding convention, axis
   order), independent of whether the WGSL rendering/dispatch is correct.
2. **Real GPU execution**: ``scripts/convertmodel/test/webgpu_tinygrad_codegen.test.mjs``
   reads the generated program back out of a real ``.onnx`` file via
   ``onnx_node_metadata.mjs`` and dispatches it on a real WebGPU device
   (Playwright/Chromium), checking the GPU output against the same
   ``onnx.reference.ReferenceEvaluator`` values baked into the fixture.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.webgpu_kernel_metadata import (
    WebgpuKernelBinding,
    WebgpuKernelSpec,
    WebgpuKernelStep,
    attach_webgpu_kernel,
)

if TYPE_CHECKING:
    from tinygrad import Tensor

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

__all__ = [
    "generate_conv_kernel",
    "generate_resize_kernel",
]


def _require_tinygrad():
    try:
        import tinygrad  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "onnxsim.webgpu_tinygrad_codegen needs the optional 'tinygrad' "
            "package: pip install tinygrad (built and verified against "
            "tinygrad==0.14.0; see this module's own docstring for why its "
            "internals are worth pinning against)."
        ) from e


def _lower_tensor_program(
    named_tensors: Dict[str, "Tensor"], output_name: str
) -> WebgpuKernelSpec:
    """Schedules ``named_tensors[output_name]``'s computation for tinygrad's
    ``WEBGPU`` device tag and renders every resulting compute kernel through
    tinygrad's own ``WGSLRenderer`` -- see this module's own docstring for
    the exact call chain and why no real WebGPU device is needed just to
    render.

    Every entry of ``named_tensors`` other than ``output_name`` must be a
    *leaf* tensor -- built directly as ``Tensor(numpy_array,
    device="WEBGPU")``, not derived from another Tensor -- so its buffer can
    be identified in the schedule by object identity and bound to that name.
    Any buffer tinygrad's own scheduler introduces that is neither a named
    leaf nor the named output (e.g. an un-fused softmax pass's scratch
    space) becomes a numbered ``intermediate`` in the returned spec.

    :raises RuntimeError: tinygrad scheduled no compute kernel at all (e.g.
            the whole graph constant-folded away).
    """
    from tinygrad.helpers import Target
    from tinygrad.renderer.wgsl import WGSLRenderer
    from tinygrad.uop.ops import Ops

    output = named_tensors[output_name]
    linear = output.schedule_linear()
    kernel_calls = [
        u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK
    ]
    if not kernel_calls:
        raise RuntimeError(
            "tinygrad scheduled no compute kernel for this graph -- it may "
            "have constant-folded away entirely"
        )

    # A Tensor's own .uop is a *view* over its actual buffer (e.g. a RESHAPE
    # wrapping the flat BUFFER a multi-dimensional leaf was created from --
    # BUFFER uops are always flat/1-D internally), not the buffer itself, so
    # comparing kernel argument buffers against named_tensors[x].uop directly
    # only ever matches by luck (a 1-D tensor's "reshape to itself" view
    # happens to collapse to the bare BUFFER). Walking each tensor's own
    # post-schedule UOp graph for the BUFFER it's actually backed by is what
    # reliably matches a kernel's buffer arguments regardless of rank.
    def _base_buffer(t):
        return next((u for u in t.uop.toposort() if u.op is Ops.BUFFER), None)

    uop_to_name = {}
    for name, t in named_tensors.items():
        buf = _base_buffer(t)
        if buf is None:
            raise RuntimeError(
                f"tensor {name!r} has no underlying BUFFER after scheduling"
            )
        uop_to_name[buf] = name
    renderer = WGSLRenderer(Target())

    steps: List[WebgpuKernelStep] = []
    intermediate_names: Dict[object, str] = {}
    intermediate_bytes: Dict[str, int] = {}
    next_intermediate = [0]

    def _name_for(buf_uop) -> Tuple[str, str]:
        """Returns ``(kind, name)`` -- ``kind`` is ``"tensor"`` for a named
        leaf/output, ``"intermediate"`` for anything else (assigning it a
        fresh name the first time it's seen, reused on every later
        appearance of the same buffer).
        """
        name = uop_to_name.get(buf_uop)
        if name is not None:
            return "tensor", name
        name = intermediate_names.get(buf_uop)
        if name is None:
            name = f"_intermediate_{next_intermediate[0]}"
            next_intermediate[0] += 1
            intermediate_names[buf_uop] = name
            intermediate_bytes[name] = buf_uop.size()[0] * buf_uop.dtype.itemsize
        return "intermediate", name

    for call in kernel_calls:
        ast = call.src[0]
        buffer_uops = list(call.src[1:])
        steps.append(_render_kernel_step(ast, buffer_uops, _name_for, renderer))

    return WebgpuKernelSpec(steps=tuple(steps), intermediates=intermediate_bytes)


def _render_kernel_step(ast, buffer_uops, name_for, renderer) -> WebgpuKernelStep:
    """Renders one already-scheduled kernel ``ast`` (an ``Ops.SINK``-rooted
    per-kernel AST -- one ``Ops.CALL``'s own ``src[0]``) to a single
    :class:`WebgpuKernelStep`, via tinygrad's own ``to_program``/
    ``WGSLRenderer``. Factored out of :func:`_lower_tensor_program`'s own
    loop body so :mod:`onnxsim.webgpu_kernel_tuning` can render several
    *alternative* (differently ``Opt``-tuned) ASTs for the very same call --
    same ``buffer_uops``/``name_for`` (a kernel's tuning options change its
    loop/tiling structure, never which buffers it reads/writes, so bindings
    are identical across every tuning candidate for one call; see that
    module's own docstring) -- without duplicating the bindings/dispatch
    bookkeeping a second time.

    :param buffer_uops: the owning ``Ops.CALL``'s own ``src[1:]`` -- fixed
            per call, independent of which optimized variant of ``ast`` is
            rendered (see above).
    :param name_for: ``_lower_tensor_program``'s own ``_name_for`` closure
            (or an equivalent) -- maps a buffer ``UOp`` to ``(kind, name)``.
    """
    from tinygrad.codegen import to_program
    from tinygrad.uop.ops import Ops

    prg = to_program(ast, renderer)
    info = prg.arg
    source_uop = next(s for s in prg.src if s.op is Ops.SOURCE)
    wgsl = source_uop.arg
    entry_point = _ANSI_RE.sub("", info.function_name)

    bindings: List[WebgpuKernelBinding] = [
        # tinygrad's WGSLRenderer always reserves binding 0 for this,
        # whether or not the kernel body reads it -- see this module's
        # docstring.
        WebgpuKernelBinding.for_constant([float("inf")], group=0, binding=0)
    ]
    for slot, buf_uop in enumerate(buffer_uops):
        kind, name = name_for(buf_uop)
        if kind == "tensor":
            bindings.append(
                WebgpuKernelBinding.for_tensor(
                    name, group=0, binding=slot + 1, access="read_write"
                )
            )
        else:
            bindings.append(
                WebgpuKernelBinding.for_intermediate(
                    name, group=0, binding=slot + 1, access="read_write"
                )
            )

    # ProgramInfo.global_size can be shorter than 3 dims (e.g. a 1-D
    # workgroup count) -- pad with 1s to match the schema's fixed-3-tuple
    # dispatch (see onnxsim/webgpu_kernel_metadata.py's own docstring for
    # why dispatch is a fixed triple, not a variable-length list).
    padded_size = [int(x) for x in info.global_size] + [1, 1, 1]
    dispatch: Tuple[int, int, int] = (padded_size[0], padded_size[1], padded_size[2])
    return WebgpuKernelStep(
        wgsl=wgsl, entry_point=entry_point, dispatch=dispatch, bindings=tuple(bindings)
    )


def _get_attr(node: onnx.NodeProto, name: str):
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return None


def _initializer_map(graph: onnx.GraphProto) -> Dict[str, onnx.TensorProto]:
    return {init.name: init for init in graph.initializer}


_shape_inference_cache: Dict[int, onnx.ModelProto] = {}


def _static_shape(model: onnx.ModelProto, tensor_name: str) -> Tuple[int, ...]:
    """The fully-static shape of ``tensor_name``, from an initializer if it
    is one, else from shape inference. Shape inference is run once per
    ``model`` object and cached (keyed by ``id(model)``, only the most
    recent) since a caller generating kernels for several nodes in the same
    model would otherwise pay for it repeatedly.

    :raises ValueError: the tensor isn't found, or has any non-static
            (symbolic or fully unknown) dimension.
    """
    init = _initializer_map(model.graph).get(tensor_name)
    if init is not None:
        return tuple(init.dims)

    key = id(model)
    if key not in _shape_inference_cache:
        _shape_inference_cache.clear()  # only ever cache the most recently inferred model
        _shape_inference_cache[key] = onnx.shape_inference.infer_shapes(model)
    inferred = _shape_inference_cache[key]

    for values in (
        inferred.graph.input,
        inferred.graph.output,
        inferred.graph.value_info,
    ):
        for vi in values:
            if vi.name != tensor_name:
                continue
            dims = vi.type.tensor_type.shape.dim
            shape = []
            for d in dims:
                if not d.HasField("dim_value"):
                    raise ValueError(
                        f"tensor {tensor_name!r} has a non-static dimension "
                        f"({d.dim_param!r} or unknown) -- shape must be fully "
                        "known to generate a kernel for it"
                    )
                shape.append(d.dim_value)
            return tuple(shape)
    raise ValueError(
        f"no shape found for tensor {tensor_name!r} (not an initializer, input, output, or value_info)"
    )


def _find_node(graph: onnx.GraphProto, node_name: str) -> onnx.NodeProto:
    for node in graph.node:
        if node.name == node_name:
            return node
    raise ValueError(f"no node named {node_name!r} in the graph")


def generate_conv_kernel(model: onnx.ModelProto, node_name: str) -> onnx.ModelProto:
    """Generates a WebGPU kernel for the ``Conv`` node named ``node_name``
    (any spatial rank, including the 3-D case
    ``onnxsim.webgpu_target.check_webgpu_conv3d_support`` flags) and attaches
    it via :func:`onnxsim.webgpu_kernel_metadata.attach_webgpu_kernel`.

    :param model: mutated in place (the generated kernel is attached
            directly) and also returned, for chaining.
    :param node_name: a ``Conv`` node's ``NodeProto.name``.
    :raises ImportError: the optional ``tinygrad`` dependency isn't
            installed.
    :raises ValueError: the node isn't a (default-domain) ``Conv``, its
            ``auto_pad`` isn't ``"NOTSET"``, or a needed tensor's shape isn't
            fully static (see :func:`_static_shape`).
    :returns: ``model``, mutated in place.
    """
    _require_tinygrad()
    from tinygrad import Tensor

    node = _find_node(model.graph, node_name)
    if node.domain not in ("", "ai.onnx") or node.op_type != "Conv":
        raise ValueError(
            f"node {node_name!r} is {node.domain!r}::{node.op_type!r}, not a default-domain Conv"
        )

    auto_pad = _get_attr(node, "auto_pad")
    if auto_pad is not None and auto_pad not in (b"NOTSET", "NOTSET"):
        raise ValueError(
            f"Conv auto_pad={auto_pad!r} is not implemented -- only the default NOTSET is"
        )

    x_name, w_name = node.input[0], node.input[1]
    b_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
    x_shape = _static_shape(model, x_name)
    w_shape = _static_shape(model, w_name)
    spatial_rank = len(w_shape) - 2

    kernel_shape = _get_attr(node, "kernel_shape")
    if kernel_shape is not None and tuple(kernel_shape) != w_shape[2:]:
        raise ValueError(
            f"kernel_shape attribute {tuple(kernel_shape)} disagrees with W's own shape {w_shape[2:]}"
        )

    strides = list(_get_attr(node, "strides") or [1] * spatial_rank)
    dilations = list(_get_attr(node, "dilations") or [1] * spatial_rank)
    group = _get_attr(node, "group") or 1
    pads = list(_get_attr(node, "pads") or [0] * (2 * spatial_rank))
    pads_begin, pads_end = pads[:spatial_rank], pads[spatial_rank:]

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal(x_shape).astype(np.float32), device="WEBGPU")
    w = Tensor(rng.standard_normal(w_shape).astype(np.float32), device="WEBGPU")
    named = {"__x": x, "__w": w}
    if pads_begin != [0] * spatial_rank or pads_end != [0] * spatial_rank:
        # Tensor.pad's padding is (dim0_before, dim0_after, dim1_before, ...)
        # innermost-first, matching Tensor.pad's own documented convention;
        # only the spatial (trailing) dims get padded, batch/channel don't.
        pad_pairs = [None, None] + [(b, e) for b, e in zip(pads_begin, pads_end)]
        x = x.pad(pad_pairs)
    if b_name:
        b_shape = _static_shape(model, b_name)
        b = Tensor(rng.standard_normal(b_shape).astype(np.float32), device="WEBGPU")
        named["__b"] = b
    else:
        b = None
    y = x.conv2d(w, bias=b, groups=group, stride=strides, dilation=dilations, padding=0)
    named["__y"] = y

    spec = _lower_tensor_program(named, "__y")
    # Rename the internal placeholders to this node's real tensor names --
    # _lower_tensor_program only ever sees leaf/output identity, not what
    # they should be called in the attached metadata.
    rename = {"__x": x_name, "__w": w_name, "__y": node.output[0]}
    if b_name:
        rename["__b"] = b_name
    spec = _rename_tensor_bindings(spec, rename)

    return attach_webgpu_kernel(model, node_name, spec)


def _rename_tensor_bindings(
    spec: WebgpuKernelSpec, rename: Dict[str, str]
) -> WebgpuKernelSpec:
    new_steps = []
    for step in spec.steps:
        new_bindings = tuple(
            b
            if b.tensor is None or b.tensor not in rename
            else WebgpuKernelBinding.for_tensor(
                rename[b.tensor], b.group, b.binding, b.access
            )
            for b in step.bindings
        )
        new_steps.append(
            WebgpuKernelStep(step.wgsl, step.entry_point, step.dispatch, new_bindings)
        )
    return WebgpuKernelSpec(
        steps=tuple(new_steps), intermediates=dict(spec.intermediates)
    )


def generate_resize_kernel(model: onnx.ModelProto, node_name: str) -> onnx.ModelProto:
    """Generates a WebGPU program for the 2-D ``Resize`` node named
    ``node_name`` -- the ``align_corners`` downsampling case
    ``onnxsim.webgpu_target.check_webgpu_resize_support`` flags -- and
    attaches it via :func:`onnxsim.webgpu_kernel_metadata.attach_webgpu_kernel`.

    :param model: mutated in place (the generated program is attached
            directly) and also returned, for chaining.
    :param node_name: a ``Resize`` node's ``NodeProto.name``. Must have a
            constant ``scales`` input (a graph initializer) and a 4-D
            (NCHW) input -- matching
            ``check_webgpu_resize_support``'s own scope; other ranks and a
            ``sizes``-based call aren't implemented.
    :raises ImportError: the optional ``tinygrad`` dependency isn't
            installed.
    :raises ValueError: the node isn't a (default-domain) ``Resize``, its
            input isn't 4-D, its ``mode`` isn't ``"linear"``, or its
            ``scales`` input isn't a constant initializer.
    :returns: ``model``, mutated in place.
    """
    _require_tinygrad()
    from tinygrad import Tensor

    node = _find_node(model.graph, node_name)
    if node.domain not in ("", "ai.onnx") or node.op_type != "Resize":
        raise ValueError(
            f"node {node_name!r} is {node.domain!r}::{node.op_type!r}, not a default-domain Resize"
        )

    mode = _get_attr(node, "mode") or b"linear"
    if mode not in (b"linear", "linear"):
        raise ValueError(f'Resize mode={mode!r} is not implemented -- only "linear" is')

    x_name = node.input[0]
    x_shape = _static_shape(model, x_name)
    if len(x_shape) != 4:
        raise ValueError(
            f"Resize input {x_name!r} has rank {len(x_shape)}, only 4-D (NCHW) is implemented"
        )

    scales_name = node.input[2] if len(node.input) > 2 else ""
    scales_init = _initializer_map(model.graph).get(scales_name)
    if scales_init is None:
        raise ValueError(
            f"Resize scales input {scales_name!r} is not a constant initializer"
        )
    scales = onnx.numpy_helper.to_array(scales_init)

    # ONNX's own align_corners formula divides by (length_resized - 1), where
    # length_resized is the *exact* (possibly fractional) input_size*scale --
    # confirmed against onnx.reference.ReferenceEvaluator, not assumed from
    # the spec text alone -- while tinygrad's Tensor.interpolate(align_corners=True)
    # always divides by (its integer `size` argument - 1). The two formulas
    # only coincide when input_size*scale is already an exact integer (e.g.
    # exact 2x/4x downsampling); for other ratios (a 9-pixel dimension halved
    # to 4, say) they diverge -- verified concretely: for a 9->4 resize this
    # produces values as far off as ~1.8 in absolute terms, not a rounding
    # difference. So this only generates a kernel when every resized
    # dimension's scale divides it exactly.
    exact_sizes = [d * s for d, s in zip(x_shape[2:], scales[2:])]
    if any(abs(s - round(s)) > 1e-6 for s in exact_sizes):
        raise ValueError(
            f"Resize scales {list(scales)} do not divide input shape {x_shape} exactly "
            f"(would resize to {exact_sizes}) -- align_corners' coordinate formula only "
            "agrees between ONNX and tinygrad's interpolate for exact integer ratios, "
            "see this function's own comment"
        )
    out_shape = tuple(int(round(d * s)) for d, s in zip(x_shape, scales))

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal(x_shape).astype(np.float32), device="WEBGPU")
    y = x.interpolate(out_shape[2:], mode="linear", align_corners=True)

    spec = _lower_tensor_program({"__x": x, "__y": y}, "__y")
    spec = _rename_tensor_bindings(spec, {"__x": x_name, "__y": node.output[0]})
    return attach_webgpu_kernel(model, node_name, spec)
