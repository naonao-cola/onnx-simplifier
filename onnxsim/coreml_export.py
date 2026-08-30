"""Convert a (simplified) ONNX model to Core ML.

onnxsim's job stops at a cleaned-up ``onnx.ModelProto``. Apple platforms want that
graph as a Core ML model (``.mlpackage``) instead. This module bridges the two by
walking the ONNX graph node by node and building the equivalent `MIL
<https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html>`_
(Model Intermediate Language) program with `coremltools
<https://github.com/apple/coremltools>`_'s own builder, then handing that program to
``coremltools.convert(..., source="milinternal")`` to produce the actual Core ML model.

coremltools itself dropped its ONNX frontend in version 7 (it only converts
TensorFlow/PyTorch models, or an in-memory MIL program), so there is no off-the-shelf
"convert this ONNX model" call to lean on -- this translator plays that role, one ONNX
op at a time. It covers a practical subset of ops (common to CNN/MLP/transformer-block
graphs: conv, pooling, normalization, matmul/gemm, elementwise math, reshapes,
reductions, ...); a node whose op isn't in ``SUPPORTED_ONNX_OPS`` raises a
``RuntimeError`` naming the op, rather than silently producing a wrong model.

Feeding a *simplified* model in is the point, same as with ``mlir_export.py``: onnxsim's
constant folding turns shape-manipulation subgraphs and foldable weight arithmetic into
plain initializers, so more of the graph lands on this translator's supported-op list
and the emitted Core ML model is smaller.

Like onnxruntime for constant folding and torch-mlir/onnx-mlir for MLIR (see
``backend.py`` / ``mlir_export.py``), coremltools is an **optional** dependency: nothing
here is imported at ``import onnxsim`` time, only when ``--emit-coreml`` / the
``export_coreml`` API run. A missing coremltools raises a ``RuntimeError`` with an
install hint.

Conversion itself needs no macOS-specific functionality (MIL construction and
``.mlpackage`` serialization are pure Python/protobuf), so it runs the same on Linux,
macOS, or Windows. Only *loading the produced model back for a prediction* needs Core
ML's runtime, i.e. an Apple OS -- this module defaults to ``skip_model_load=True`` so
conversion still succeeds off of macOS; pass ``skip_model_load=False`` on macOS to get a
model that's ready to run.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import onnx
from onnx import numpy_helper

_COREML_INSTALL_HINT = (
    "coremltools is required to export Core ML models but is not installed. "
    "Install it with `pip install coremltools`."
)

# ONNX TensorProto dtypes this translator can carry through to Core ML, downcasting
# where Core ML/MIL has no matching type (float64 -> float32, int64 -> int32; Core ML
# arrays have no 64-bit numeric type).
_NP_DOWNCAST = {
    np.dtype(np.float64): np.float32,
    np.dtype(np.int64): np.int32,
}
_SUPPORTED_NP_DTYPES = {np.float32, np.float16, np.int32, np.bool_}


def has_coremltools() -> bool:
    """Whether coremltools is importable in this environment."""
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def _import_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(_COREML_INSTALL_HINT) from exc
    return ct


def _import_mil():
    try:
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import Function, Program, types
    except ImportError as exc:
        raise RuntimeError(_COREML_INSTALL_HINT) from exc
    return mb, types, Function, Program


def _as_mil_array(arr: np.ndarray) -> np.ndarray:
    """Downcast ``arr`` to a dtype Core ML/MIL supports, or raise."""
    arr = np.asarray(arr)
    target = _NP_DOWNCAST.get(arr.dtype)
    if target is not None:
        arr = arr.astype(target)
    if arr.dtype.type not in _SUPPORTED_NP_DTYPES:
        raise RuntimeError(
            f"Unsupported tensor dtype {arr.dtype} for Core ML export (supported: "
            "float16, float32, float64, int32, int64, bool; 64-bit types are "
            "downcast to their 32-bit equivalent)."
        )
    return arr


def _onnx_elem_type_to_mil(elem_type: int, types) -> Any:
    TP = onnx.TensorProto
    mapping = {
        TP.FLOAT: types.fp32,
        TP.FLOAT16: types.fp16,
        TP.DOUBLE: types.fp32,
        TP.INT32: types.int32,
        TP.INT64: types.int32,
        TP.BOOL: types.bool,
    }
    if elem_type not in mapping:
        raise RuntimeError(
            f"Unsupported input dtype {TP.DataType.Name(elem_type)}; onnxsim's Core "
            "ML exporter supports float16, float32, float64, int32, int64, and bool "
            "graph inputs (64-bit types are represented as their 32-bit equivalent "
            "in the exported model)."
        )
    return mapping[elem_type]


def _make_input_spec(value_info: onnx.ValueInfoProto, mb, types):
    tt = value_info.type.tensor_type
    shape = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            shape.append(int(d.dim_value))
        else:
            raise RuntimeError(
                f"Input '{value_info.name}' has a non-static dimension "
                f"({d.dim_param or '?'!r}); onnxsim's Core ML exporter requires "
                "fully static input shapes. Give the model concrete input shapes "
                "first (e.g. via --overwrite-input-shape or "
                "onnx.tools.update_model_dims) before exporting."
            )
    dtype = _onnx_elem_type_to_mil(tt.elem_type, types)
    return mb.TensorSpec(shape=tuple(shape), dtype=dtype)


def _node_attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, onnx.TensorProto):
            v = numpy_helper.to_array(v)
        elif isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], bytes):
            v = [x.decode("utf-8") for x in v]
        attrs[a.name] = v
    return attrs


def _pad_type_and_pads(attrs: Dict[str, Any], n: int):
    """Map ONNX ``auto_pad``/``pads`` to a MIL ``(pad_type, pad)`` pair.

    ``pad`` is ``None`` when ``pad_type`` alone determines the padding (``"valid"``,
    ``"same"``, ``"same_lower"``); otherwise it is MIL's interleaved
    ``[before_0, after_0, before_1, after_1, ...]`` form, converted from ONNX's
    ``[before_0, ..., before_{n-1}, after_0, ..., after_{n-1}]`` form.
    """
    auto_pad = attrs.get("auto_pad", "NOTSET")
    if auto_pad == "SAME_UPPER":
        return "same", None
    if auto_pad == "SAME_LOWER":
        return "same_lower", None
    if auto_pad == "VALID":
        return "valid", None
    pads = attrs.get("pads")
    if not pads or not any(pads):
        return "valid", None
    begins, ends = pads[:n], pads[n:]
    interleaved: List[int] = []
    for b, e in zip(begins, ends):
        interleaved += [int(b), int(e)]
    return "custom", interleaved


def _const_scalar(var) -> float:
    if var.val is None:
        raise RuntimeError(
            "expected a compile-time constant here, but the tensor is only known at "
            "runtime (onnxsim's Core ML exporter can't trace dynamic values)"
        )
    return float(np.asarray(var.val).reshape(-1)[0])


class _Lowerer:
    """Walks an ONNX graph once, building the equivalent MIL ops as it goes."""

    def __init__(self, mb, types, opset: int):
        self.mb = mb
        self.types = types
        self.opset = opset
        self._values: Dict[str, Any] = {}
        self._counter = 0

    def bind(self, name: str, var) -> None:
        self._values[name] = var

    def get(self, name: str):
        if name not in self._values:
            raise RuntimeError(
                f"reference to unknown tensor '{name}' (the graph may not be "
                "topologically sorted, or the producing node uses an unsupported "
                "feature)"
            )
        return self._values[name]

    def make_const(self, name: str, arr: np.ndarray):
        self._counter += 1
        return self.mb.const(
            val=_as_mil_array(arr), name=f"{name}__const{self._counter}"
        )

    def fresh_name(self, node: onnx.NodeProto, suffix: Optional[str] = None) -> str:
        self._counter += 1
        base = (
            node.output[0]
            if node.output and node.output[0]
            else (node.name or node.op_type)
        )
        if suffix:
            base = f"{base}_{suffix}"
        return f"{base}__{self._counter}"

    def lower_node(self, node: onnx.NodeProto) -> None:
        handler = _OP_HANDLERS.get(node.op_type)
        if handler is None:
            raise RuntimeError(
                f"ONNX op '{node.op_type}' is not supported by onnxsim's Core ML "
                f"exporter (node {node.name or node.output[0]!r}). Supported ops: "
                + ", ".join(sorted(_OP_HANDLERS))
            )
        ins = [self.get(name) if name else None for name in node.input]
        attrs = _node_attrs(node)
        try:
            outs = handler(self, node, ins, attrs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to convert ONNX node {node.name or node.output[0]!r} "
                f"({node.op_type}) to Core ML: {exc}"
            ) from exc
        for out_name, var in zip(node.output, outs):
            if out_name:
                self._values[out_name] = var


# ---------------------------------------------------------------------------
# Op handlers. Each handler is ``(lowerer, node, ins, attrs) -> list[Var]``, with
# ``ins[i]`` the already-converted MIL value for ``node.input[i]`` (``None`` for an
# omitted optional input) and ``attrs`` the node's parsed attribute dict.
# ---------------------------------------------------------------------------

_OP_HANDLERS: Dict[str, Callable] = {}


def _register(*op_types: str):
    def deco(fn):
        for op_type in op_types:
            _OP_HANDLERS[op_type] = fn
        return fn

    return deco


def _simple_unary(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        fn = getattr(lowerer.mb, mil_name)
        return [fn(x=ins[0], name=lowerer.fresh_name(node))]

    return handler


def _simple_binary(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        fn = getattr(lowerer.mb, mil_name)
        return [fn(x=ins[0], y=ins[1], name=lowerer.fresh_name(node))]

    return handler


for _onnx_op, _mil_op in [
    ("Relu", "relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("Exp", "exp"),
    ("Log", "log"),
    ("Sqrt", "sqrt"),
    ("Erf", "erf"),
    ("Identity", "identity"),
]:
    _OP_HANDLERS[_onnx_op] = _simple_unary(_mil_op)

for _onnx_op, _mil_op in [
    ("Add", "add"),
    ("Sub", "sub"),
    ("Mul", "mul"),
    ("Div", "real_div"),
    ("Pow", "pow"),
]:
    _OP_HANDLERS[_onnx_op] = _simple_binary(_mil_op)


@_register("Neg")
def _op_neg(lowerer, node, ins, attrs):
    return [lowerer.mb.mul(x=ins[0], y=np.float32(-1.0), name=lowerer.fresh_name(node))]


@_register("LeakyRelu")
def _op_leaky_relu(lowerer, node, ins, attrs):
    alpha = float(attrs.get("alpha", 0.01))
    return [lowerer.mb.leaky_relu(x=ins[0], alpha=alpha, name=lowerer.fresh_name(node))]


@_register("Gelu")
def _op_gelu(lowerer, node, ins, attrs):
    approximate = attrs.get("approximate", "none")
    mode = "TANH_APPROXIMATION" if approximate == "tanh" else "EXACT"
    return [lowerer.mb.gelu(x=ins[0], mode=mode, name=lowerer.fresh_name(node))]


@_register("Clip")
def _op_clip(lowerer, node, ins, attrs):
    lo = attrs.get("min")
    hi = attrs.get("max")
    if len(ins) > 1 and ins[1] is not None:
        lo = _const_scalar(ins[1])
    if len(ins) > 2 and ins[2] is not None:
        hi = _const_scalar(ins[2])
    lo = float(lo) if lo is not None else float(np.finfo(np.float32).min)
    hi = float(hi) if hi is not None else float(np.finfo(np.float32).max)
    return [lowerer.mb.clip(x=ins[0], alpha=lo, beta=hi, name=lowerer.fresh_name(node))]


@_register("Softmax")
def _op_softmax(lowerer, node, ins, attrs):
    x = ins[0]
    default_axis = -1 if lowerer.opset >= 13 else 1
    axis = attrs.get("axis", default_axis)
    axis = axis if axis >= 0 else axis + x.rank
    if lowerer.opset >= 13:
        return [lowerer.mb.softmax(x=x, axis=axis, name=lowerer.fresh_name(node))]
    # Pre-opset-13 Softmax coerces x into 2-D at `axis` (flattening the leading and
    # trailing dimensions together) and applies softmax over the trailing axis, unlike
    # MIL's softmax which normalizes a single axis in place -- reproduce that by
    # reshaping down to 2-D, softmaxing, then reshaping back.
    shape = x.shape
    if any(not isinstance(d, (int, np.integer)) for d in shape):
        raise RuntimeError(
            "Softmax below opset 13 requires a fully static input shape to reproduce "
            "its flatten-then-normalize semantics"
        )
    lead = int(np.prod(shape[:axis])) if axis > 0 else 1
    trail = int(np.prod(shape[axis:]))
    flat = lowerer.mb.reshape(
        x=x, shape=[lead, trail], name=lowerer.fresh_name(node, "flat")
    )
    sm = lowerer.mb.softmax(x=flat, axis=-1, name=lowerer.fresh_name(node, "softmax"))
    return [lowerer.mb.reshape(x=sm, shape=list(shape), name=lowerer.fresh_name(node))]


@_register("MatMul")
def _op_matmul(lowerer, node, ins, attrs):
    return [lowerer.mb.matmul(x=ins[0], y=ins[1], name=lowerer.fresh_name(node))]


@_register("Gemm")
def _op_gemm(lowerer, node, ins, attrs):
    a, b = ins[0], ins[1]
    c = ins[2] if len(ins) > 2 else None
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = bool(attrs.get("transA", 0))
    trans_b = bool(attrs.get("transB", 0))
    y = lowerer.mb.matmul(
        x=a,
        y=b,
        transpose_x=trans_a,
        transpose_y=trans_b,
        name=lowerer.fresh_name(node, "matmul"),
    )
    if alpha != 1.0:
        y = lowerer.mb.mul(
            x=y, y=np.float32(alpha), name=lowerer.fresh_name(node, "alpha")
        )
    if c is not None:
        term = (
            c
            if beta == 1.0
            else lowerer.mb.mul(
                x=c, y=np.float32(beta), name=lowerer.fresh_name(node, "beta")
            )
        )
        y = lowerer.mb.add(x=y, y=term, name=lowerer.fresh_name(node))
    return [y]


@_register("Conv")
def _op_conv(lowerer, node, ins, attrs):
    x, w = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    n = x.rank - 2
    pad_type, pad = _pad_type_and_pads(attrs, n)
    kwargs = dict(
        x=x,
        weight=w,
        strides=list(attrs.get("strides", [1] * n)),
        pad_type=pad_type,
        dilations=list(attrs.get("dilations", [1] * n)),
        groups=int(attrs.get("group", 1)),
        name=lowerer.fresh_name(node),
    )
    if pad is not None:
        kwargs["pad"] = pad
    if bias is not None:
        kwargs["bias"] = bias
    return [lowerer.mb.conv(**kwargs)]


@_register("ConvTranspose")
def _op_conv_transpose(lowerer, node, ins, attrs):
    x, w = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    output_padding = attrs.get("output_padding")
    if output_padding and any(output_padding):
        raise RuntimeError(
            "ConvTranspose with a non-zero 'output_padding' is not supported (the "
            "output shape it implies isn't derivable from strides/pads alone)"
        )
    n = x.rank - 2
    pad_type, pad = _pad_type_and_pads(attrs, n)
    kwargs = dict(
        x=x,
        weight=w,
        strides=list(attrs.get("strides", [1] * n)),
        pad_type=pad_type,
        dilations=list(attrs.get("dilations", [1] * n)),
        groups=int(attrs.get("group", 1)),
        name=lowerer.fresh_name(node),
    )
    if pad is not None:
        kwargs["pad"] = pad
    if bias is not None:
        kwargs["bias"] = bias
    return [lowerer.mb.conv_transpose(**kwargs)]


@_register("BatchNormalization")
def _op_batch_norm(lowerer, node, ins, attrs):
    if len(node.output) > 1:
        raise RuntimeError(
            "training-mode BatchNormalization (with running-stats outputs) is not "
            "supported; only inference-mode (single-output) BatchNormalization is"
        )
    x, scale, bias, mean, var = ins[:5]
    eps = float(attrs.get("epsilon", 1e-5))
    return [
        lowerer.mb.batch_norm(
            x=x,
            mean=mean,
            variance=var,
            gamma=scale,
            beta=bias,
            epsilon=eps,
            name=lowerer.fresh_name(node),
        )
    ]


@_register("InstanceNormalization")
def _op_instance_norm(lowerer, node, ins, attrs):
    x, scale, bias = ins[:3]
    eps = float(attrs.get("epsilon", 1e-5))
    return [
        lowerer.mb.instance_norm(
            x=x, gamma=scale, beta=bias, epsilon=eps, name=lowerer.fresh_name(node)
        )
    ]


@_register("LayerNormalization")
def _op_layer_norm(lowerer, node, ins, attrs):
    x = ins[0]
    scale = ins[1] if len(ins) > 1 and ins[1] is not None else None
    bias = ins[2] if len(ins) > 2 and ins[2] is not None else None
    axis = int(attrs.get("axis", -1))
    axis = axis if axis >= 0 else axis + x.rank
    eps = float(attrs.get("epsilon", 1e-5))
    kwargs = dict(
        x=x, axes=list(range(axis, x.rank)), epsilon=eps, name=lowerer.fresh_name(node)
    )
    if scale is not None:
        kwargs["gamma"] = scale
    if bias is not None:
        kwargs["beta"] = bias
    return [lowerer.mb.layer_norm(**kwargs)]


def _pool(mil_name: str, extra_kwargs: Optional[Callable] = None):
    def handler(lowerer, node, ins, attrs):
        if len(node.output) > 1:
            raise RuntimeError(
                f"{node.op_type} with an Indices output is not supported"
            )
        dilations = attrs.get("dilations")
        if dilations and any(d != 1 for d in dilations):
            raise RuntimeError(f"{node.op_type} with dilations != 1 is not supported")
        x = ins[0]
        n = x.rank - 2
        pad_type, pad = _pad_type_and_pads(attrs, n)
        kwargs = dict(
            x=x,
            kernel_sizes=list(attrs["kernel_shape"]),
            strides=list(attrs.get("strides", [1] * n)),
            pad_type=pad_type,
            ceil_mode=bool(attrs.get("ceil_mode", 0)),
            name=lowerer.fresh_name(node),
        )
        if pad is not None:
            kwargs["pad"] = pad
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs(attrs))
        return [getattr(lowerer.mb, mil_name)(**kwargs)]

    return handler


_OP_HANDLERS["MaxPool"] = _pool("max_pool")
_OP_HANDLERS["AveragePool"] = _pool(
    "avg_pool",
    lambda attrs: {
        "exclude_padding_from_average": not bool(attrs.get("count_include_pad", 0))
    },
)


def _global_reduce(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        x = ins[0]
        axes = list(range(2, x.rank))
        fn = getattr(lowerer.mb, mil_name)
        return [fn(x=x, axes=axes, keep_dims=True, name=lowerer.fresh_name(node))]

    return handler


_OP_HANDLERS["GlobalAveragePool"] = _global_reduce("reduce_mean")
_OP_HANDLERS["GlobalMaxPool"] = _global_reduce("reduce_max")


def _reduce(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        x = ins[0]
        axes = attrs.get("axes")
        if len(ins) > 1 and ins[1] is not None:
            axes = [int(v) for v in ins[1].val]
        keep_dims = bool(attrs.get("keepdims", 1))
        kwargs = dict(x=x, keep_dims=keep_dims, name=lowerer.fresh_name(node))
        if axes is not None:
            kwargs["axes"] = [int(a) if a >= 0 else int(a) + x.rank for a in axes]
        return [getattr(lowerer.mb, mil_name)(**kwargs)]

    return handler


_OP_HANDLERS["ReduceMean"] = _reduce("reduce_mean")
_OP_HANDLERS["ReduceSum"] = _reduce("reduce_sum")
_OP_HANDLERS["ReduceMax"] = _reduce("reduce_max")


@_register("Reshape")
def _op_reshape(lowerer, node, ins, attrs):
    shape_var = ins[1]
    if shape_var.val is None:
        raise RuntimeError("Reshape requires a compile-time-constant 'shape' input")
    shape = [int(v) for v in shape_var.val]
    return [lowerer.mb.reshape(x=ins[0], shape=shape, name=lowerer.fresh_name(node))]


@_register("Transpose")
def _op_transpose(lowerer, node, ins, attrs):
    x = ins[0]
    perm = list(attrs.get("perm", list(reversed(range(x.rank)))))
    return [lowerer.mb.transpose(x=x, perm=perm, name=lowerer.fresh_name(node))]


@_register("Flatten")
def _op_flatten(lowerer, node, ins, attrs):
    axis = int(attrs.get("axis", 1))
    return [lowerer.mb.flatten2d(x=ins[0], axis=axis, name=lowerer.fresh_name(node))]


@_register("Squeeze")
def _op_squeeze(lowerer, node, ins, attrs):
    x = ins[0]
    axes = None
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(v) for v in ins[1].val]
    elif "axes" in attrs:
        axes = list(attrs["axes"])
    kwargs = dict(x=x, name=lowerer.fresh_name(node))
    if axes is not None:
        kwargs["axes"] = [int(a) if a >= 0 else int(a) + x.rank for a in axes]
    return [lowerer.mb.squeeze(**kwargs)]


@_register("Unsqueeze")
def _op_unsqueeze(lowerer, node, ins, attrs):
    x = ins[0]
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(v) for v in ins[1].val]
    else:
        axes = list(attrs["axes"])
    out_rank = x.rank + len(axes)
    axes = sorted(a if a >= 0 else a + out_rank for a in axes)
    return [lowerer.mb.expand_dims(x=x, axes=axes, name=lowerer.fresh_name(node))]


@_register("Concat")
def _op_concat(lowerer, node, ins, attrs):
    axis = int(attrs["axis"])
    axis = axis if axis >= 0 else axis + ins[0].rank
    return [lowerer.mb.concat(values=ins, axis=axis, name=lowerer.fresh_name(node))]


@_register("Split")
def _op_split(lowerer, node, ins, attrs):
    x = ins[0]
    axis = int(attrs.get("axis", 0))
    axis = axis if axis >= 0 else axis + x.rank
    split_sizes = attrs.get("split")
    if len(ins) > 1 and ins[1] is not None:
        split_sizes = [int(v) for v in ins[1].val]
    name = lowerer.fresh_name(node)
    if split_sizes is not None:
        outs = lowerer.mb.split(
            x=x, split_sizes=[int(s) for s in split_sizes], axis=axis, name=name
        )
    else:
        outs = lowerer.mb.split(x=x, num_splits=len(node.output), axis=axis, name=name)
    return list(outs)


@_register("Pad")
def _op_pad(lowerer, node, ins, attrs):
    x = ins[0]
    if len(ins) > 1 and ins[1] is not None:
        pads_flat = [int(v) for v in ins[1].val]
    else:
        pads_flat = [int(v) for v in attrs["pads"]]
    mode = attrs.get("mode", "constant")
    mil_mode = {"constant": "constant", "reflect": "reflect", "edge": "replicate"}.get(
        mode
    )
    if mil_mode is None:
        raise RuntimeError(
            f"Pad mode {mode!r} is not supported (supported: constant, reflect, edge)"
        )
    const_val = 0.0
    if len(ins) > 2 and ins[2] is not None:
        const_val = _const_scalar(ins[2])
    elif "value" in attrs:
        const_val = float(attrs["value"])
    n = x.rank
    begins, ends = pads_flat[:n], pads_flat[n:]
    interleaved: List[int] = []
    for b, e in zip(begins, ends):
        interleaved += [b, e]
    return [
        lowerer.mb.pad(
            x=x,
            pad=interleaved,
            mode=mil_mode,
            constant_val=float(const_val),
            name=lowerer.fresh_name(node),
        )
    ]


@_register("Slice")
def _op_slice(lowerer, node, ins, attrs):
    x = ins[0]
    rank = x.rank
    shape = x.shape
    if any(not isinstance(d, (int, np.integer)) for d in shape):
        raise RuntimeError("Slice requires a fully static input shape")
    if len(ins) > 1:
        starts = [int(v) for v in ins[1].val]
        ends = [int(v) for v in ins[2].val]
        axes = (
            [int(v) for v in ins[3].val]
            if len(ins) > 3 and ins[3] is not None
            else list(range(len(starts)))
        )
        steps = (
            [int(v) for v in ins[4].val]
            if len(ins) > 4 and ins[4] is not None
            else [1] * len(starts)
        )
    else:
        starts = [int(v) for v in attrs["starts"]]
        ends = [int(v) for v in attrs["ends"]]
        axes = [int(v) for v in attrs.get("axes", range(len(starts)))]
        steps = [1] * len(starts)

    # Reproduce ONNX Slice's own clamp algorithm (numpy-style negative-index
    # wraparound, then clamp) rather than Python's `slice.indices()`: for a
    # negative stride, a fully-clamped end of -1 means "include index 0", but MIL's
    # `end` re-wraps a literal -1 to `dim - 1` (numpy indexing semantics), which
    # would silently turn that into an empty slice. Route that one case through
    # `end_mask` (MIL's "ignore `end`, go to the natural boundary" flag) instead.
    begin, end, stride = [0] * rank, list(shape), [1] * rank
    end_mask = [False] * rank
    for s, e, a, st in zip(starts, ends, axes, steps):
        a = a if a >= 0 else a + rank
        dim = int(shape[a])
        s = s + dim if s < 0 else s
        e = e + dim if e < 0 else e
        if st > 0:
            s = max(0, min(s, dim))
            e = max(0, min(e, dim))
        else:
            s = max(0, min(s, dim - 1))
            e = max(-1, min(e, dim - 1))
        begin[a], stride[a] = s, st
        if e == -1 and st < 0:
            end_mask[a] = True
            end[a] = 0  # ignored: end_mask[a] takes over
        else:
            end[a] = e
    kwargs = dict(
        x=x, begin=begin, end=end, stride=stride, name=lowerer.fresh_name(node)
    )
    if any(end_mask):
        kwargs["end_mask"] = end_mask
    return [lowerer.mb.slice_by_index(**kwargs)]


@_register("Gather")
def _op_gather(lowerer, node, ins, attrs):
    x, indices = ins[0], ins[1]
    axis = int(attrs.get("axis", 0))
    axis = axis if axis >= 0 else axis + x.rank
    return [
        lowerer.mb.gather(
            x=x, indices=indices, axis=axis, name=lowerer.fresh_name(node)
        )
    ]


@_register("Tile")
def _op_tile(lowerer, node, ins, attrs):
    reps = ins[1]
    if reps.val is None:
        raise RuntimeError("Tile requires a compile-time-constant 'repeats' input")
    return [
        lowerer.mb.tile(
            x=ins[0], reps=[int(v) for v in reps.val], name=lowerer.fresh_name(node)
        )
    ]


_ONNX_TO_MIL_CAST = {
    onnx.TensorProto.FLOAT: "fp32",
    onnx.TensorProto.FLOAT16: "fp16",
    onnx.TensorProto.DOUBLE: "fp32",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int32",
    onnx.TensorProto.BOOL: "bool",
}


@_register("Cast")
def _op_cast(lowerer, node, ins, attrs):
    to = int(attrs["to"])
    dtype = _ONNX_TO_MIL_CAST.get(to)
    if dtype is None:
        raise RuntimeError(
            f"Cast to {onnx.TensorProto.DataType.Name(to)} is not supported "
            "(supported targets: float16, float32, double, int32, int64, bool)"
        )
    return [lowerer.mb.cast(x=ins[0], dtype=dtype, name=lowerer.fresh_name(node))]


@_register("Constant")
def _op_constant(lowerer, node, ins, attrs):
    if "value" in attrs:
        arr = np.asarray(attrs["value"])
    elif "value_float" in attrs:
        arr = np.array(attrs["value_float"], dtype=np.float32)
    elif "value_floats" in attrs:
        arr = np.array(list(attrs["value_floats"]), dtype=np.float32)
    elif "value_int" in attrs:
        arr = np.array(attrs["value_int"], dtype=np.int32)
    elif "value_ints" in attrs:
        arr = np.array(list(attrs["value_ints"]), dtype=np.int32)
    else:
        raise RuntimeError(
            "Constant node has no supported value attribute (supported: value, "
            "value_float(s), value_int(s))"
        )
    return [lowerer.make_const(node.output[0], arr)]


SUPPORTED_ONNX_OPS = tuple(sorted(_OP_HANDLERS))


def _build_mil_program(model: onnx.ModelProto, mb, types, Function, Program):
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    opset = 1
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = imp.version

    lowerer = _Lowerer(mb=mb, types=types, opset=opset)

    input_specs = {}
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        input_specs[inp.name] = _make_input_spec(inp, mb, types)

    with Function(input_specs) as func:
        for name, var in func.inputs.items():
            lowerer.bind(name, var)
        for init in graph.initializer:
            lowerer.bind(
                init.name, lowerer.make_const(init.name, numpy_helper.to_array(init))
            )
        for node in graph.node:
            lowerer.lower_node(node)
        outputs = [
            mb.identity(x=lowerer.get(out.name), name=out.name) for out in graph.output
        ]
        func.set_outputs(outputs)

    prog = Program()
    prog.add_function("main", func)
    return prog


def _resolve_compute_units(ct, compute_units: Union[str, Any]):
    if isinstance(compute_units, str):
        try:
            return getattr(ct.ComputeUnit, compute_units)
        except AttributeError as exc:
            valid = ", ".join(u.name for u in ct.ComputeUnit)
            raise RuntimeError(
                f"Unknown compute_units {compute_units!r}; valid values: {valid}"
            ) from exc
    return compute_units


def _resolve_deployment_target(ct, target: Union[str, Any]):
    if isinstance(target, str):
        try:
            return getattr(ct.target, target)
        except AttributeError as exc:
            raise RuntimeError(f"Unknown minimum_deployment_target {target!r}") from exc
    return target


def convert_to_coreml(
    model: onnx.ModelProto,
    *,
    convert_to: str = "mlprogram",
    compute_units: Union[str, Any] = "ALL",
    compute_precision: Optional[Any] = None,
    minimum_deployment_target: Optional[Union[str, Any]] = None,
    skip_model_load: bool = True,
):
    """Convert an ONNX model to an in-memory Core ML model.

    Parameters
    ----------
    model:
        The ONNX model to convert. Typically the output of :func:`onnxsim.simplify`.
        All graph inputs must have fully static shapes (see ``_make_input_spec``);
        every node's op must be one of ``SUPPORTED_ONNX_OPS``.
    convert_to:
        Core ML model type: ``"mlprogram"`` (the modern ``.mlpackage`` format,
        default) or ``"neuralnetwork"`` (the legacy ``.mlmodel`` format).
    compute_units:
        Which compute devices the model may run on: ``"ALL"`` (default),
        ``"CPU_ONLY"``, ``"CPU_AND_GPU"``, or ``"CPU_AND_NE"``. Accepts either the
        string name or a ``coremltools.ComputeUnit`` member.
    compute_precision:
        Forwarded to ``coremltools.convert`` (e.g. ``coremltools.precision.FLOAT16``);
        left at coremltools' own default when ``None``.
    minimum_deployment_target:
        Minimum OS version the model must run on, e.g. ``"iOS16"``/``"macOS13"``, or a
        ``coremltools.target`` member. Left at coremltools' own default when ``None``.
    skip_model_load:
        Skip compiling/loading the produced model for prediction (default ``True``).
        Compiling requires Apple's Core ML toolchain and only succeeds on macOS; leave
        this ``True`` to convert models on any OS, or pass ``False`` on macOS to get a
        model that's ready to call ``.predict()`` on.

    Returns
    -------
    coremltools.models.MLModel

    Raises
    ------
    RuntimeError
        If coremltools is not installed, an input has a non-static shape, or the graph
        uses an ONNX op/feature this translator does not support.
    """
    ct = _import_coremltools()
    mb, types, Function, Program = _import_mil()

    prog = _build_mil_program(model, mb, types, Function, Program)

    kwargs: Dict[str, Any] = {}
    if compute_precision is not None:
        kwargs["compute_precision"] = compute_precision
    if minimum_deployment_target is not None:
        kwargs["minimum_deployment_target"] = _resolve_deployment_target(
            ct, minimum_deployment_target
        )

    return ct.convert(
        prog,
        source="milinternal",
        convert_to=convert_to,
        compute_units=_resolve_compute_units(ct, compute_units),
        skip_model_load=skip_model_load,
        **kwargs,
    )


def export_coreml(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    **kwargs,
):
    """Convert ``model`` to Core ML, optionally saving it to ``output_path``.

    This is the public entry point used by the ``onnxsim --emit-coreml`` CLI and is
    re-exported as ``onnxsim.export_coreml``. It returns the ``MLModel`` regardless of
    whether ``output_path`` is given, so it is equally usable in-memory.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    output_path:
        If given, the model is saved here with ``MLModel.save`` -- a ``.mlpackage``
        directory for ``convert_to="mlprogram"`` (the default), or a ``.mlmodel`` file
        for ``convert_to="neuralnetwork"``. If ``None``, the model is only returned.

    Other keyword arguments are forwarded to :func:`convert_to_coreml`.

    Returns
    -------
    coremltools.models.MLModel
    """
    mlmodel = convert_to_coreml(model, **kwargs)
    if output_path is not None:
        mlmodel.save(output_path)
    return mlmodel
