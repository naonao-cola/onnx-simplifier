"""Convert a (simplified) ONNX model to TensorFlow Lite.

onnxsim's job stops at a cleaned-up ``onnx.ModelProto``. Mobile/embedded runtimes
built on TensorFlow Lite want that graph as a ``.tflite`` flatbuffer instead. This
module bridges the two: it walks the ONNX graph node by node, builds the equivalent
computation with plain TensorFlow ops inside a ``tf.function``, traces it into a
concrete function, and hands that to ``tf.lite.TFLiteConverter`` to produce the
actual ``.tflite`` model.

There is no maintained "convert this ONNX model" entry point to lean on here either
(``onnx-tensorflow``/``onnx-tf`` has been unmaintained for years and only tracks very
old opsets) -- same situation as Core ML after coremltools dropped its ONNX frontend,
see ``coreml_export.py``. This translator plays that role, one ONNX op at a time. It
covers a practical subset of ops (common to CNN/MLP graphs: conv, pooling,
normalization, matmul/gemm, elementwise math, reshapes, reductions, ...); a node whose
op isn't in ``SUPPORTED_ONNX_OPS`` raises a ``RuntimeError`` naming the op, rather than
silently producing a wrong model.

Feeding a *simplified* model in is the point, same as with ``coreml_export.py``:
onnxsim's constant folding turns shape-manipulation subgraphs into plain
initializers, so parameters this translator needs at conversion time (a ``Reshape``'s
target shape, a ``Slice``'s bounds, ...) are far more likely to already be constants
by the time they reach here instead of values only known at runtime.

Like onnxruntime for constant folding and coremltools for Core ML export, TensorFlow
is an **optional** dependency: nothing here is imported at ``import onnxsim`` time,
only when ``--emit-tflite`` / the ``export_tflite`` API run. A missing TensorFlow
raises a ``RuntimeError`` with an install hint.

Graph inputs must have fully static shapes (dynamic axes aren't supported) -- pin them
first with onnxsim's own ``--overwrite-input-shape``/``--test-input-shape`` if needed.
TensorFlow Lite's own op kernels are NHWC-only, while ONNX's conv/pool ops are NCHW;
this translator keeps the graph's public tensors in ONNX's NCHW layout and transposes
to/from NHWC only around the ops (``Conv``/``MaxPool``/``AveragePool``) that need it.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

_TFLITE_INSTALL_HINT = (
    "TensorFlow is required to export TFLite models but is not installed. "
    "Install it with `pip install tensorflow` (or `tensorflow-cpu`)."
)

# ONNX TensorProto dtypes this translator can carry through to TensorFlow/TFLite,
# downcasting where TFLite has no matching type (float64 -> float32, int64 -> int32).
_NP_DOWNCAST = {
    np.dtype(np.float64): np.float32,
    np.dtype(np.int64): np.int32,
}
_SUPPORTED_NP_DTYPES = {np.float32, np.float16, np.int32, np.bool_}


def has_tensorflow() -> bool:
    """Whether TensorFlow is importable in this environment."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(_TFLITE_INSTALL_HINT) from exc
    return tf


def _as_tf_array(arr: np.ndarray) -> np.ndarray:
    """Downcast ``arr`` to a dtype TensorFlow Lite supports, or raise."""
    arr = np.asarray(arr)
    if arr.dtype == np.int64:
        # Saturate rather than wrap: ONNX graphs routinely use INT64_MAX/MIN as
        # "unbounded" sentinels, and a plain .astype(int32) wraps those around to a
        # small in-range number instead of a large one, silently corrupting the
        # sentinel.
        arr = np.clip(arr, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    target = _NP_DOWNCAST.get(arr.dtype)
    if target is not None:
        arr = arr.astype(target)
    if arr.dtype.type not in _SUPPORTED_NP_DTYPES:
        raise RuntimeError(
            f"Unsupported tensor dtype {arr.dtype} for TFLite export (supported: "
            "float16, float32, float64, int32, int64, bool; 64-bit types are "
            "downcast to their 32-bit equivalent)."
        )
    return arr


def _onnx_elem_type_to_tf(elem_type: int, tf) -> Any:
    TP = onnx.TensorProto
    mapping = {
        TP.FLOAT: tf.float32,
        TP.FLOAT16: tf.float16,
        TP.DOUBLE: tf.float32,
        TP.INT32: tf.int32,
        TP.INT64: tf.int32,
        TP.BOOL: tf.bool,
    }
    if elem_type not in mapping:
        raise RuntimeError(
            f"Unsupported input dtype {TP.DataType.Name(elem_type)}; onnxsim's "
            "TFLite exporter supports float16, float32, float64, int32, int64, and "
            "bool graph inputs (64-bit types are represented as their 32-bit "
            "equivalent in the exported model)."
        )
    return mapping[elem_type]


def _static_input_shape(value_info: onnx.ValueInfoProto) -> List[int]:
    t = value_info.type.tensor_type
    if not t.HasField("shape"):
        raise RuntimeError(
            f"input '{value_info.name}' has no shape information; onnxsim's TFLite "
            "exporter requires fully static input shapes."
        )
    shape = []
    for i, d in enumerate(t.shape.dim):
        if d.HasField("dim_value"):
            shape.append(int(d.dim_value))
        else:
            label = d.dim_param or "<unknown>"
            raise RuntimeError(
                f"input '{value_info.name}' has a dynamic dimension (dim {i}: "
                f"{label!r}); onnxsim's TFLite exporter requires fully static "
                "input shapes -- pin it first with onnxsim's own "
                "--overwrite-input-shape/--test-input-shape."
            )
    return shape


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


def _compute_spatial_pad(
    attrs: Dict[str, Any],
    in_shape: List[int],
    kernel_shape: List[int],
    strides: List[int],
    dilations: List[int],
) -> List[Tuple[int, int]]:
    """Resolve ONNX ``auto_pad``/``pads`` into explicit ``(before, after)`` pairs per
    spatial axis, so Conv/pooling can always be run as an explicit ``tf.pad`` +
    ``"VALID"`` -- avoiding any ambiguity between ONNX's and TensorFlow's own
    ``"SAME"`` padding conventions (in particular, TensorFlow has no equivalent of
    ONNX's ``SAME_LOWER``).
    """
    n = len(kernel_shape)
    auto_pad = attrs.get("auto_pad", "NOTSET")
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        pads = []
        for i in range(n):
            eff_k = (kernel_shape[i] - 1) * dilations[i] + 1
            out_size = -(-in_shape[i] // strides[i])  # ceil division
            total_pad = max((out_size - 1) * strides[i] + eff_k - in_shape[i], 0)
            if auto_pad == "SAME_UPPER":
                before = total_pad // 2
            else:
                before = total_pad - total_pad // 2
            pads.append((before, total_pad - before))
        return pads
    if auto_pad == "VALID":
        return [(0, 0)] * n
    raw = attrs.get("pads")
    if not raw:
        return [(0, 0)] * n
    begins, ends = raw[:n], raw[n:]
    return [(int(b), int(e)) for b, e in zip(begins, ends)]


def _avg_pool_counts(
    in_size: int, k: int, s: int, pad_before: int, pad_after: int
) -> np.ndarray:
    """Per-output-position count of *non-padded* input elements inside the pooling
    window, along one spatial axis -- used to implement ONNX AveragePool's default
    ``count_include_pad=0`` (TFLite/TF's own average pool has no such option and
    always divides by the full window area)."""
    padded = in_size + pad_before + pad_after
    out_size = (padded - k) // s + 1
    counts = np.empty(out_size, dtype=np.float32)
    for o in range(out_size):
        start = o * s - pad_before
        end = start + k
        valid = min(end, in_size) - max(start, 0)
        counts[o] = max(valid, 1)
    return counts


class Val:
    """A traced TensorFlow tensor, plus its compile-time value when known.

    Mirrors coremltools MIL's ``Var.val``: most ONNX ops this translator lowers are
    plain tensor math and only need ``.t`` (the traced ``tf.Tensor``), but a handful
    of ops (``Reshape``'s target shape, ``Slice``'s bounds, ``Gather``'s indices, ...)
    need an actual Python/NumPy value at conversion time, not just a traced tensor --
    ``.const`` carries that when the value came from an initializer or was computed
    from other compile-time constants.
    """

    __slots__ = ("t", "const")

    def __init__(self, t, const: Optional[np.ndarray] = None):
        self.t = t
        self.const = const


class _Lowerer:
    """Walks an ONNX graph once, building the equivalent TensorFlow ops as it goes."""

    def __init__(self, tf):
        self.tf = tf
        self._values: Dict[str, Val] = {}

    def bind(self, name: str, val: Val) -> None:
        self._values[name] = val

    def get(self, name: str) -> Val:
        if name not in self._values:
            raise RuntimeError(
                f"reference to unknown tensor '{name}' (the graph may not be "
                "topologically sorted, or the producing node uses an unsupported "
                "feature)"
            )
        return self._values[name]

    def lower_node(self, node: onnx.NodeProto) -> None:
        handler = _OP_HANDLERS.get(node.op_type)
        if handler is None:
            raise RuntimeError(
                f"ONNX op '{node.op_type}' is not supported by onnxsim's TFLite "
                f"exporter (node {node.name or (node.output[0] if node.output else '')!r})."
                " Supported ops: " + ", ".join(sorted(_OP_HANDLERS))
            )
        ins = [self.get(name) if name else None for name in node.input]
        attrs = _node_attrs(node)
        try:
            outs = handler(self, node, ins, attrs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to convert ONNX node {node.name or node.output[0]!r} "
                f"({node.op_type}) to TFLite: {exc}"
            ) from exc
        for out_name, val in zip(node.output, outs):
            if out_name:
                self.bind(out_name, val)


# ---------------------------------------------------------------------------
# Op handlers. Each handler is ``(lowerer, node, ins, attrs) -> list[Val]``, with
# ``ins[i]`` the already-converted ``Val`` for ``node.input[i]`` (``None`` for an
# omitted optional input) and ``attrs`` the node's parsed attribute dict.
# ---------------------------------------------------------------------------

_OP_HANDLERS: Dict[str, Any] = {}


def _register(*op_types: str):
    def deco(fn):
        for op_type in op_types:
            _OP_HANDLERS[op_type] = fn
        return fn

    return deco


def _require_const(val: Val, what: str) -> np.ndarray:
    if val is None or val.const is None:
        raise RuntimeError(
            f"expected a compile-time constant for {what}, but it is only known at "
            "runtime (onnxsim's TFLite exporter can't trace dynamic values)"
        )
    return np.asarray(val.const)


def _simple_unary(dotted_tf_name: str):
    """``dotted_tf_name`` is a dotted attribute path off ``tf``, e.g. ``"nn.relu"``
    or ``"math.log"``, resolved lazily (only once TensorFlow is actually imported)."""

    def handler(lowerer, node, ins, attrs):
        fn = lowerer.tf
        for part in dotted_tf_name.split("."):
            fn = getattr(fn, part)
        return [Val(fn(ins[0].t))]

    return handler


for _onnx_op, _tf_name in [
    ("Relu", "nn.relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("Neg", "negative"),
    ("Abs", "abs"),
    ("Sqrt", "sqrt"),
    ("Exp", "exp"),
    ("Log", "math.log"),
    ("Erf", "math.erf"),
    ("Identity", "identity"),
]:
    _OP_HANDLERS[_onnx_op] = _simple_unary(_tf_name)


def _binary(tf_name: str, np_fn):
    def handler(lowerer, node, ins, attrs):
        tf = lowerer.tf
        fn = getattr(tf, tf_name)
        t = ins[0].t
        for other in ins[1:]:
            t = fn(t, other.t)
        const = None
        if all(i.const is not None for i in ins):
            const = np.asarray(ins[0].const)
            for other in ins[1:]:
                const = np_fn(const, np.asarray(other.const))
        return [Val(t, const)]

    return handler


for _onnx_op, _tf_name, _np_fn in [
    ("Add", "add", np.add),
    ("Sub", "subtract", np.subtract),
    ("Mul", "multiply", np.multiply),
    ("Div", "divide", np.divide),
    ("Pow", "pow", np.power),
    ("Max", "maximum", np.maximum),
    ("Min", "minimum", np.minimum),
]:
    _OP_HANDLERS[_onnx_op] = _binary(_tf_name, _np_fn)


@_register("LeakyRelu")
def _op_leaky_relu(lowerer, node, ins, attrs):
    tf = lowerer.tf
    alpha = float(attrs.get("alpha", 0.01))
    return [Val(tf.nn.leaky_relu(ins[0].t, alpha=alpha))]


@_register("Gelu")
def _op_gelu(lowerer, node, ins, attrs):
    tf = lowerer.tf
    approximate = attrs.get("approximate", "none") == "tanh"
    return [Val(tf.nn.gelu(ins[0].t, approximate=approximate))]


@_register("Softmax")
def _op_softmax(lowerer, node, ins, attrs):
    tf = lowerer.tf
    axis = int(attrs.get("axis", -1))
    return [Val(tf.nn.softmax(ins[0].t, axis=axis))]


@_register("Clip")
def _op_clip(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    min_v = attrs.get("min")
    max_v = attrs.get("max")
    if len(ins) > 1 and ins[1] is not None:
        min_v = float(_require_const(ins[1], "Clip's 'min' input").reshape(-1)[0])
    if len(ins) > 2 and ins[2] is not None:
        max_v = float(_require_const(ins[2], "Clip's 'max' input").reshape(-1)[0])
    t = x.t
    if min_v is not None:
        t = tf.maximum(t, min_v)
    if max_v is not None:
        t = tf.minimum(t, max_v)
    return [Val(t)]


@_register("Cast")
def _op_cast(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    dtype = _onnx_elem_type_to_tf(int(attrs["to"]), tf)
    t = tf.cast(x.t, dtype)
    const = (
        np.asarray(x.const).astype(dtype.as_numpy_dtype)
        if x.const is not None
        else None
    )
    return [Val(t, const)]


@_register("MatMul")
def _op_matmul(lowerer, node, ins, attrs):
    tf = lowerer.tf
    a, b = ins
    return [Val(tf.matmul(a.t, b.t))]


@_register("Gemm")
def _op_gemm(lowerer, node, ins, attrs):
    tf = lowerer.tf
    a, b = ins[0], ins[1]
    c = ins[2] if len(ins) > 2 else None
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = bool(attrs.get("transA", 0))
    trans_b = bool(attrs.get("transB", 0))
    y = tf.matmul(a.t, b.t, transpose_a=trans_a, transpose_b=trans_b)
    if alpha != 1.0:
        y = y * alpha
    if c is not None:
        bias = c.t if beta == 1.0 else c.t * beta
        y = y + bias
    return [Val(y)]


@_register("Conv")
def _op_conv(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x, w = ins[0], ins[1]
    b = ins[2] if len(ins) > 2 else None
    w_shape = w.t.shape.as_list()
    kernel_shape = [int(k) for k in attrs.get("kernel_shape", w_shape[2:4])]
    if len(kernel_shape) != 2:
        raise RuntimeError("only 2-D Conv is supported by onnxsim's TFLite exporter")
    strides = [int(s) for s in attrs.get("strides", [1, 1])]
    dilations = [int(d) for d in attrs.get("dilations", [1, 1])]
    group = int(attrs.get("group", 1))
    out_c, in_c_per_group = w_shape[0], w_shape[1]
    x_shape = x.t.shape.as_list()
    in_c = x_shape[1]

    pads = _compute_spatial_pad(attrs, x_shape[2:4], kernel_shape, strides, dilations)
    filt = tf.transpose(w.t, [2, 3, 1, 0])  # OIHW -> HWIO
    x_nhwc = tf.transpose(x.t, [0, 2, 3, 1])
    if any(p != (0, 0) for p in pads):
        x_nhwc = tf.pad(x_nhwc, [[0, 0], list(pads[0]), list(pads[1]), [0, 0]])

    conv_strides = [1, strides[0], strides[1], 1]
    conv_dilations = [1, dilations[0], dilations[1], 1]
    if group == 1:
        y = tf.nn.conv2d(
            x_nhwc,
            filt,
            strides=conv_strides,
            padding="VALID",
            dilations=conv_dilations,
        )
    elif in_c_per_group == 1 and group == in_c:
        multiplier = out_c // group
        dw_filt = tf.reshape(filt, [kernel_shape[0], kernel_shape[1], in_c, multiplier])
        y = tf.nn.depthwise_conv2d(
            x_nhwc,
            dw_filt,
            strides=conv_strides,
            padding="VALID",
            dilations=[dilations[0], dilations[1]],
        )
    else:
        x_groups = tf.split(x_nhwc, group, axis=3)
        w_groups = tf.split(filt, group, axis=3)
        y = tf.concat(
            [
                tf.nn.conv2d(
                    xg,
                    wg,
                    strides=conv_strides,
                    padding="VALID",
                    dilations=conv_dilations,
                )
                for xg, wg in zip(x_groups, w_groups)
            ],
            axis=3,
        )
    if b is not None:
        y = tf.nn.bias_add(y, b.t)
    y = tf.transpose(y, [0, 3, 1, 2])
    return [Val(y)]


def _pool_2d(reduce_kind: str):
    def handler(lowerer, node, ins, attrs):
        tf = lowerer.tf
        x = ins[0]
        x_shape = x.t.shape.as_list()
        kernel_shape = [int(k) for k in attrs["kernel_shape"]]
        if len(kernel_shape) != 2:
            raise RuntimeError(
                "only 2-D pooling is supported by onnxsim's TFLite exporter"
            )
        strides = [int(s) for s in attrs.get("strides", kernel_shape)]
        dilations = [int(d) for d in attrs.get("dilations", [1, 1])]
        if any(d != 1 for d in dilations):
            raise RuntimeError(
                "dilated pooling is not supported by onnxsim's TFLite exporter"
            )
        if int(attrs.get("ceil_mode", 0)):
            raise RuntimeError(
                "ceil_mode=1 pooling is not supported by onnxsim's TFLite exporter"
            )
        in_hw = x_shape[2:4]
        pads = _compute_spatial_pad(attrs, in_hw, kernel_shape, strides, dilations)
        pad_needed = any(p != (0, 0) for p in pads)
        x_nhwc = tf.transpose(x.t, [0, 2, 3, 1])

        if reduce_kind == "max":
            if pad_needed:
                x_nhwc = tf.pad(
                    x_nhwc,
                    [[0, 0], list(pads[0]), list(pads[1]), [0, 0]],
                    constant_values=float("-inf"),
                )
            y = tf.nn.max_pool2d(
                x_nhwc, ksize=kernel_shape, strides=strides, padding="VALID"
            )
        else:
            count_include_pad = int(attrs.get("count_include_pad", 0))
            if pad_needed:
                x_nhwc = tf.pad(x_nhwc, [[0, 0], list(pads[0]), list(pads[1]), [0, 0]])
            window_area = kernel_shape[0] * kernel_shape[1]
            sum_pool = (
                tf.nn.avg_pool2d(
                    x_nhwc, ksize=kernel_shape, strides=strides, padding="VALID"
                )
                * window_area
            )
            if pad_needed and not count_include_pad:
                counts_h = _avg_pool_counts(
                    in_hw[0], kernel_shape[0], strides[0], pads[0][0], pads[0][1]
                )
                counts_w = _avg_pool_counts(
                    in_hw[1], kernel_shape[1], strides[1], pads[1][0], pads[1][1]
                )
                divisor = np.outer(counts_h, counts_w).astype(np.float32)
                divisor = divisor.reshape(1, divisor.shape[0], divisor.shape[1], 1)
                y = sum_pool / tf.constant(divisor)
            else:
                y = sum_pool / float(window_area)
        return [Val(tf.transpose(y, [0, 3, 1, 2]))]

    return handler


_OP_HANDLERS["MaxPool"] = _pool_2d("max")
_OP_HANDLERS["AveragePool"] = _pool_2d("avg")


@_register("GlobalAveragePool")
def _op_global_avg_pool(lowerer, node, ins, attrs):
    tf = lowerer.tf
    return [Val(tf.reduce_mean(ins[0].t, axis=[2, 3], keepdims=True))]


@_register("GlobalMaxPool")
def _op_global_max_pool(lowerer, node, ins, attrs):
    tf = lowerer.tf
    return [Val(tf.reduce_max(ins[0].t, axis=[2, 3], keepdims=True))]


def _reduce(tf_name: str):
    def handler(lowerer, node, ins, attrs):
        tf = lowerer.tf
        x = ins[0]
        x_shape = x.t.shape.as_list()
        rank = len(x_shape)
        if len(ins) > 1 and ins[1] is not None:
            axes = [int(a) for a in _require_const(ins[1], f"{node.op_type}'s 'axes'")]
        elif "axes" in attrs:
            axes = [int(a) for a in attrs["axes"]]
        else:
            axes = list(range(rank))
        axes = sorted(a % rank for a in axes)
        keepdims = bool(attrs.get("keepdims", 1))
        fn = getattr(tf, tf_name)
        return [Val(fn(x.t, axis=axes, keepdims=keepdims))]

    return handler


for _onnx_op, _tf_name in [
    ("ReduceMean", "reduce_mean"),
    ("ReduceSum", "reduce_sum"),
    ("ReduceMax", "reduce_max"),
    ("ReduceMin", "reduce_min"),
    ("ReduceProd", "reduce_prod"),
]:
    _OP_HANDLERS[_onnx_op] = _reduce(_tf_name)


@_register("BatchNormalization")
def _op_batch_norm(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x, scale, bias, mean, var = ins[0], ins[1], ins[2], ins[3], ins[4]
    eps = float(attrs.get("epsilon", 1e-5))
    c = scale.t.shape.as_list()[0]
    shape = [1, c, 1, 1]
    s = tf.reshape(scale.t, shape)
    b = tf.reshape(bias.t, shape)
    m = tf.reshape(mean.t, shape)
    v = tf.reshape(var.t, shape)
    y = (x.t - m) / tf.sqrt(v + eps) * s + b
    return [Val(y)]


@_register("Reshape")
def _op_reshape(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    target = [int(v) for v in _require_const(ins[1], "Reshape's 'shape' input")]
    allowzero = int(attrs.get("allowzero", 0))
    x_shape = x.t.shape.as_list()
    resolved = [
        x_shape[i] if d == 0 and not allowzero else d for i, d in enumerate(target)
    ]
    t = tf.reshape(x.t, resolved)
    const = np.reshape(np.asarray(x.const), resolved) if x.const is not None else None
    return [Val(t, const)]


@_register("Flatten")
def _op_flatten(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    shape = x.t.shape.as_list()
    axis = int(attrs.get("axis", 1)) % (len(shape) + 1)
    outer = int(np.prod(shape[:axis], dtype=np.int64))
    inner = int(np.prod(shape[axis:], dtype=np.int64))
    return [Val(tf.reshape(x.t, [outer, inner]))]


@_register("Squeeze")
def _op_squeeze(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    x_shape = x.t.shape.as_list()
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(a) for a in _require_const(ins[1], "Squeeze's 'axes' input")]
    elif "axes" in attrs:
        axes = [int(a) for a in attrs["axes"]]
    else:
        axes = [i for i, d in enumerate(x_shape) if d == 1]
    axes = {a % len(x_shape) for a in axes}
    new_shape = [d for i, d in enumerate(x_shape) if i not in axes]
    t = tf.reshape(x.t, new_shape)
    const = np.reshape(np.asarray(x.const), new_shape) if x.const is not None else None
    return [Val(t, const)]


@_register("Unsqueeze")
def _op_unsqueeze(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    x_shape = x.t.shape.as_list()
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(a) for a in _require_const(ins[1], "Unsqueeze's 'axes' input")]
    else:
        axes = [int(a) for a in attrs["axes"]]
    out_rank = len(x_shape) + len(axes)
    axes = sorted(a % out_rank for a in axes)
    new_shape = list(x_shape)
    for a in axes:
        new_shape.insert(a, 1)
    t = tf.reshape(x.t, new_shape)
    const = np.reshape(np.asarray(x.const), new_shape) if x.const is not None else None
    return [Val(t, const)]


@_register("Transpose")
def _op_transpose(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    perm = attrs.get("perm")
    if perm is None:
        perm = list(reversed(range(len(x.t.shape))))
    return [Val(tf.transpose(x.t, [int(p) for p in perm]))]


@_register("Concat")
def _op_concat(lowerer, node, ins, attrs):
    tf = lowerer.tf
    axis = int(attrs["axis"])
    t = tf.concat([i.t for i in ins], axis=axis)
    const = None
    if all(i.const is not None for i in ins):
        const = np.concatenate([np.asarray(i.const) for i in ins], axis=axis)
    return [Val(t, const)]


@_register("Split")
def _op_split(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    x_shape = x.t.shape.as_list()
    axis = int(attrs.get("axis", 0)) % len(x_shape)
    if len(ins) > 1 and ins[1] is not None:
        sizes = [int(v) for v in _require_const(ins[1], "Split's 'split' input")]
    elif "split" in attrs:
        sizes = [int(v) for v in attrs["split"]]
    else:
        num_outputs = int(attrs.get("num_outputs", len(node.output)))
        dim = x_shape[axis]
        base, rem = divmod(dim, num_outputs)
        sizes = [base + (1 if i < rem else 0) for i in range(num_outputs)]
    return [Val(o) for o in tf.split(x.t, sizes, axis=axis)]


@_register("Gather")
def _op_gather(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x, idx = ins
    axis = int(attrs.get("axis", 0))
    if idx.const is not None:
        idx_arr = np.asarray(idx.const)
        dim = x.t.shape.as_list()[axis]
        idx_arr = np.where(idx_arr < 0, idx_arr + dim, idx_arr)
        idx_t = tf.constant(idx_arr.astype(np.int32))
    else:
        idx_t = idx.t
    t = tf.gather(x.t, idx_t, axis=axis)
    const = None
    if x.const is not None and idx.const is not None:
        const = np.take(np.asarray(x.const), np.asarray(idx.const), axis=axis)
    return [Val(t, const)]


@_register("Tile")
def _op_tile(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x, reps = ins
    repeats = [int(r) for r in _require_const(reps, "Tile's 'repeats' input")]
    return [Val(tf.tile(x.t, repeats))]


@_register("Pad")
def _op_pad(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    if len(ins) > 1 and ins[1] is not None:
        pads_flat = [int(v) for v in _require_const(ins[1], "Pad's 'pads' input")]
    else:
        pads_flat = [int(v) for v in attrs["pads"]]
    mode = attrs.get("mode", "constant")
    rank = len(x.t.shape)
    begins, ends = pads_flat[:rank], pads_flat[rank:]
    paddings = [[int(b), int(e)] for b, e in zip(begins, ends)]
    if mode == "constant":
        const_value = 0.0
        if len(ins) > 2 and ins[2] is not None and ins[2].const is not None:
            const_value = float(np.asarray(ins[2].const).reshape(-1)[0])
        y = tf.pad(x.t, paddings, mode="CONSTANT", constant_values=const_value)
    elif mode == "reflect":
        y = tf.pad(x.t, paddings, mode="REFLECT")
    else:
        raise RuntimeError(
            f"Pad mode {mode!r} is not supported by onnxsim's TFLite exporter "
            "(supported: constant, reflect)"
        )
    return [Val(y)]


@_register("Slice")
def _op_slice(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    x_shape = x.t.shape.as_list()
    rank = len(x_shape)
    if len(ins) > 1:
        starts = [int(v) for v in _require_const(ins[1], "Slice's 'starts' input")]
        ends = [int(v) for v in _require_const(ins[2], "Slice's 'ends' input")]
        axes = (
            [int(v) for v in _require_const(ins[3], "Slice's 'axes' input")]
            if len(ins) > 3 and ins[3] is not None
            else list(range(len(starts)))
        )
        steps = (
            [int(v) for v in _require_const(ins[4], "Slice's 'steps' input")]
            if len(ins) > 4 and ins[4] is not None
            else [1] * len(starts)
        )
    else:
        starts = [int(v) for v in attrs["starts"]]
        ends = [int(v) for v in attrs["ends"]]
        axes = [int(v) for v in attrs.get("axes", list(range(len(starts))))]
        steps = [1] * len(starts)

    begin = [0] * rank
    end = list(x_shape)
    strides = [1] * rank
    end_mask = 0
    for ax, s, e, st in zip(axes, starts, ends, steps):
        ax = ax % rank
        # slice().indices() implements exactly the clamping semantics ONNX's spec
        # describes for Slice (it's explicitly modeled on numpy/Python slicing).
        norm_s, norm_e, norm_st = slice(s, e, st).indices(x_shape[ax])
        begin[ax], strides[ax] = norm_s, norm_st
        if norm_st < 0 and norm_e == -1:
            # "reverse through index 0": tf.strided_slice has no way to spell this
            # as a literal end index -- like numpy, it wraps a negative end the same
            # way a negative start is wrapped, silently turning -1 back into "the
            # last element" and producing an empty slice instead. `end_mask` is
            # strided_slice's dedicated escape hatch: it tells the op to ignore
            # `end[ax]` and extend to the boundary the stride's direction implies.
            end_mask |= 1 << ax
        else:
            end[ax] = norm_e
    return [Val(tf.strided_slice(x.t, begin, end, strides, end_mask=end_mask))]


@_register("Shape")
def _op_shape(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = ins[0]
    shape = np.array(x.t.shape.as_list(), dtype=np.int64)
    start = int(attrs.get("start", 0))
    end = int(attrs.get("end", len(shape)))
    shape = shape[start:end]
    return [Val(tf.constant(shape.astype(np.int32)), shape)]


@_register("Constant")
def _op_constant(lowerer, node, ins, attrs):
    tf = lowerer.tf
    if "value" in attrs:
        arr = np.asarray(attrs["value"])
    elif "value_float" in attrs:
        arr = np.array(attrs["value_float"], dtype=np.float32)
    elif "value_int" in attrs:
        arr = np.array(attrs["value_int"], dtype=np.int64)
    elif "value_floats" in attrs:
        arr = np.array(list(attrs["value_floats"]), dtype=np.float32)
    elif "value_ints" in attrs:
        arr = np.array(list(attrs["value_ints"]), dtype=np.int64)
    else:
        raise RuntimeError("unsupported Constant attribute variant")
    return [Val(tf.constant(_as_tf_array(arr)), arr)]


SUPPORTED_ONNX_OPS = tuple(sorted(_OP_HANDLERS))


def _build_concrete_function(model: onnx.ModelProto, tf):
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}

    lowerer = _Lowerer(tf)
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        lowerer.bind(init.name, Val(tf.constant(_as_tf_array(arr)), arr))

    input_names = []
    specs = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        shape = _static_input_shape(inp)
        dtype = _onnx_elem_type_to_tf(inp.type.tensor_type.elem_type, tf)
        input_names.append(inp.name)
        specs.append(tf.TensorSpec(shape=shape, dtype=dtype))

    output_names = [o.name for o in graph.output]
    if not output_names:
        raise RuntimeError("model has no graph outputs to export")

    def forward(*args):
        for name, arg in zip(input_names, args):
            lowerer.bind(name, Val(arg))
        for node in graph.node:
            lowerer.lower_node(node)
        return [lowerer.get(name).t for name in output_names]

    # autograph=False: `forward`'s only control flow is a plain Python `for` loop
    # over the graph's (fixed, concrete) node list -- there is nothing for autograph
    # to rewrite -- and disabling it keeps exceptions raised while lowering a node
    # (e.g. an unsupported op) as plain RuntimeErrors instead of autograph wrapping
    # them in an "in user code" traceback.
    concrete = tf.function(forward, autograph=False).get_concrete_function(*specs)
    return concrete


def convert_to_tflite(
    model: onnx.ModelProto,
    *,
    optimizations: Optional[List[Any]] = None,
):
    """Convert an ONNX model to an in-memory TFLite flatbuffer (``bytes``).

    Parameters
    ----------
    model:
        The ONNX model to convert. Typically the output of :func:`onnxsim.simplify`.
        Every graph input dimension must be static; every node's op must be one of
        ``SUPPORTED_ONNX_OPS``.
    optimizations:
        Optional list forwarded to ``tf.lite.TFLiteConverter.optimizations``, e.g.
        ``["DEFAULT"]`` (string names of ``tf.lite.Optimize`` members are accepted, as
        well as the enum members themselves) to enable TFLite's post-training
        (dynamic-range) quantization.

    Returns
    -------
    bytes
        The serialized ``.tflite`` flatbuffer.

    Raises
    ------
    RuntimeError
        If TensorFlow is not installed, an input has a non-static dimension, the
        graph uses an ONNX op/feature this translator does not support, or
        conversion itself fails.
    """
    tf = _import_tensorflow()
    concrete = _build_concrete_function(model, tf)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    if optimizations:
        converter.optimizations = [
            getattr(tf.lite.Optimize, o) if isinstance(o, str) else o
            for o in optimizations
        ]
    try:
        return converter.convert()
    except Exception as exc:
        raise RuntimeError(f"TFLite conversion failed: {exc}") from exc


def export_tflite(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    **kwargs,
) -> bytes:
    """Convert ``model`` to TFLite, optionally saving it to ``output_path``.

    This is the public entry point used by the ``onnxsim --emit-tflite`` CLI and is
    re-exported as ``onnxsim.export_tflite``. It returns the serialized flatbuffer
    regardless of whether ``output_path`` is given.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    output_path:
        If given, the ``.tflite`` flatbuffer is written here. If ``None``, the model
        is only returned.

    Other keyword arguments are forwarded to :func:`convert_to_tflite`.

    Returns
    -------
    bytes
    """
    tflite_model = convert_to_tflite(model, **kwargs)
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    return tflite_model
