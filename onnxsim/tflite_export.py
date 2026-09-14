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
By default (``io_layout="nchw"``) this translator keeps the graph's public tensors
in ONNX's NCHW layout and transposes to/from NHWC only around the ops
(``Conv``/``MaxPool``/``AveragePool``) that need it; pass ``io_layout="nhwc"``
(CLI: ``--tflite-layout nhwc``) to carry 4-D tensors channel-last end to end
instead -- the public 4-D I/O changes dimension order, but no transposes are
emitted at all, which keeps large models compilable on the Edge TPU (its
compiler refuses the NCHW entry transpose above modest activation sizes).

A model whose op set goes beyond ``SUPPORTED_ONNX_OPS`` can instead be routed through
`onnx2tf <https://github.com/PINTO0309/onnx2tf>`_, an actively maintained third-party
project with far broader op coverage, via ``backend="onnx2tf"`` on
:func:`convert_to_tflite`/:func:`export_tflite` (CLI: ``--tflite-backend onnx2tf``).
See ``onnx2tf_export.py`` for what that trades off in return (a much heavier
dependency, and a default channel-last I/O layout unlike this module's NCHW-preserving
translator).
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

    ``.layout`` tracks the physical dimension order of 4-D activation tensors when
    the translator runs in NHWC mode (``io_layout="nhwc"``): ``"NCHW"`` for ONNX's
    own order, ``"NHWC"`` for the channel-last order TFLite kernels (and the Edge
    TPU) stream natively. It is always kept consistent with ``.t``/``.const``: any
    transpose applied to one is applied to all. Non-4-D tensors are layout-free
    (``None``).
    """

    __slots__ = ("t", "const", "layout")

    def __init__(
        self,
        t,
        const: Optional[np.ndarray] = None,
        layout: Optional[str] = None,
    ):
        self.t = t
        self.const = const
        self.layout = layout


# Permutation NCHW -> NHWC and back (4-D activation tensors only).
_NCHW_TO_NHWC_PERM = (0, 2, 3, 1)
_NHWC_TO_NCHW_PERM = (0, 3, 1, 2)
# NCHW axis -> NHWC axis: channel 1 -> 3, height 2 -> 1, width 3 -> 2.
_NCHW_TO_NHWC_AXIS = (0, 3, 1, 2)


def _transpose_const(const: Optional[np.ndarray], perm) -> Optional[np.ndarray]:
    if const is None:
        return None
    return np.transpose(np.asarray(const), perm)


class _Lowerer:
    """Walks an ONNX graph once, building the equivalent TensorFlow ops as it goes."""

    def __init__(self, tf, nhwc: bool = False):
        self.tf = tf
        # NHWC mode (``io_layout="nhwc"``): 4-D activation tensors are carried in
        # channel-last order end to end, so conv/pool need no transposes and the
        # Edge TPU compiler can tile the model. OFF by default: NCHW mode keeps
        # the graph's public tensors in ONNX's own order exactly as before.
        self.nhwc = nhwc
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

    @staticmethod
    def _rank(val: Val) -> int:
        return len(val.t.shape.as_list())

    def as_nhwc(self, val: Val) -> Val:
        """Return ``val`` physically NHWC (transposing a shared NCHW value makes a
        copy -- the original is never mutated, so other consumers are unaffected)."""
        if val is None or not self.nhwc:
            return val
        if self._rank(val) != 4 or val.layout != "NCHW":
            return val
        return Val(
            self.tf.transpose(val.t, _NCHW_TO_NHWC_PERM),
            _transpose_const(val.const, _NCHW_TO_NHWC_PERM),
            "NHWC",
        )

    def as_nchw(self, val: Val) -> Val:
        """Return ``val`` physically NCHW (copy on transpose, like :meth:`as_nhwc`)."""
        if val is None or not self.nhwc:
            return val
        if self._rank(val) != 4 or val.layout != "NHWC":
            return val
        return Val(
            self.tf.transpose(val.t, _NHWC_TO_NCHW_PERM),
            _transpose_const(val.const, _NHWC_TO_NCHW_PERM),
            "NCHW",
        )

    def to_public(self, val: Val) -> Val:
        """Convert a graph output to the requested public layout (``io_layout``)."""
        if val is None or not self.nhwc:
            return val
        if self._rank(val) == 4 and val.layout != "NHWC":
            return self.as_nhwc(val)
        return val

    def child_tag(self, data_val: Val, out_rank: int) -> Optional[str]:
        """Layout tag for the output of a rank-changing op (Squeeze/Unsqueeze/Gather).

        Rank-preserving ops inherit the (unified) input tag implicitly, but a
        new rank needs an explicit decision: NHWC lineage stays NHWC (so e.g. a
        5-D tensor unsqueezed from NHWC data squeezes back to NHWC), while a
        4-D tensor built from opaque data asserts NCHW -- matching NCHW mode,
        which feeds such tensors to convs as NCHW. Non-4-D outputs simply carry
        the lineage forward (inert until a 4-D tensor is built from them).
        """
        if not self.nhwc:
            return None
        data_rank = self._rank(data_val)
        if out_rank == 4 and data_rank != 4:
            return "NHWC" if data_val.layout == "NHWC" else "NCHW"
        return data_val.layout

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
                if self.nhwc and val.layout is None and self._rank(val) == 4:
                    # Handlers lowering straight to NHWC leave the tag unset;
                    # anything still NCHW-ordered at this point (Constant data,
                    # NCHW islands, ...) must have tagged itself explicitly.
                    val.layout = "NHWC"
                self.bind(out_name, val)


# ---------------------------------------------------------------------------
# Op handlers. Each handler is ``(lowerer, node, ins, attrs) -> list[Val]``, with
# ``ins[i]`` the already-converted ``Val`` for ``node.input[i]`` (``None`` for an
# omitted optional input) and ``attrs`` the node's parsed attribute dict.
#
# NHWC mode (``io_layout="nhwc"``) rules every handler follows:
# - Unify 4-D activation inputs with ``lowerer.as_nhwc`` first (a no-op in NCHW
#   mode, for non-4-D tensors, and for already-NHWC values). Never unify
#   non-activation 4-D tensors (notably Conv weights, which are OIHW).
# - Remap every NCHW-semantic axis/perm/pads through ``_remap_axis`` et al.
#   when the tensor rank is 4; other ranks are physically identical in both
#   modes and pass through untouched.
# - Reshape/Flatten (which couple values to physical order) and mixed-rank
#   broadcasting elementwise ops run in an NCHW island instead (see
#   ``_needs_nchw_island``); rank-preserving handlers inherit the unified layout.
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


def _remap_axis(axis: int, rank: int, nhwc: bool) -> int:
    """Map an ONNX (NCHW-semantic) axis onto the physical axis in NHWC mode.

    Returns ``axis`` untouched unless NHWC mode is active and the tensor rank
    is 4 (other ranks are carried identically in both modes); for 4-D tensors
    in NHWC mode, channel 1 -> 3, height 2 -> 1, width 3 -> 2.
    """
    if not nhwc or rank != 4:
        return axis
    return _NCHW_TO_NHWC_AXIS[axis % rank]


def _remap_perm(perm: Optional[List[int]], rank: int, nhwc: bool) -> List[int]:
    """Conjugate an ONNX transpose perm into NHWC physical axes.

    For physical ``p = T_P(t)`` and logical ``t2 = T_Q(t)``, the equivalent
    physical perm is ``R = P^-1 . Q . P``; ``perm=None`` (reverse all axes) is
    conjugated the same way rather than naively reversed. Returns the perm in
    the exact pre-existing form (including ``None`` handling) unless NHWC mode
    is active and the rank is 4.
    """
    if not nhwc or rank != 4:
        if perm is None:
            return list(reversed(range(rank)))
        return [int(p) for p in perm]
    q = (
        [int(p) % rank for p in perm]
        if perm is not None
        else list(reversed(range(rank)))
    )
    p = _NCHW_TO_NHWC_PERM
    inv = _NHWC_TO_NCHW_PERM
    return [inv[q[p[i]]] for i in range(rank)]


def _physical_axes(values, rank: int, nhwc: bool) -> List[int]:
    """Remap logical (ONNX) axis references to physical axes in NHWC mode.

    Applies to axis/axes inputs and attributes: initializers, graph inputs and
    Shape results all carry ONNX-logical positions, so 4-D tensors remap
    (channel 1 -> 3, height 2 -> 1, width 3 -> 2) while other ranks -- carried
    identically in both modes -- pass through.
    """
    values = [int(v) for v in values]
    if not nhwc or rank != 4:
        return values
    return [_NCHW_TO_NHWC_AXIS[v % rank] for v in values]


def _physical_order(values, rank: int, nhwc: bool) -> List[int]:
    """Reorder per-axis value lists (Tile repeats, Pad pairs) to physical order.

    Same contract as :func:`_physical_axes`: value lists always arrive in ONNX
    order, so 4-D tensors reorder in NHWC mode and everything else passes
    through. (Slice bounds/steps and Gather indices pair with their axes
    positionally rather than by layout, so they need no such handling.)
    """
    values = [int(v) for v in values]
    if not nhwc or rank != 4:
        return values
    if len(values) == 2 * rank:
        begins, ends = values[:rank], values[rank:]
        begins = [begins[i] for i in _NCHW_TO_NHWC_PERM]
        ends = [ends[i] for i in _NCHW_TO_NHWC_PERM]
        return begins + ends
    return [values[i] for i in _NCHW_TO_NHWC_PERM]


def _needs_nchw_island(ins: List[Val]) -> bool:
    """Whether mixed-rank broadcasting makes NHWC reinterpretation ambiguous.

    A scalar/vector combined positionally with a 4-D tensor means different
    math in NCHW vs NHWC order (e.g. a ``[C]`` vector broadcasts over W in
    NCHW but over C in NHWC), so such ops run in an NCHW island instead. Same
    ranks (or scalars, which broadcast identically) stay native.
    """
    ranks = set()
    for v in ins:
        if v is None:
            continue
        r = len(v.t.shape.as_list())
        if r != 0:
            ranks.add(r)
    return 4 in ranks and len(ranks) > 1


def _island_exit(lowerer, vals: List[Val]) -> List[Val]:
    """Move NCHW-island outputs back to NHWC (no-op unless NHWC mode)."""
    if not lowerer.nhwc:
        return vals
    outs = []
    for val in vals:
        if len(val.t.shape.as_list()) == 4:
            outs.append(
                Val(
                    lowerer.tf.transpose(val.t, _NCHW_TO_NHWC_PERM),
                    _transpose_const(val.const, _NCHW_TO_NHWC_PERM),
                    "NHWC",
                )
            )
        else:
            outs.append(val)
    return outs


def _simple_unary(dotted_tf_name: str):
    """``dotted_tf_name`` is a dotted attribute path off ``tf``, e.g. ``"nn.relu"``
    or ``"math.log"``, resolved lazily (only once TensorFlow is actually imported)."""

    def handler(lowerer, node, ins, attrs):
        fn = lowerer.tf
        for part in dotted_tf_name.split("."):
            fn = getattr(fn, part)
        x = lowerer.as_nhwc(ins[0])
        return [Val(fn(x.t))]

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
        if lowerer.nhwc and _needs_nchw_island(ins):
            ins = [lowerer.as_nchw(i) if i is not None else None for i in ins]
            island = True
        else:
            ins = [lowerer.as_nhwc(i) if i is not None else None for i in ins]
            island = False
        fn = getattr(tf, tf_name)
        t = ins[0].t
        for other in ins[1:]:
            t = fn(t, other.t)
        const = None
        if all(i.const is not None for i in ins):
            const = np.asarray(ins[0].const)
            for other in ins[1:]:
                const = np_fn(const, np.asarray(other.const))
        return _island_exit(lowerer, [Val(t, const)]) if island else [Val(t, const)]

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
    x = lowerer.as_nhwc(ins[0])
    return [Val(tf.nn.leaky_relu(x.t, alpha=alpha))]


@_register("PRelu")
def _op_prelu(lowerer, node, ins, attrs):
    # y = x if x >= 0 else slope * x, lowered as Relu/Minimum/Mul/Add so every
    # emitted op is Edge TPU compatible (TFLite's own PRELU op has stricter
    # layout requirements on the slope than ONNX does).
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    slope = lowerer.as_nhwc(ins[1])
    x_shape = x.t.shape.as_list()
    s = np.asarray(_require_const(slope, "PRelu's 'slope' input"))
    if lowerer.nhwc and len(x_shape) == 4 and s.ndim != 4:
        # The slope carries NCHW channel semantics ([C], [C,1,1], scalar, ...):
        # broadcast it against the logical shape, then carry the full array in
        # physical order.
        logical = [x_shape[i] for i in _NHWC_TO_NCHW_PERM]
        try:
            full = np.broadcast_to(s, logical)
        except ValueError:
            raise RuntimeError(
                f"PRelu slope shape {s.shape} is not broadcastable to input shape "
                f"{logical}."
            )
        slope_phys = np.transpose(full, _NCHW_TO_NHWC_PERM).copy()
        slope_full = tf.constant(slope_phys)
        slope_const = slope_phys
    else:
        try:
            np.broadcast_shapes(s.shape, tuple(x_shape))
        except ValueError:
            raise RuntimeError(
                f"PRelu slope shape {s.shape} is not broadcastable to input shape "
                f"{x_shape}."
            )
        # Broadcast explicitly: TF aligns trailing dimensions for implicit
        # broadcasting, which would misplace an ONNX [C]-shaped slope on an NCHW
        # input, so spell the full-shape slope out instead.
        slope_full = tf.broadcast_to(slope.t, x_shape)
        slope_const = s
    y = tf.nn.relu(x.t) + slope_full * tf.minimum(x.t, 0.0)
    const = None
    if x.const is not None:
        xc = np.asarray(x.const)
        const = np.maximum(xc, 0) + np.broadcast_to(slope_const, xc.shape) * np.minimum(
            xc, 0
        )
    return [Val(y, const)]


@_register("Dropout")
def _op_dropout(lowerer, node, ins, attrs):
    # Inference-mode Dropout is an identity (ONNX only scales/drops in
    # training mode), so it lowers to nothing -- this keeps models that still
    # carry Dropout nodes convertible.
    x = ins[0]
    training = bool(attrs.get("training_mode", 0))
    if len(ins) > 2 and ins[2] is not None:
        flag = np.asarray(_require_const(ins[2], "Dropout's 'training_mode' input"))
        training = training or bool(flag.reshape(-1)[0])
    if training:
        raise RuntimeError(
            "Dropout with training_mode=1 cannot be exported to TFLite "
            "(only inference-mode Dropout, which is an identity, is supported)."
        )
    x = lowerer.as_nhwc(ins[0])
    const = np.asarray(x.const) if x.const is not None else None
    outs = [Val(x.t, const)]
    if len(node.output) > 1:
        # Second output is the dropout mask, never consumed by inference
        # graphs; bind an all-ones mask so the name still resolves.
        tf = lowerer.tf
        mask_shape = x.t.shape.as_list()
        outs.append(
            Val(
                tf.ones(mask_shape, tf.bool),
                np.ones(mask_shape, dtype=np.bool_),
                x.layout,
            )
        )
    return outs


@_register("Gelu")
def _op_gelu(lowerer, node, ins, attrs):
    tf = lowerer.tf
    approximate = attrs.get("approximate", "none") == "tanh"
    x = lowerer.as_nhwc(ins[0])
    return [Val(tf.nn.gelu(x.t, approximate=approximate))]


@_register("Softmax")
def _op_softmax(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    rank = len(x.t.shape.as_list())
    axis = _remap_axis(int(attrs.get("axis", -1)), rank, lowerer.nhwc)
    return [Val(tf.nn.softmax(x.t, axis=axis))]


@_register("Clip")
def _op_clip(lowerer, node, ins, attrs):
    tf = lowerer.tf
    if lowerer.nhwc and _needs_nchw_island(ins):
        ins = [lowerer.as_nchw(i) if i is not None else None for i in ins]
        island = True
    else:
        ins = [lowerer.as_nhwc(i) if i is not None else None for i in ins]
        island = False
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
    return _island_exit(lowerer, [Val(t)]) if island else [Val(t)]


@_register("Cast")
def _op_cast(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
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
    if lowerer.nhwc and (
        len(a.t.shape.as_list()) == 4 or len(b.t.shape.as_list()) == 4
    ):
        # 4-D batch matmul contracts the last two physical axes, so it runs in
        # an NCHW island (rare in CNN backbones; the head is 2-D).
        a = lowerer.as_nchw(a)
        b = lowerer.as_nchw(b)
        return _island_exit(lowerer, [Val(tf.matmul(a.t, b.t))])
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
    # NB: only the activation is unified -- the weight stays OIHW.
    x = lowerer.as_nhwc(ins[0])
    w = ins[1]
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
    # In NHWC mode the unified input is already channel-last, so the entry and
    # exit transposes below are skipped and the channel/spatial dims are read
    # from their NHWC positions (this is what keeps large models compilable on
    # the Edge TPU -- see io_layout="nhwc").
    native_nhwc = lowerer.nhwc and len(x_shape) == 4 and x.layout == "NHWC"
    in_c = x_shape[3] if native_nhwc else x_shape[1]
    spatial = x_shape[1:3] if native_nhwc else x_shape[2:4]

    pads = _compute_spatial_pad(attrs, spatial, kernel_shape, strides, dilations)
    filt = tf.transpose(w.t, [2, 3, 1, 0])  # OIHW -> HWIO
    x_nhwc = x.t if native_nhwc else tf.transpose(x.t, [0, 2, 3, 1])
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
    if native_nhwc:
        return [Val(y)]
    y = tf.transpose(y, [0, 3, 1, 2])
    return [Val(y)]


def _pool_2d(reduce_kind: str):
    def handler(lowerer, node, ins, attrs):
        tf = lowerer.tf
        x = lowerer.as_nhwc(ins[0])
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
        native_nhwc = lowerer.nhwc and len(x_shape) == 4 and x.layout == "NHWC"
        in_hw = x_shape[1:3] if native_nhwc else x_shape[2:4]
        pads = _compute_spatial_pad(attrs, in_hw, kernel_shape, strides, dilations)
        pad_needed = any(p != (0, 0) for p in pads)
        x_nhwc = x.t if native_nhwc else tf.transpose(x.t, [0, 2, 3, 1])

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
                # Tile the divisor to the full output shape instead of relying on
                # broadcasting over the channel axis: the Edge TPU compiler
                # rejects that broadcast ("non-broadcastable operands") and
                # refuses the whole model, while a full-shape constant compiles
                # with every op mapped to the Edge TPU (verified against
                # edgetpu_compiler; numerics are unchanged).
                channels = x_nhwc.shape.as_list()[3]
                divisor = np.broadcast_to(
                    divisor, (1, divisor.shape[1], divisor.shape[2], channels)
                ).copy()
                y = sum_pool / tf.constant(divisor)
            else:
                y = sum_pool / float(window_area)
        if native_nhwc:
            return [Val(y)]
        return [Val(tf.transpose(y, [0, 3, 1, 2]))]

    return handler


_OP_HANDLERS["MaxPool"] = _pool_2d("max")
_OP_HANDLERS["AveragePool"] = _pool_2d("avg")


@_register("GlobalAveragePool")
def _op_global_avg_pool(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    rank = len(x.t.shape.as_list())
    axes = [1, 2] if (lowerer.nhwc and rank == 4) else [2, 3]
    return [Val(tf.reduce_mean(x.t, axis=axes, keepdims=True))]


@_register("GlobalMaxPool")
def _op_global_max_pool(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    rank = len(x.t.shape.as_list())
    axes = [1, 2] if (lowerer.nhwc and rank == 4) else [2, 3]
    return [Val(tf.reduce_max(x.t, axis=axes, keepdims=True))]


def _reduce(tf_name: str):
    def handler(lowerer, node, ins, attrs):
        tf = lowerer.tf
        x = lowerer.as_nhwc(ins[0])
        x_shape = x.t.shape.as_list()
        rank = len(x_shape)
        if len(ins) > 1 and ins[1] is not None:
            axes = [int(a) for a in _require_const(ins[1], f"{node.op_type}'s 'axes'")]
        elif "axes" in attrs:
            axes = [int(a) for a in attrs["axes"]]
        else:
            axes = list(range(rank))
        axes = sorted(a % rank for a in axes)
        if lowerer.nhwc and rank == 4:
            axes = sorted(_remap_axis(a, rank, lowerer.nhwc) for a in axes)
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
    x = lowerer.as_nhwc(ins[0])
    scale, bias, mean, var = ins[1], ins[2], ins[3], ins[4]
    eps = float(attrs.get("epsilon", 1e-5))
    c = scale.t.shape.as_list()[0]
    if lowerer.nhwc and len(x.t.shape.as_list()) == 4:
        shape = [1, 1, 1, c]
    else:
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
    if lowerer.nhwc:
        # Reshape couples values to physical order, so it runs in an NCHW
        # island: transpose 4-D inputs in, run the existing logic, transpose
        # 4-D outputs back out. A 4-D target coming from a Shape op in NHWC
        # territory is permuted back to NCHW order first.
        data = lowerer.as_nchw(ins[0])
        ins = [data, ins[1]]
        island = True
    else:
        island = False
    x = ins[0]
    target = [int(v) for v in _require_const(ins[1], "Reshape's 'shape' input")]
    allowzero = int(attrs.get("allowzero", 0))
    x_shape = x.t.shape.as_list()
    resolved = [
        x_shape[i] if d == 0 and not allowzero else d for i, d in enumerate(target)
    ]
    t = tf.reshape(x.t, resolved)
    const = np.reshape(np.asarray(x.const), resolved) if x.const is not None else None
    return _island_exit(lowerer, [Val(t, const)]) if island else [Val(t, const)]


@_register("Flatten")
def _op_flatten(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nchw(ins[0]) if lowerer.nhwc else ins[0]
    shape = x.t.shape.as_list()
    axis = int(attrs.get("axis", 1)) % (len(shape) + 1)
    outer = int(np.prod(shape[:axis], dtype=np.int64))
    inner = int(np.prod(shape[axis:], dtype=np.int64))
    t = tf.reshape(x.t, [outer, inner])
    const = None
    if lowerer.nhwc and x.const is not None:
        const = np.asarray(x.const).reshape([outer, inner])
    island = lowerer.nhwc and len(shape) == 4
    return _island_exit(lowerer, [Val(t, const)]) if island else [Val(t, const)]


@_register("Squeeze")
def _op_squeeze(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    x_shape = x.t.shape.as_list()
    rank = len(x_shape)
    if len(ins) > 1 and ins[1] is not None:
        raw = [int(a) % rank for a in _require_const(ins[1], "Squeeze's 'axes' input")]
        axes = set(_physical_axes(raw, rank, lowerer.nhwc))
    elif "axes" in attrs:
        raw = [int(a) % rank for a in attrs["axes"]]
        axes = set(_physical_axes(raw, rank, lowerer.nhwc))
    else:
        # Axes derived from the physical shape itself are already physical.
        axes = {i for i, d in enumerate(x_shape) if d == 1}
    new_shape = [d for i, d in enumerate(x_shape) if i not in axes]
    t = tf.reshape(x.t, new_shape)
    const = np.reshape(np.asarray(x.const), new_shape) if x.const is not None else None
    return [Val(t, const, lowerer.child_tag(x, len(new_shape)))]


@_register("Unsqueeze")
def _op_unsqueeze(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    x_shape = x.t.shape.as_list()
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(a) for a in _require_const(ins[1], "Unsqueeze's 'axes' input")]
    else:
        axes = [int(a) for a in attrs["axes"]]
    # NB: axes index positions of the output built from a non-4-D tensor, so
    # they are already physical -- unlike Squeeze, no remap applies here.
    out_rank = len(x_shape) + len(axes)
    axes = sorted(a % out_rank for a in axes)
    new_shape = list(x_shape)
    for a in axes:
        new_shape.insert(a, 1)
    t = tf.reshape(x.t, new_shape)
    const = np.reshape(np.asarray(x.const), new_shape) if x.const is not None else None
    return [Val(t, const, lowerer.child_tag(x, len(new_shape)))]


@_register("Transpose")
def _op_transpose(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    rank = len(x.t.shape.as_list())
    perm = _remap_perm(attrs.get("perm"), rank, lowerer.nhwc)
    return [Val(tf.transpose(x.t, perm))]


@_register("Concat")
def _op_concat(lowerer, node, ins, attrs):
    tf = lowerer.tf
    unified = [lowerer.as_nhwc(i) for i in ins]
    rank = len(unified[0].t.shape.as_list())
    axis = _remap_axis(int(attrs["axis"]), rank, lowerer.nhwc)
    t = tf.concat([i.t for i in unified], axis=axis)
    const = None
    if all(i.const is not None for i in unified):
        const = np.concatenate([np.asarray(i.const) for i in unified], axis=axis)
    return [Val(t, const)]


@_register("Split")
def _op_split(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    x_shape = x.t.shape.as_list()
    rank = len(x_shape)
    axis = int(attrs.get("axis", 0)) % rank
    if lowerer.nhwc and rank == 4:
        axis = _remap_axis(axis, rank, lowerer.nhwc)
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
    x = lowerer.as_nhwc(ins[0])
    idx = ins[1]
    x_shape = x.t.shape.as_list()
    rank = len(x_shape)
    axis = int(attrs.get("axis", 0)) % rank
    if lowerer.nhwc and rank == 4:
        axis = _remap_axis(axis, rank, lowerer.nhwc)
    if idx.const is not None:
        idx_arr = np.asarray(idx.const)
        dim = x_shape[axis]
        idx_arr = np.where(idx_arr < 0, idx_arr + dim, idx_arr)
        idx_t = tf.constant(idx_arr.astype(np.int32))
    else:
        idx_t = idx.t
    t = tf.gather(x.t, idx_t, axis=axis)
    const = None
    if x.const is not None and idx.const is not None:
        const = np.take(np.asarray(x.const), np.asarray(idx.const), axis=axis)
    return [Val(t, const, lowerer.child_tag(x, len(t.shape.as_list())))]


@_register("Tile")
def _op_tile(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    reps = ins[1]
    rank = len(x.t.shape.as_list())
    repeats = _physical_order(
        [int(r) for r in _require_const(reps, "Tile's 'repeats' input")],
        rank,
        lowerer.nhwc,
    )
    return [Val(tf.tile(x.t, repeats))]


@_register("Pad")
def _op_pad(lowerer, node, ins, attrs):
    tf = lowerer.tf
    x = lowerer.as_nhwc(ins[0])
    rank = len(x.t.shape.as_list())
    if len(ins) > 1 and ins[1] is not None:
        pads_flat = _physical_order(
            [int(v) for v in _require_const(ins[1], "Pad's 'pads' input")],
            rank,
            lowerer.nhwc,
        )
    else:
        pads_flat = _physical_order([int(v) for v in attrs["pads"]], rank, lowerer.nhwc)
    mode = attrs.get("mode", "constant")
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
    x = lowerer.as_nhwc(ins[0])
    x_shape = x.t.shape.as_list()
    rank = len(x_shape)
    if len(ins) > 1:
        starts = [int(v) for v in _require_const(ins[1], "Slice's 'starts' input")]
        ends = [int(v) for v in _require_const(ins[2], "Slice's 'ends' input")]
        if len(ins) > 3 and ins[3] is not None:
            axes_in = [int(v) for v in _require_const(ins[3], "Slice's 'axes' input")]
            axes = _physical_axes(axes_in, rank, lowerer.nhwc)
        else:
            axes = _physical_axes(list(range(len(starts))), rank, lowerer.nhwc)
        steps = (
            [int(v) for v in _require_const(ins[4], "Slice's 'steps' input")]
            if len(ins) > 4 and ins[4] is not None
            else [1] * len(starts)
        )
    else:
        starts = [int(v) for v in attrs["starts"]]
        ends = [int(v) for v in attrs["ends"]]
        axes = _physical_axes(
            [int(v) for v in attrs.get("axes", list(range(len(starts))))],
            rank,
            lowerer.nhwc,
        )
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
    x = lowerer.as_nhwc(ins[0])
    # Report ONNX-logical (NCHW) dims, not physical ones: downstream index
    # arithmetic (Gather indices, ...) is authored against ONNX order, so the
    # values must match what onnx.shape_inference would say. (The traced
    # tensor therefore differs from a raw tf.shape in NHWC mode -- by design.)
    logical = (
        [x.t.shape.as_list()[i] for i in _NHWC_TO_NCHW_PERM]
        if (lowerer.nhwc and len(x.t.shape.as_list()) == 4)
        else list(x.t.shape.as_list())
    )
    shape = np.array(logical, dtype=np.int64)
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
    out = Val(tf.constant(_as_tf_array(arr)), arr)
    if lowerer.nhwc and arr.ndim == 4:
        # 4-D constant data carries ONNX (NCHW) order; unify() transposes it on
        # demand alongside the traced tensor.
        out.layout = "NCHW"
    return [out]


SUPPORTED_ONNX_OPS = tuple(sorted(_OP_HANDLERS))


def _onnx_elem_type_to_np(elem_type: int) -> Any:
    TP = onnx.TensorProto
    mapping = {
        TP.FLOAT: np.float32,
        TP.FLOAT16: np.float16,
        TP.DOUBLE: np.float64,
        TP.INT32: np.int32,
        TP.INT64: np.int64,
        TP.BOOL: np.bool_,
    }
    if elem_type not in mapping:
        raise RuntimeError(
            f"Unsupported input dtype {TP.DataType.Name(elem_type)} for TFLite "
            "export calibration data."
        )
    return mapping[elem_type]


def _public_shape(shape: List[int], io_layout: str) -> List[int]:
    """Permute a 4-D static shape to the requested public layout (else unchanged)."""
    if io_layout == "nhwc" and len(shape) == 4:
        return [shape[i] for i in _NCHW_TO_NHWC_PERM]
    return shape


def random_representative_dataset(
    model: onnx.ModelProto,
    num_samples: int = 100,
    seed: int = 0,
    io_layout: str = "nchw",
):
    """Build a ``representative_dataset`` callable for full-integer quantization.

    Generates ``num_samples`` calibration samples of uniform-random data
    (float inputs in ``[0, 1)``, small ints/bools for the other dtypes) with
    the model's own static input shapes. This is enough to let the converter
    measure activation ranges and produce a valid quantized model -- and, with
    ``inference_io_dtype=``, an Edge TPU-compilable one -- but random data
    does not match any real data distribution, so for production accuracy
    prefer a callable yielding batches of real, representative inputs (see
    :func:`convert_to_tflite`'s ``representative_dataset`` parameter).

    ``io_layout`` must match :func:`convert_to_tflite`'s: with ``"nhwc"`` the
    generated batches carry 4-D inputs in channel-last order.
    """
    initializer_names = {t.name for t in model.graph.initializer}
    specs = []
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        specs.append(
            (
                _public_shape(_static_input_shape(inp), _validate_io_layout(io_layout)),
                _onnx_elem_type_to_np(inp.type.tensor_type.elem_type),
            )
        )
    if not specs:
        raise RuntimeError(
            "cannot build calibration data: the model has no (non-initializer) inputs"
        )
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    def _sample(rng: np.random.Generator, shape: List[int], dtype: Any) -> np.ndarray:
        if np.dtype(dtype) == np.dtype(np.bool_):
            return rng.integers(0, 2, size=shape).astype(np.bool_)
        if np.issubdtype(np.dtype(dtype), np.integer):
            return rng.integers(-5, 6, size=shape).astype(dtype)
        return rng.random(size=shape).astype(dtype)

    def gen():
        rng = np.random.default_rng(seed)
        for _ in range(num_samples):
            yield [_sample(rng, shape, dtype) for shape, dtype in specs]

    return gen


def _validate_io_layout(io_layout: str) -> str:
    if io_layout not in ("nchw", "nhwc"):
        raise ValueError(f"io_layout must be 'nchw' or 'nhwc', got {io_layout!r}")
    return io_layout


def _resolve_inference_dtype(inference_io_dtype: Any, tf: Any) -> Any:
    if isinstance(inference_io_dtype, str):
        mapping = {"uint8": tf.uint8, "int8": tf.int8}
        if inference_io_dtype not in mapping:
            raise ValueError(
                f"inference_io_dtype must be 'uint8' or 'int8', got "
                f"{inference_io_dtype!r}"
            )
        return mapping[inference_io_dtype]
    return inference_io_dtype


def _build_concrete_function(model: onnx.ModelProto, tf, io_layout: str = "nchw"):
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    nhwc = _validate_io_layout(io_layout) == "nhwc"

    lowerer = _Lowerer(tf, nhwc=nhwc)
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        val = Val(tf.constant(_as_tf_array(arr)), arr)
        if nhwc and arr.ndim == 4:
            # 4-D initializer data carries ONNX (NCHW) order; unify() moves it
            # to NHWC on demand (Conv weights are OIHW, never unified, so the
            # tag is inert for them).
            val.layout = "NCHW"
        lowerer.bind(init.name, val)

    input_names = []
    specs = []
    layouts = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        shape = _public_shape(_static_input_shape(inp), io_layout)
        dtype = _onnx_elem_type_to_tf(inp.type.tensor_type.elem_type, tf)
        input_names.append(inp.name)
        specs.append(tf.TensorSpec(shape=shape, dtype=dtype))
        layouts.append("NHWC" if (nhwc and len(shape) == 4) else None)

    output_names = [o.name for o in graph.output]
    if not output_names:
        raise RuntimeError("model has no graph outputs to export")

    def forward(*args):
        for name, arg, layout in zip(input_names, args, layouts):
            lowerer.bind(name, Val(arg, None, layout))
        for node in graph.node:
            lowerer.lower_node(node)
        return [lowerer.to_public(lowerer.get(name)).t for name in output_names]

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
    backend: str = "builtin",
    optimizations: Optional[List[Any]] = None,
    int8_quantize: bool = False,
    representative_dataset: Any = None,
    num_calibration_samples: int = 100,
    inference_io_dtype: Any = None,
    io_layout: str = "nchw",
    **backend_kwargs: Any,
):
    """Convert an ONNX model to an in-memory TFLite flatbuffer (``bytes``).

    Parameters
    ----------
    model:
        The ONNX model to convert. Typically the output of :func:`onnxsim.simplify`.
    backend:
        Which ONNX-to-TensorFlow translator to use: ``"builtin"`` (default, this
        module's own hand-written translator -- every graph input dimension must
        be static and every node's op must be one of ``SUPPORTED_ONNX_OPS``) or
        ``"onnx2tf"`` (delegates to `onnx2tf <https://github.com/PINTO0309/onnx2tf>`_,
        which covers far more ops at the cost of a much heavier dependency and
        changing the model's public input/output tensor layout to channel-last by
        default -- see ``onnxsim/onnx2tf_export.py``). Reach for ``"onnx2tf"`` when
        a model hits an unsupported op with the builtin translator.
    optimizations:
        ``backend="builtin"`` only. Optional list forwarded to
        ``tf.lite.TFLiteConverter.optimizations``, e.g. ``["DEFAULT"]`` (string
        names of ``tf.lite.Optimize`` members are accepted, as well as the enum
        members themselves) to enable TFLite's post-training (dynamic-range)
        quantization. Mutually exclusive with ``int8_quantize``.
    int8_quantize:
        ``backend="builtin"`` only. When true, run full-integer post-training
        quantization instead: ``target_spec.supported_ops`` is pinned to
        ``TFLITE_BUILTINS_INT8`` so conversion fails loudly on any op without
        an integer kernel rather than silently leaving it in float. This is the
        quantization the Coral Edge TPU requires -- combine it with
        ``inference_io_dtype="uint8"`` (or ``"int8"``) for fully-quantized I/O
        and compile the result with :func:`onnxsim.edgetpu_export.compile_for_edgetpu`
        (or :func:`onnxsim.export_edgetpu` for the one-shot path).
    representative_dataset:
        ``backend="builtin"`` only, requires ``int8_quantize=True``. A callable
        with no arguments yielding calibration batches (each a list of NumPy
        arrays in graph-input order, following the
        ``tf.lite.TFLiteConverter.representative_dataset`` protocol), e.g. built
        from real inputs. When ``None`` (the default),
        :func:`random_representative_dataset` generates ``num_calibration_samples``
        uniform-random batches from the model's own input shapes -- enough to
        produce a valid quantized model, but random data cannot match a real
        data distribution, so prefer real inputs for production accuracy.
    num_calibration_samples:
        How many random batches :func:`random_representative_dataset` generates
        when ``representative_dataset`` is not given. Ignored otherwise.
    inference_io_dtype:
        ``backend="builtin"`` only, requires ``int8_quantize=True``. ``"uint8"``
        or ``"int8"`` (or the corresponding ``tf.dtypes.DType``), forwarded to
        the converter's ``inference_input_type``/``inference_output_type`` so the
        model's public I/O is quantized too. The Edge TPU runs fastest -- and
        avoids a CPU-side quantize/dequantize pair at each boundary -- with
        quantized I/O.
    io_layout:
        ``backend="builtin"`` only. ``"nchw"`` (default) keeps the graph's
        public tensors in ONNX's NCHW order, transposing to NHWC only around
        the conv/pool ops that need it. ``"nhwc"`` instead carries 4-D tensors
        channel-last end to end -- the public 4-D I/O changes dimension order
        to NHWC, but conv/pool/concat emit no transposes at all. Prefer
        ``"nhwc"`` for Edge TPU deployment: this investigation measured the
        NCHW entry transpose refusing compilation (``large activation
        tensors``) from ~64K activation elements up (64ch x 32x32 fails, the
        identical NHWC graph maps fully), while the exit transpose is harmless.
        A passed ``representative_dataset`` must then yield NHWC-ordered
        batches.
    **backend_kwargs:
        ``backend="onnx2tf"`` only. Forwarded to
        :func:`onnxsim.onnx2tf_export.convert_to_tflite_via_onnx2tf` (and from there
        to ``onnx2tf.convert()``); use onnx2tf's own quantization options there.

    Returns
    -------
    bytes
        The serialized ``.tflite`` flatbuffer.

    Raises
    ------
    RuntimeError
        If the selected backend's dependency is not installed, or conversion
        fails -- for ``"builtin"``, an input has a non-static dimension or the
        graph uses an ONNX op/feature the translator does not support.
    """
    _validate_io_layout(io_layout)
    if backend == "onnx2tf":
        from onnxsim import onnx2tf_export

        if (
            int8_quantize
            or representative_dataset is not None
            or inference_io_dtype is not None
            or io_layout != "nchw"
        ):
            raise TypeError(
                "convert_to_tflite() with backend='onnx2tf' does not accept "
                "int8_quantize/representative_dataset/inference_io_dtype/io_layout; "
                "use onnx2tf's own quantization options (forwarded as extra keyword "
                "arguments) instead (onnx2tf is channel-last by default)."
            )
        return onnx2tf_export.convert_to_tflite_via_onnx2tf(model, **backend_kwargs)
    if backend != "builtin":
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'builtin' or 'onnx2tf'"
        )
    if backend_kwargs:
        raise TypeError(
            f"convert_to_tflite() with backend='builtin' got unexpected keyword "
            f"arguments: {sorted(backend_kwargs)}"
        )
    if optimizations and int8_quantize:
        raise ValueError(
            "optimizations= and int8_quantize=True are mutually exclusive: "
            "int8_quantize already enables the DEFAULT optimization internally "
            "as part of full-integer quantization."
        )
    if representative_dataset is not None and not int8_quantize:
        raise ValueError("representative_dataset= requires int8_quantize=True.")
    if inference_io_dtype is not None and not int8_quantize:
        raise ValueError("inference_io_dtype= requires int8_quantize=True.")

    tf = _import_tensorflow()
    concrete = _build_concrete_function(model, tf, io_layout=io_layout)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    if int8_quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = (
            representative_dataset
            if representative_dataset is not None
            else random_representative_dataset(
                model, num_calibration_samples, io_layout=io_layout
            )
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if inference_io_dtype is not None:
            io_dtype = _resolve_inference_dtype(inference_io_dtype, tf)
            converter.inference_input_type = io_dtype
            converter.inference_output_type = io_dtype
    elif optimizations:
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

    Other keyword arguments (including ``backend``) are forwarded to
    :func:`convert_to_tflite`.

    Returns
    -------
    bytes
    """
    tflite_model = convert_to_tflite(model, **kwargs)
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    return tflite_model
