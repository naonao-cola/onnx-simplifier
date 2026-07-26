import copy
import warnings
from collections import defaultdict
from typing import Callable, Any, Iterable, List, Optional, Tuple, Dict, Union

import onnx
from onnx import defs, helper, shape_inference
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from rich.table import Table
from rich.text import Text
from rich import print

try:
    import sympy
except ImportError:  # sympy is an optional dependency; see _tensor_shape below.
    sympy = None

try:
    from onnx import inliner as onnx_inliner
except ImportError:  # onnx.inliner was added in onnx 1.14; see ModelInfo.__init__.
    onnx_inliner = None


__all__ = ['ModelInfo', 'print_simplifying_info']


def _iter_graph_tensors(graph: onnx.GraphProto) -> Iterable[onnx.TensorProto]:
    """Yield every ``TensorProto`` stored in ``graph``, recursing into subgraphs.

    Covers initializers as well as tensors carried in node attributes (e.g. the
    ``value`` of a ``Constant``), matching every place a model may hold tensor
    data.
    """
    for initializer in graph.initializer:
        yield initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            for tensor in attr.tensors:
                yield tensor
            if attr.HasField("g"):
                yield from _iter_graph_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_graph_tensors(subgraph)


def _external_data_size(graph: onnx.GraphProto) -> int:
    """Total bytes of tensor data held in external files, from metadata alone.

    ``ExternalDataInfo(tensor).length`` reads the ``length`` entry of a tensor's
    ``external_data`` record, so this never loads the data itself: the size of a
    model whose weights live on disk can be reported without materializing them.
    """
    total = 0
    for tensor in _iter_graph_tensors(graph):
        if uses_external_data(tensor):
            total += ExternalDataInfo(tensor).length or 0
    return total


def human_readable_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def human_readable_num(num, suffix=""):
    # A symbolic MAC count (dynamic shapes + sympy) is printed as the formula
    # itself, e.g. "512*batch*seq**2 + 5419008*batch".
    if _is_symbolic(num):
        return str(sympy.factor(num))
    for unit in ["", "K", "M", "G", "T", "P", "E"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Z{suffix}"


# A dimension is a concrete int, a symbolic size (a sympy Symbol standing in for
# a ``dim_param`` such as "batch"), or None when the size is entirely unknown.
Dim = Union[int, "sympy.Expr", None]
# A MAC count is an int, or a sympy expression once any symbolic dim is involved.
Macs = Union[int, "sympy.Expr"]
ShapeMap = Dict[str, List[Dim]]


def _dim_symbol(name: str) -> "sympy.Expr":
    # Same dim_param name -> same symbol, so a dynamic dim shared across tensors
    # (e.g. "batch") combines correctly in the accumulated formula. Sizes are
    # positive integers, which lets sympy simplify/factor the result.
    return sympy.Symbol(name, positive=True, integer=True)


def _prod(vals: List[Dim]) -> Macs:
    result = 1
    for v in vals:
        result *= v
    return result


def _tensor_shape(type_proto: onnx.TypeProto) -> Optional[List[Dim]]:
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: List[Dim] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.dim_param:
            # Dynamic (symbolic) dimension. With sympy we keep it as a symbol so
            # the MAC total becomes a formula in terms of it; without sympy we
            # assume 1 and report per-sample MACs (as onnx-tool does).
            shape.append(_dim_symbol(dim.dim_param) if sympy is not None else 1)
        else:
            # Rank is known but this dimension is entirely unknown (no value and
            # no name); it stays None and disables MAC counting for the node.
            shape.append(None)
    return shape


def _is_symbolic(value: Macs) -> bool:
    return sympy is not None and isinstance(value, sympy.Expr) and bool(value.free_symbols)


def _representative_number(value: Macs) -> int:
    # Collapse a (possibly symbolic) MAC count to a single number by setting
    # every free dimension to 1. Used only for ordering and the summary table's
    # highlighting -- never for the reported value, which stays symbolic.
    if sympy is not None and isinstance(value, sympy.Expr):
        value = value.subs({s: 1 for s in value.free_symbols})
    return int(value)


def _collect_shapes(graph: onnx.GraphProto, inherited: ShapeMap) -> ShapeMap:
    shapes: ShapeMap = dict(inherited)
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = _tensor_shape(value_info.type)
        if shape is not None:
            shapes[value_info.name] = shape
    for initializer in graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _known(shape: Optional[List[Optional[int]]]) -> bool:
    return bool(shape) and all(d is not None for d in shape)


# --- Per-op MAC (multiply-accumulate) counters --------------------------------
# Each counter returns the number of MACs for a single node, or 0 when the
# required tensor shapes are unknown. FLOPs are reported as 2 * MACs.
# Coverage is intentionally limited to the compute-dominant operators; these
# account for the vast majority of a typical model's arithmetic.
_MAC_COUNTERS: Dict[str, Callable[[onnx.NodeProto, ShapeMap], Macs]] = {}


def _register(*op_types: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        for op_type in op_types:
            _MAC_COUNTERS[op_type] = fn
        return fn
    return deco


# QLinearConv reorders inputs: x, x_scale, x_zp, w, w_scale, w_zp, ... so the
# weight is input[3] rather than input[1]. Everything else about the MAC count
# (a quantized conv does the same multiply-accumulates as its float twin) is
# identical, so the counters below just point at the right operand indices.
@_register("Conv", "ConvInteger")
def _conv_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    return _conv_macs_impl(node, shapes, weight_idx=1)


@_register("QLinearConv")
def _qlinearconv_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    return _conv_macs_impl(node, shapes, weight_idx=3)


def _conv_macs_impl(node: onnx.NodeProto, shapes: ShapeMap, weight_idx: int) -> Macs:
    # weight: [out_channels, in_channels / group, *kernel_shape]
    # output: [batch, out_channels, *spatial_out]
    weight = shapes.get(node.input[weight_idx])
    output = shapes.get(node.output[0])
    if not _known(weight) or not _known(output):
        return 0
    in_channels_per_group = weight[1]
    kernel = weight[2:]
    return _prod(output) * in_channels_per_group * _prod(kernel)


@_register("ConvTranspose")
def _conv_transpose_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    # weight: [in_channels, out_channels / group, *kernel_shape]
    # input:  [batch, in_channels, *spatial_in]
    x = shapes.get(node.input[0])
    weight = shapes.get(node.input[1])
    if not _known(x) or not _known(weight):
        return 0
    out_channels_per_group = weight[1]
    kernel = weight[2:]
    return _prod(x) * out_channels_per_group * _prod(kernel)


@_register("Gemm")
def _gemm_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    a = shapes.get(node.input[0])
    b = shapes.get(node.input[1])
    if not _known(a) or not _known(b) or len(a) != 2 or len(b) != 2:
        return 0
    trans_a = _attr_int(node, "transA", 0)
    trans_b = _attr_int(node, "transB", 0)
    m, k = (a[1], a[0]) if trans_a else (a[0], a[1])
    n = b[0] if trans_b else b[1]
    return m * n * k


# MatMul, MatMulInteger and QLinearMatMul all take A as input[0] and produce Y
# as output[0]; the quantized variants only add scale/zero-point operands, which
# don't affect the multiply-accumulate count.
@_register("MatMul", "MatMulInteger", "QLinearMatMul")
def _matmul_macs(node: onnx.NodeProto, shapes: ShapeMap) -> Macs:
    # output: [*batch, M, N]; contraction dim K is the last dim of input A.
    a = shapes.get(node.input[0])
    output = shapes.get(node.output[0])
    if not _known(a) or not _known(output):
        return 0
    k = a[-1]
    return _prod(output) * k


@_register("Attention")
def _attention_macs(node: onnx.NodeProto, shapes: ShapeMap) -> Macs:
    """Scaled dot-product attention (ai.onnx opset 23+).

    Cost is dominated by the two batched matmuls, QK^T and (softmax)V; the
    softmax/scale/mask are elementwise and negligible by comparison. For heads
    ``h``, query/key sequence lengths ``sq``/``skv`` and per-head sizes ``d``
    (QK) and ``dv`` (V):

        QK^T : batch * h * sq * skv * d
        P V  : batch * h * sq * skv * dv

    Q/K/V are either 4D ``(batch, num_heads, seq, head_size)`` or 3D
    ``(batch, seq, num_heads * head_size)``; in the 3D form the head count comes
    from the ``q_num_heads`` / ``kv_num_heads`` attributes. Grouped-query
    attention (kv heads < q heads) still evaluates all ``q_num_heads`` query
    heads, so the query head count drives both matmuls.
    """
    q = shapes.get(node.input[0])
    k = shapes.get(node.input[1])
    v = shapes.get(node.input[2])
    if not _known(q) or not _known(k) or not _known(v):
        return 0

    if len(q) == 4 and len(k) == 4 and len(v) == 4:
        batch, q_heads, sq, d = q
        skv = k[2]
        dv = v[3]
    elif len(q) == 3 and len(k) == 3 and len(v) == 3:
        q_heads = _attr_int(node, "q_num_heads", 0)
        kv_heads = _attr_int(node, "kv_num_heads", 0)
        if q_heads <= 0 or kv_heads <= 0:
            return 0  # head split is unknowable without the attributes
        batch, sq, _ = q
        skv = k[1]
        d = k[2] // kv_heads   # per-head size (K packs kv_heads * head_size)
        dv = v[2] // kv_heads
    else:
        return 0

    qk = _prod([batch, q_heads, sq, skv, d])
    pv = _prod([batch, q_heads, sq, skv, dv])
    return qk + pv


def _type_map(graph: onnx.GraphProto) -> Dict[str, onnx.TypeProto]:
    types: Dict[str, onnx.TypeProto] = {}
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.type.ByteSize():
            types[value_info.name] = value_info.type
    for initializer in graph.initializer:
        types[initializer.name] = helper.make_tensor_type_proto(
            initializer.data_type, list(initializer.dims))
    return types


def _schema_function_body(
    node: onnx.NodeProto, types: Dict[str, onnx.TypeProto], opsets: Dict[str, int]
) -> Optional[onnx.FunctionProto]:
    """The expanded body of a context-dependent schema function, or None.

    Only context-dependent functions (e.g. Attention, LayerNormalization) are
    handled: their body is generated from the node's attributes and input types,
    so it needs no attribute plumbing when turned into a local function.
    """
    version = opsets.get(node.domain, opsets.get("", 1))
    try:
        schema = defs.get_schema(node.op_type, version, node.domain)
    except Exception:
        return None
    if not schema.has_context_dependent_function:
        return None
    # Every non-optional input must have a known type to generate the body.
    if any(name and types.get(name) is None for name in node.input):
        return None
    input_types = [
        types[name].SerializeToString() if name else onnx.TypeProto().SerializeToString()
        for name in node.input
    ]
    try:
        body_bytes = schema.get_context_dependent_function(
            node.SerializeToString(), input_types)
    except Exception:
        return None
    body = onnx.FunctionProto()
    body.ParseFromString(body_bytes)
    return body


def _expand_schema_functions(model: onnx.ModelProto) -> None:
    """In place: turn schema-registered function ops that have no bespoke MAC
    counter into uniquely-named model-local functions, so a later inlining pass
    exposes the compute (MatMuls, Convs, ...) inside their bodies.

    Best-effort and top-level only: a node whose body cannot be generated is
    left untouched (and thus counts as 0, as before). Ops with a bespoke counter
    are skipped so their exact counters keep winning.
    """
    opsets = {imp.domain: imp.version for imp in model.opset_import}
    types = _type_map(ModelInfo._infer_shapes(model).graph)
    existing = {(f.domain, f.name) for f in model.functions}
    new_functions = []
    for i, node in enumerate(model.graph.node):
        if node.op_type in _MAC_COUNTERS or (node.domain, node.op_type) in existing:
            continue
        body = _schema_function_body(node, types, opsets)
        if body is None:
            continue
        domain = f"__macs_expand_{i}"
        body.name = f"{node.op_type}__expanded"
        body.domain = domain
        new_functions.append(body)
        node.op_type = body.name
        node.domain = domain
        del node.attribute[:]  # the body is already specialized for these attrs
        model.opset_import.append(helper.make_opsetid(domain, 1))
    model.functions.extend(new_functions)


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    3. MACs / FLOPs of the compute-dominant operators: Conv, ConvTranspose,
       Gemm, MatMul, Attention, and the quantized twins (ConvInteger,
       QLinearConv, MatMulInteger, QLinearMatMul). Shapes come from ONNX shape
       inference; nodes whose shapes cannot be inferred contribute 0. Dynamic
       dimensions (``dim_param``, e.g. "batch") become sympy symbols when sympy
       is installed, so ``macs`` / ``flops`` may be a symbolic formula; without
       sympy they are assumed 1 (per-sample MACs). Function ops are expanded
       before counting, so the compute inside their bodies is included in the
       MAC total while op counts still list the function op itself: model-local
       functions (and nested ones) are inlined, and schema-registered function
       ops without a bespoke counter fall back to their context-dependent body
       (best-effort).
    TODO:
    Based on onnx runtime, get
    1、forward memory footprint
    2、memory access
    3、compute density
    """

    def get_info(self, graph: onnx.GraphProto, inherited_shapes: Optional[ShapeMap] = None) -> Tuple[Dict[str, int], Macs]:
        if inherited_shapes is None:
            inherited_shapes = {}
        shapes = _collect_shapes(graph, inherited_shapes)
        op_nums = defaultdict(int)
        macs = 0
        for node in graph.node:
            op_nums[node.op_type] += 1
            counter = _MAC_COUNTERS.get(node.op_type)
            if counter is not None:
                try:
                    macs += counter(node, shapes)
                except Exception as e:
                    warnings.warn(
                        f"Failed to count MACs for {node.op_type} node "
                        f"'{node.name}' ({e}); it is excluded from the total.",
                        stacklevel=2,
                    )
            for attr in node.attribute:
                sub_graphs = []
                if attr.HasField("g"):
                    sub_graphs.append(attr.g)
                sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_op_nums, sub_macs = self.get_info(sub_graph, shapes)
                    op_nums = defaultdict(int, {k: op_nums[k] + sub_op_nums[k] for k in set(op_nums) | set(sub_op_nums)})
                    macs += sub_macs
        op_nums["Constant"] += len(graph.initializer)
        return op_nums, macs

    def __init__(self, model: onnx.ModelProto):
        inferred = self._infer_shapes(model)
        # Op counts describe the model as authored, so a function op (e.g. a
        # fused block) is reported as a single op.
        self.op_nums, macs = self.get_info(inferred.graph)
        # ``graph.ByteSize()`` is the serialized size of the whole graph -- nested
        # subgraphs and inline tensor data included -- so it is taken once at the
        # top rather than summed per subgraph (which double-counted nested
        # graphs). Tensors kept as external data contribute nothing to ByteSize
        # (their ``raw_data`` is empty), so their on-disk lengths are added from
        # metadata. The size is therefore correct whether or not the external
        # data has been loaded, so callers need not materialize multi-GB weights
        # just to measure the model.
        self.model_size = model.graph.ByteSize() + _external_data_size(model.graph)
        # A function op's compute lives in its body, so for MACs we expose
        # function bodies and recount -- the primitive counters then see the
        # MatMuls, Convs, etc. inside every function instance.
        macs_graph = self._expanded_macs_graph(model)
        if macs_graph is not None:
            _, macs = self.get_info(macs_graph)
        self.macs = macs

    @staticmethod
    def _expanded_macs_graph(model: onnx.ModelProto) -> Optional[onnx.GraphProto]:
        """A shape-inferred graph with function bodies exposed for MAC counting,
        or None when there is nothing to expand.

        Model-local functions are inlined, and schema-registered function ops
        without a bespoke counter are first converted to local functions (see
        ``_expand_schema_functions``). Shapes are inferred with data propagation
        so the dynamic reshapes inside generated bodies (e.g. Attention) resolve
        and the internal MatMuls get concrete shapes.
        """
        if onnx_inliner is None:
            return None
        try:
            work = copy.deepcopy(model)
            _expand_schema_functions(work)
            if not work.functions:
                return None
            inlined = onnx_inliner.inline_local_functions(work)
            return ModelInfo._infer_shapes(inlined, data_prop=True).graph
        except Exception as e:
            warnings.warn(
                f"Failed to expand function bodies ({e}); MACs inside function "
                "bodies are not counted.",
                stacklevel=2,
            )
            return None

    @staticmethod
    def _infer_shapes(model: onnx.ModelProto, data_prop: bool = False) -> onnx.ModelProto:
        try:
            return shape_inference.infer_shapes(model, data_prop=data_prop)
        except Exception as e:
            # Shape inference can fail (e.g. models > 2GB); MACs then fall back
            # to 0 for nodes without pre-existing value_info.
            warnings.warn(
                f"Shape inference failed ({e}); MACs/FLOPs may be underestimated "
                "for nodes without existing shape info.",
                stacklevel=2,
            )
            return model

    @property
    def flops(self) -> int:
        return self.macs * 2


def print_simplifying_info(model_ori: onnx.ModelProto, model_opt: onnx.ModelProto) -> None:
    """
    --------------------------------------------------------
    |             | original model | simplified model |
    --------------------------------------------------------
    | ****        | ****           | ****             |
    --------------------------------------------------------
    | Model Size  | ****           | ****             |
    --------------------------------------------------------
    """
    ori_info = ModelInfo(model_ori)
    opt_info = ModelInfo(model_opt)
    table = Table()
    table.add_column('')
    table.add_column('Original Model')
    table.add_column('Simplified Model')

    def add_row(table: Table, key, ori_data, opt_data, is_better: Callable[[Any, Any], Any], postprocess: Optional[Callable[[Any], Any]] = None) -> None:
        if postprocess is None:
            postprocess = str
        if is_better(opt_data, ori_data):
            table.add_row(key, postprocess(ori_data), Text(
                postprocess(opt_data), style='bold green1'))
        else:
            table.add_row(key, postprocess(ori_data), postprocess(opt_data))

    for key in sorted(list(set(ori_info.op_nums.keys()) | set(opt_info.op_nums.keys()))):
        add_row(table, key, ori_info.op_nums[key],
                opt_info.op_nums[key], lambda opt, ori: opt < ori)
    add_row(
        table, 'Model Size', ori_info.model_size, opt_info.model_size, lambda opt, ori: opt < ori, postprocess=human_readable_size)

    # MACs/FLOPs may be symbolic, for which "<" yields an undecidable sympy
    # relational; compare representative magnitudes (all free dims -> 1) so the
    # highlighting still works without raising.
    def macs_improved(opt: Macs, ori: Macs) -> bool:
        return _representative_number(opt) < _representative_number(ori)

    add_row(
        table, 'MACs', ori_info.macs, opt_info.macs, macs_improved, postprocess=human_readable_num)
    add_row(
        table, 'FLOPs', ori_info.flops, opt_info.flops, macs_improved, postprocess=human_readable_num)
    print(table)
