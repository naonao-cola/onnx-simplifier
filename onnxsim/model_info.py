import warnings
from collections import defaultdict
from typing import Callable, Any, List, Optional, Tuple, Dict, Union

import onnx
from onnx import shape_inference
from rich.table import Table
from rich.text import Text
from rich import print

try:
    import sympy
except ImportError:  # sympy is an optional dependency; see _tensor_shape below.
    sympy = None


__all__ = ['ModelInfo', 'print_simplifying_info']


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
_MAC_COUNTERS: Dict[str, Callable[[onnx.NodeProto, ShapeMap], int]] = {}


def _register(*op_types: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        for op_type in op_types:
            _MAC_COUNTERS[op_type] = fn
        return fn
    return deco


@_register("Conv")
def _conv_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    # weight: [out_channels, in_channels / group, *kernel_shape]
    # output: [batch, out_channels, *spatial_out]
    weight = shapes.get(node.input[1])
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


@_register("MatMul")
def _matmul_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    # output: [*batch, M, N]; contraction dim K is the last dim of input A.
    a = shapes.get(node.input[0])
    output = shapes.get(node.output[0])
    if not _known(a) or not _known(output):
        return 0
    k = a[-1]
    return _prod(output) * k


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    3. MACs / FLOPs of the compute-dominant operators (Conv, ConvTranspose,
       Gemm, MatMul). Shapes come from ONNX shape inference; nodes whose shapes
       cannot be inferred contribute 0. Dynamic dimensions (``dim_param``, e.g.
       "batch") become sympy symbols when sympy is installed, so ``macs`` /
       ``flops`` may be a symbolic formula; without sympy they are assumed 1
       (per-sample MACs).
    TODO:
    Based on onnx runtime, get
    1、forward memory footprint
    2、memory access
    3、compute density
    """

    def get_info(self, graph: onnx.GraphProto, inherited_shapes: Optional[ShapeMap] = None) -> Tuple[Dict[str, int], int, int]:
        if inherited_shapes is None:
            inherited_shapes = {}
        shapes = _collect_shapes(graph, inherited_shapes)
        op_nums = defaultdict(int)
        model_size = 0
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
                if attr.g is not None:
                    sub_graphs.append(attr.g)
                if attr.graphs is not None:
                    sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_op_nums, sub_model_size, sub_macs = self.get_info(sub_graph, shapes)
                    op_nums = defaultdict(int, {k: op_nums[k] + sub_op_nums[k] for k in set(op_nums) | set(sub_op_nums)})
                    model_size += sub_model_size
                    macs += sub_macs
        op_nums["Constant"] += len(graph.initializer)
        model_size += graph.ByteSize()
        return op_nums, model_size, macs

    def __init__(self, model: onnx.ModelProto):
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            # Shape inference can fail (e.g. models > 2GB); MACs then fall back
            # to 0 for nodes without pre-existing value_info.
            warnings.warn(
                f"Shape inference failed ({e}); MACs/FLOPs may be underestimated "
                "for nodes without existing shape info.",
                stacklevel=2,
            )
        self.op_nums, self.model_size, self.macs = self.get_info(model.graph)

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
