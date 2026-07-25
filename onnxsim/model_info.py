from collections import defaultdict
from typing import Callable, Any, Iterable, Optional, Dict

import onnx
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from rich.table import Table
from rich.text import Text
from rich import print


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


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    TODO: 
    Based on onnx runtime, get
    1、FLOPs
    2、forward memory footprint
    3、memory access
    4、compute density
    """

    def get_info(self, graph: onnx.GraphProto) -> Dict[str, int]:
        op_nums = defaultdict(int)
        for node in graph.node:
            op_nums[node.op_type] += 1
            for attr in node.attribute:
                sub_graphs = []
                if attr.HasField("g"):
                    sub_graphs.append(attr.g)
                sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_op_nums = self.get_info(sub_graph)
                    op_nums = defaultdict(int, {k: op_nums[k] + sub_op_nums[k] for k in set(op_nums) | set(sub_op_nums)})
        op_nums["Constant"] += len(graph.initializer)
        return op_nums

    def __init__(self, model: onnx.ModelProto):
        self.op_nums = self.get_info(model.graph)
        # ``graph.ByteSize()`` is the serialized size of the whole graph -- nested
        # subgraphs and inline tensor data included -- so it is taken once at the
        # top rather than summed per subgraph (which double-counted nested
        # graphs). Tensors kept as external data contribute nothing to ByteSize
        # (their ``raw_data`` is empty), so their on-disk lengths are added from
        # metadata. The size is therefore correct whether or not the external
        # data has been loaded, so callers need not materialize multi-GB weights
        # just to measure the model.
        self.model_size = model.graph.ByteSize() + _external_data_size(model.graph)


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
    print(table)
