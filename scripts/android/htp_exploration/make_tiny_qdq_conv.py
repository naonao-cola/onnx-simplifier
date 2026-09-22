"""Build the smallest possible QDQ int8 conv ONNX model -- DequantizeLinear(x) ->
DequantizeLinear(w) -> Conv -> QuantizeLinear(y) -- matching the real Mask R-CNN
backbone's per-layer QDQ pattern (uint8 activation, int8 weight, uint8 output), used
to isolate whether a QNN Execution Provider failure is graph-specific or environmental.
See README.md's "Stage 1" for why this exists and what it found.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(out_path: str = "tiny_qdq_conv.onnx") -> None:
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 8, 16, 16])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 8, 16, 16])

    rng = np.random.default_rng(0)
    w = rng.integers(-40, 40, (8, 8, 3, 3)).astype(np.int8)
    initializers = [
        numpy_helper.from_array(w, name="w"),
        numpy_helper.from_array(np.float32(0.02), name="x_scale"),
        numpy_helper.from_array(np.uint8(114), name="x_zp"),
        numpy_helper.from_array(np.float32(0.01), name="w_scale"),
        numpy_helper.from_array(np.int8(0), name="w_zp"),
        numpy_helper.from_array(np.float32(0.05), name="y_scale"),
        numpy_helper.from_array(np.uint8(120), name="y_zp"),
        numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="bias"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["x", "x_scale", "x_zp"], ["x_f"]),
        helper.make_node("DequantizeLinear", ["w", "w_scale", "w_zp"], ["w_f"]),
        helper.make_node("Conv", ["x_f", "w_f", "bias"], ["y_f"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("QuantizeLinear", ["y_f", "y_scale", "y_zp"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "tiny_qdq_conv", [x], [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


if __name__ == "__main__":
    build()
    print("wrote tiny_qdq_conv.onnx")
