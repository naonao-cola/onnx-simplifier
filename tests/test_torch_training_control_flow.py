"""`onnxsim.torch_training._raise_if_control_flow_survived` needs nothing
torch-specific -- it is a plain node scan -- so it gets its own,
torch-independent test file rather than living in `test_torch_training.py`
(which needs torch at import time for everything else in it).
"""

import onnx
from onnx import helper, parser

from onnxsim.torch_training import _raise_if_control_flow_survived


def test_a_model_with_no_control_flow_passes_through_unchanged():
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"])],
            "g",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    assert _raise_if_control_flow_survived(model) is model


def test_a_surviving_if_raises_naming_it():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        agraph (bool cond, float[4] x) => (float[4] y) {
          y = If<
            then_branch = then_g () => (float[4] t) { t = Relu(x) },
            else_branch = else_g () => (float[4] e) { e = Neg(x) }
          >(cond)
        }
        """
    )
    try:
        _raise_if_control_flow_survived(model)
    except ValueError as error:
        assert "If" in str(error)
    else:
        raise AssertionError("expected a ValueError naming the surviving If node")
