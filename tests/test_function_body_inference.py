"""Coverage for onnx's Graph-native function-body inference (added upstream in
onnxsim/onnx's ``function-body-inference-on-graph`` branch): a node calling an
onnx function -- model-local (``ModelProto.functions()``) or schema-attached
(``OpSchema::HasFunction()``) -- previously had its output left completely
untyped by onnxsim's shape-inference pass, exactly as if the op had no
registered schema at all. onnxsim now builds and passes the model-local
function map its C++ core needs (see ``BuildModelLocalFunctionsMap`` in
onnxsim.cpp) so this works even without inlining the function away first
(``inline_functions`` defaults to ``False``).
"""

import onnx
from onnx import parser

import onnxsim


def _clear_output_type(model, name, rank):
    # onnx.parser's own untyped output syntax (`=> (y)`, with no type
    # annotation at all) parses to a ValueInfoProto with no `type` field at
    # all, and onnxsim's own model checker requires every graph output to
    # declare both a `type` and a `shape` (a definite rank -- ONNX doesn't
    # allow a graph-boundary value to have a wholly unknown rank, only
    # individual unknown dim values within a known rank). So this builds the
    # least informative value_info the checker will accept: elem_type
    # UNDEFINED, `rank` dims each with neither dim_value nor dim_param set.
    # Any concrete type/dim values found on this output afterward must have
    # come from onnxsim actually running inference through the function
    # body, not from the original model.
    for vi in model.graph.output:
        if vi.name == name:
            vi.CopyFrom(
                onnx.helper.make_tensor_value_info(
                    name, onnx.TensorProto.UNDEFINED, [None] * rank
                )
            )
            return
    raise AssertionError(f"no such graph output: {name}")


def test_model_local_function_call_gets_shape_inferred_without_inlining():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 18, "custom.domain": 1]>
        agg (float[2,3,4] x) => (y)
        {
            y = custom.domain.MyNormalize (x)
        }
        """
    )
    func = parser.parse_function(
        """
        <domain: "custom.domain", opset_import: ["": 18]>
        MyNormalize (X) => (Y)
        {
            Y = Relu(X)
        }
        """
    )
    model.functions.append(func)
    _clear_output_type(model, "y", rank=3)

    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok

    # The call must survive un-inlined (inline_functions=False is the
    # default) -- this test is specifically about inferring *through* a
    # function call left in place, not about inlining making it moot.
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["MyNormalize"], op_types
    assert len(sim_model.functions) == 1

    y = sim_model.graph.output[0]
    assert y.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    dims = [d.dim_value for d in y.type.tensor_type.shape.dim]
    assert dims == [2, 3, 4], dims


def test_schema_attached_function_call_gets_shape_inferred():
    # MeanVarianceNormalization has a schema-attached function body
    # (OpSchema::HasFunction()) and no ordinary type/shape inference
    # function of its own -- confirmed via
    # onnx.defs.get_all_schemas_with_history() -- so it needs the same new
    # inference path as a model-local function, but with no map to plumb
    # through at all: it's resolved directly off the op's own schema.
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        agg (float[2,3,4] x) => (y)
        {
            y = MeanVarianceNormalization <axes = [0, 2]> (x)
        }
        """
    )
    _clear_output_type(model, "y", rank=3)

    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok

    y = sim_model.graph.output[0]
    assert y.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    dims = [d.dim_value for d in y.type.tensor_type.shape.dim]
    assert dims == [2, 3, 4], dims
