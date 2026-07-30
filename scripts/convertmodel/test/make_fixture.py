#!/usr/bin/env python3
"""Generate the tiny ONNX model + reference IO used by the inference test.

The model is deliberately small but does real compute (a MatMul, a bias Add and
a Relu) with baked-in constant weights, so running it must produce the same
numbers on every iteration and on every execution provider. It is then annotated
with onnxsim's MAC/FLOP metrics (onnxsim PR #527) so the fixture also exercises
the metadata_props reader used by the "Run inference" panel and macs.test.mjs.
Re-run this to regenerate ``model.onnx`` and ``io.json`` after changing the
graph.

    python3 make_fixture.py

The MatMul MACs are hand-checkable: output [1, 3] with contraction dim K = 4 ->
1 * 3 * 4 = 12 MACs (24 FLOPs); Add and Relu contribute 0.
"""

import importlib.util
import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = pathlib.Path(__file__).parent
# onnxsim/model_info.py lives at the repo root; import it directly (its
# per-node annotation path is pure Python and needs no compiled extension).
_MODEL_INFO_PATH = (HERE / ".." / ".." / ".." / "onnxsim" / "model_info.py").resolve()

# Fixed weights -> deterministic reference outputs. Chosen so the pre-Relu
# result has both positive and negative entries, exercising the Relu clamp.
W = np.array(
    [
        [0.5, -1.0, 0.25],
        [-0.5, 0.5, 1.0],
        [1.0, 0.0, -0.75],
        [0.25, 0.5, -0.5],
    ],
    dtype=np.float32,
)
B = np.array([0.1, -0.2, 0.3], dtype=np.float32)
X = np.array([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)

REF = np.maximum(X @ W + B, 0.0)


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "B"], ["xwb"]),
        helper.make_node("Relu", ["xwb"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "inference_fixture",
        [x],
        [y],
        initializer=[
            numpy_helper.from_array(W, name="W"),
            numpy_helper.from_array(B, name="B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9  # onnxruntime-web 1.27 supports up to IR 10; 9 is safe.
    onnx.checker.check_model(model)
    return model


def _load_model_info():
    spec = importlib.util.spec_from_file_location("_onnxsim_model_info", _MODEL_INFO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotate(model: onnx.ModelProto) -> onnx.ModelProto:
    """Bake onnxsim's MAC/FLOP metrics into ``metadata_props`` (PR #527).

    Uses the real ``annotate_metadata`` when the compiled onnxsim extension is
    importable, otherwise the identical pure-Python annotation path (the C++
    metrics mirror those counters), so this works without building onnxsim.
    """
    import copy

    mi = _load_model_info()
    try:
        return mi.annotate_metadata(model)
    except Exception:  # compiled extension unavailable
        prefix = mi.METADATA_PREFIX
        work = mi.ModelInfo._infer_shapes(copy.deepcopy(model))
        macs, mem = mi._annotate_graph(work.graph, prefix, {}, {})
        footprint = mi.ModelInfo.__new__(mi.ModelInfo)._peak_memory_footprint(work.graph)
        density = 0 if mi._representative_number(mem) == 0 else (macs * 2) / mem
        totals = {
            "macs": macs, "flops": macs * 2, "mem_access": mem,
            "memory_footprint": footprint, "compute_density": density,
            "model_size": len(model.SerializeToString()),
        }
        for key, value in totals.items():
            mi._set_metadata(work, prefix + key, mi._metric_str(value))
        for key in ("macs", "flops", "mem_access"):
            mi._set_metadata(work.graph, prefix + key, mi._metric_str(totals[key]))
        return work


def main() -> None:
    model = annotate(build_model())
    onnx.save(model, HERE / "model.onnx")
    io = {
        "input": {"name": "X", "dims": list(X.shape), "data": X.flatten().tolist()},
        "output": {"name": "Y", "dims": list(REF.shape), "data": REF.flatten().tolist()},
    }
    (HERE / "io.json").write_text(json.dumps(io, indent=2) + "\n")
    print("wrote model.onnx and io.json; reference Y =", REF.flatten().tolist())


if __name__ == "__main__":
    main()
