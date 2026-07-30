#!/usr/bin/env python3
"""Annotate an ONNX model with onnxsim's MAC/FLOP metrics for the browser demo.

This is the *offline* half of the ort-web-macs example. It takes an ONNX model
and writes onnxsim's per-node and per-model metrics into the model's
``metadata_props`` using :func:`onnxsim.model_info.annotate_metadata` (added in
https://github.com/onnxsim/onnxsim/pull/527). The annotated model is then loaded
by ``index.html``, which reads those ``metadata_props`` back in the browser to
report MACs/FLOPs while onnxruntime-web runs the actual inference.

The keys written (all under the ``onnxsim.`` prefix) are:

* model level: ``macs``, ``flops``, ``mem_access``, ``memory_footprint``,
  ``compute_density``, ``model_size``
* node level:  ``macs``, ``flops``, ``mem_access``
* value level: ``bytes``

Usage
-----
Annotate an existing model::

    python prepare_model.py path/to/model.onnx -o model.annotated.onnx

Or generate + annotate the bundled demo model (a tiny CNN classifier) so the
example is self-contained and needs no download::

    python prepare_model.py --demo -o sample_model.annotated.onnx

The demo model's MAC count is hand-verifiable:

    Conv1 : (1*16*32*32) * 3  * (3*3) =   442,368
    Conv2 : (1*32*16*16) * 16 * (3*3) = 1,179,648
    Gemm  :  1*10*32                  =       320
    total                             = 1,622,336 MACs  (3,244,672 FLOPs)
"""

import argparse
import importlib.util
import os
import sys

import onnx
from onnx import TensorProto, helper


# ---------------------------------------------------------------------------
# Import onnxsim.model_info without importing the whole onnxsim package.
#
# onnxsim/__init__.py pulls in the compiled C++ extension, which is not needed
# to annotate a model: the per-node metric annotation (annotate_metadata ->
# _annotate_graph) and the MAC counters are pure Python. We therefore load
# model_info.py directly as a standalone module. If the compiled extension *is*
# available, annotate_metadata uses it for the model-level totals; otherwise we
# fall back to summing the pure-Python per-node counters, which are identical by
# construction (the C++ side mirrors these counters).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_INFO_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "onnxsim", "model_info.py")
)


def _load_model_info():
    spec = importlib.util.spec_from_file_location("_onnxsim_model_info", _MODEL_INFO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mi = _load_model_info()


def annotate(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a shape-inferred copy of ``model`` with onnxsim metrics baked into
    ``metadata_props``.

    Prefers the real :func:`annotate_metadata` (which routes the model-level
    totals through the compiled C++ metrics). If the compiled extension is not
    importable in this environment, falls back to a pure-Python reproduction
    that uses the very same annotation code path (``_annotate_graph`` and the
    ``_MAC_COUNTERS``) for the per-node values and sums them for the totals.
    """
    try:
        return mi.annotate_metadata(model)
    except Exception as exc:  # compiled extension unavailable, etc.
        print(
            f"[prepare_model] annotate_metadata() unavailable ({exc}); "
            "using the pure-Python annotation path.",
            file=sys.stderr,
        )
        return _annotate_pure_python(model)


def _annotate_pure_python(model: onnx.ModelProto) -> onnx.ModelProto:
    import copy

    prefix = mi.METADATA_PREFIX
    ModelInfo = mi.ModelInfo

    # Shape-inferred working copy, exactly as annotate_metadata does.
    work = ModelInfo._infer_shapes(copy.deepcopy(model))

    # Per-node + per-value annotations and the graph subtotal (pure Python).
    macs, mem_access = mi._annotate_graph(work.graph, prefix, {}, {})
    flops = macs * 2

    # Peak footprint via the same liveness pass ModelInfo uses. Instantiate
    # without __init__ so we do not touch the compiled extension.
    inst = ModelInfo.__new__(ModelInfo)
    footprint = inst._peak_memory_footprint(work.graph)

    # Arithmetic intensity (roofline). 0 when no traffic is known.
    if mi._representative_number(mem_access) == 0:
        density = 0
    else:
        density = flops / mem_access

    model_size = len(model.SerializeToString())

    totals = {
        "macs": macs,
        "flops": flops,
        "mem_access": mem_access,
        "memory_footprint": footprint,
        "compute_density": density,
        "model_size": model_size,
    }
    for key, value in totals.items():
        mi._set_metadata(work, prefix + key, mi._metric_str(value))
    # Keep model- and graph-level macs/flops/mem_access in agreement.
    for key in ("macs", "flops", "mem_access"):
        mi._set_metadata(work.graph, prefix + key, mi._metric_str(totals[key]))
    return work


def build_demo_model() -> onnx.ModelProto:
    """A tiny but real CNN image classifier: Conv-Relu-Conv-Relu-GAP-Gemm.

    Input  : float32 [1, 3, 32, 32]  (name "input")
    Output : float32 [1, 10]         (name "output")

    Small enough to ship in the repo and to run instantly in the browser, yet it
    exercises the Conv and Gemm MAC counters so the reported numbers are
    non-trivial and hand-checkable.
    """
    import numpy as np

    rng = np.random.default_rng(0)

    def const(name, array):
        return helper.make_tensor(
            name, TensorProto.FLOAT, array.shape, array.astype(np.float32).ravel()
        )

    w1 = const("w1", rng.standard_normal((16, 3, 3, 3)) * 0.1)
    b1 = const("b1", np.zeros((16,)))
    w2 = const("w2", rng.standard_normal((32, 16, 3, 3)) * 0.1)
    b2 = const("b2", np.zeros((32,)))
    # Gemm with transB=1: weight is [out, in] = [10, 32], bias [10].
    gw = const("gw", rng.standard_normal((10, 32)) * 0.1)
    gb = const("gb", np.zeros((10,)))

    nodes = [
        helper.make_node(
            "Conv", ["input", "w1", "b1"], ["c1"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], name="conv1",
        ),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node(
            "Conv", ["r1", "w2", "b2"], ["c2"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2], name="conv2",
        ),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
        helper.make_node("GlobalAveragePool", ["r2"], ["p"], name="gap"),
        helper.make_node("Flatten", ["p"], ["f"], axis=1, name="flatten"),
        helper.make_node(
            "Gemm", ["f", "gw", "gb"], ["output"], transB=1, name="classifier",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "ort_web_macs_demo",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        initializer=[w1, b1, w2, b2, gw, gb],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _summarize(model: onnx.ModelProto) -> None:
    prefix = mi.METADATA_PREFIX
    meta = {p.key: p.value for p in model.metadata_props}
    print("Model-level onnxsim metrics (from metadata_props):")
    for key in (
        "macs", "flops", "mem_access", "memory_footprint",
        "compute_density", "model_size",
    ):
        val = meta.get(prefix + key)
        if val is not None:
            print(f"  {prefix + key:28s} = {val}")
    annotated_nodes = sum(
        1 for n in model.graph.node
        if any(p.key == prefix + "macs" for p in n.metadata_props)
    )
    print(f"Annotated {annotated_nodes}/{len(model.graph.node)} nodes with per-node metrics.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("model", nargs="?", help="input .onnx model to annotate")
    src.add_argument("--demo", action="store_true",
                     help="generate and annotate the bundled demo CNN instead")
    parser.add_argument("-o", "--output", default="sample_model.annotated.onnx",
                        help="path to write the annotated model (default: %(default)s)")
    args = parser.parse_args(argv)

    if args.demo:
        print("[prepare_model] building demo CNN classifier ...")
        model = build_demo_model()
    else:
        print(f"[prepare_model] loading {args.model} ...")
        model = onnx.load(args.model)

    annotated = annotate(model)
    onnx.save(annotated, args.output)
    print(f"[prepare_model] wrote {args.output}")
    _summarize(annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
