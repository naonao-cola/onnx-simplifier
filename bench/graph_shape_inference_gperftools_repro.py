#!/usr/bin/env python3
"""Synthetic repro for gperftools-based profiling of
``InferShapesOnGraph`` (third_party/onnx/onnx/common/graph_shape_inference.cc),
run through onnxsim's real Simplify() pipeline (real schema-registered
inference functions, real fixed-point loop) -- see
bench/RESULTS_graph_shape_inference_gperftools.md for what this found.

Unlike bench/graph_shape_inference_teardown_bench.cpp (which builds the
scratch message tree standalone, without the schema registry), this script
drives the actual C++ extension end to end via onnxsim.simplify(), so a
CPU-time sampling profiler (gperftools' libprofiler.so) attached to the
Python process sees real ProcessNode/EncodeCurrentType/ConstantDataFor/
encodeTensor call stacks.

The model is built to spend a lot of time in InferShapesOnGraph across many
fixed-point rounds, and to exercise both open hypotheses from the
"graph-native shape inference caching" discussion:

  1. EncodeCurrentType's CopyFrom of a non-tensor TypeProto (Sequence/
     Optional/Map) -- exercised by periodic SequenceConstruct/SequenceAt
     round-trips through a single shared Constant-index node.
  2. ConstantDataFor + encodeTensor -- exercised by (a) three small
     initializers, each reused as the RHS of an Add/Mul/Sub on every block,
     and (b) two shared Constant nodes (int64 index/1-D-shape tensors)
     reused as Gather's index and Concat's second operand across every
     Shape/Gather/Concat/Reshape block, so the *same* Tensor/Constant-
     attribute objects get re-encoded from scratch on every node visit,
     every round.

See bench/RESULTS_graph_shape_inference_gperftools.md for what a profiling
run against this model actually found -- spoiler: both hypotheses are real
but tiny, dwarfed by an unrelated, much larger cost in graph Import itself.

Usage:
    python bench/graph_shape_inference_gperftools_repro.py gen out.onnx \\
        [--blocks N]
    python bench/graph_shape_inference_gperftools_repro.py run out.onnx \\
        [--loops N]

`run` calls onnxsim.simplify(..., perform_optimization=False) in a loop (so
total wall time is easily controlled to give a CPU profiler enough samples)
and prints wall time + a couple of sanity numbers. Meant to be invoked under
gperftools, e.g.:

    CPUPROFILE=/tmp/prof.out CPUPROFILE_FREQUENCY=1000 \\
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so \\
    python bench/graph_shape_inference_gperftools_repro.py run out.onnx --loops 8
"""

import argparse
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HIDDEN = 64  # small enough that HIDDEN-length tensors are well under
             # kMaxInputDataElements (4096), so ConstantDataFor's size gate
             # never excludes them.


def build_model(num_blocks: int) -> onnx.ModelProto:
    """Builds a chain of `num_blocks` Relu+Add+Mul+Sub blocks over a
    [dim_param("N"), HIDDEN] activation, with:
      - every 4th block: a Shape->Gather->Concat->Reshape group computing
        the (dynamic-batch, static-hidden) shape from scratch and
        reshaping the activation to it -- exercises the dynamic dim_param
        through EncodeCurrentType's plain-tensor path, plus (via the
        shared Constant/initializer nodes below) hypothesis 2.
      - every 8th block: a SequenceConstruct/SequenceAt round-trip --
        exercises hypothesis 1's non-tensor-type CopyFrom path.
    """
    nodes = []
    inits = []

    x_name = "x0"
    inputs = [helper.make_tensor_value_info(
        x_name, TensorProto.FLOAT, ["N", HIDDEN])]

    # Reused (shared) small initializers: consumed as the RHS of three
    # elementwise ops on every block (Add/Mul/Sub), so ConstantDataFor +
    # encodeTensor is exercised 3x per block instead of once, on the *same*
    # Tensor* every round every time -- the target of hypothesis 2's
    # caching idea. Three separate initializers (not the same one reused
    # three times) so the encoded-TensorProto cache a real implementation
    # would need is keyed correctly (three distinct cache entries, not one).
    rng = np.random.RandomState(0)
    for j, name in enumerate(("bias_vec", "scale_vec", "shift_vec")):
        vec = (rng.randn(HIDDEN).astype(np.float32) * 0.01) + j
        inits.append(numpy_helper.from_array(vec, name=name))

    # Reused (shared) Constant nodes: a single node each, whose output feeds
    # every block's Gather-index / Concat-operand / SequenceAt-position
    # input. Same Node/Value (and, once folded to an initializer, same
    # Tensor*) every round -- also targets hypothesis 2, but via a
    # Constant-node attribute value rather than a graph initializer (see the
    # address-stability question in the background doc).
    nodes.append(helper.make_node(
        "Constant", [], ["gather_idx0"], name="gather_idx0_const",
        value=numpy_helper.from_array(
            np.array([0], dtype=np.int64), name="gather_idx0_t")))
    nodes.append(helper.make_node(
        "Constant", [], ["const_hidden_1d"], name="const_hidden_1d_const",
        value=numpy_helper.from_array(
            np.array([HIDDEN], dtype=np.int64), name="const_hidden_1d_t")))
    nodes.append(helper.make_node(
        "Constant", [], ["seq_pos0"], name="seq_pos0_const",
        value=numpy_helper.from_array(
            np.array(0, dtype=np.int64), name="seq_pos0_t")))

    cur = x_name
    for i in range(num_blocks):
        relu_out = f"relu_{i}"
        nodes.append(helper.make_node("Relu", [cur], [relu_out]))
        add_out = f"add_{i}"
        nodes.append(helper.make_node("Add", [relu_out, "bias_vec"], [add_out]))
        mul_out = f"mul_{i}"
        nodes.append(helper.make_node("Mul", [add_out, "scale_vec"], [mul_out]))
        sub_out = f"sub_{i}"
        nodes.append(helper.make_node("Sub", [mul_out, "shift_vec"], [sub_out]))
        cur = sub_out

        if i % 4 == 0:
            shape_out = f"shape_{i}"
            gather_out = f"gather_{i}"
            concat_out = f"newshape_{i}"
            reshape_out = f"reshaped_{i}"
            nodes.append(helper.make_node("Shape", [cur], [shape_out]))
            nodes.append(helper.make_node(
                "Gather", [shape_out, "gather_idx0"], [gather_out], axis=0))
            nodes.append(helper.make_node(
                "Concat", [gather_out, "const_hidden_1d"], [concat_out], axis=0))
            nodes.append(helper.make_node("Reshape", [cur, concat_out], [reshape_out]))
            cur = reshape_out

        if i % 8 == 0:
            seq_out = f"seq_{i}"
            seq_at_out = f"seqat_{i}"
            nodes.append(helper.make_node("SequenceConstruct", [cur], [seq_out]))
            nodes.append(helper.make_node(
                "SequenceAt", [seq_out, "seq_pos0"], [seq_at_out]))
            cur = seq_at_out

    outputs = [helper.make_tensor_value_info(cur, TensorProto.FLOAT, ["N", HIDDEN])]

    graph = helper.make_graph(nodes, "gperftools_repro", inputs, outputs,
                               initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    return model


def cmd_gen(args):
    model = build_model(args.blocks)
    onnx.save(model, args.out)
    print(f"wrote {args.out}: {len(model.graph.node)} nodes, "
          f"{len(model.graph.initializer)} initializers, blocks={args.blocks}")


def cmd_run(args):
    # Imported here (not at module scope) so `gen` works even before the
    # onnxsim C++ extension is built.
    from onnxsim import simplify

    model = onnx.load(args.model, load_external_data=False)
    print(f"loaded {args.model}: {len(model.graph.node)} nodes")

    t0 = time.time()
    last_node_count = None
    for i in range(args.loops):
        out, ok = simplify(
            model,
            perform_optimization=False,
            check_n=0,
        )
        assert ok
        last_node_count = len(out.graph.node)
    dt = time.time() - t0
    print(f"loops={args.loops} total={dt:.3f}s avg={dt / args.loops:.3f}s "
          f"final_node_count={last_node_count}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen")
    p_gen.add_argument("out")
    p_gen.add_argument("--blocks", type=int, default=2000)
    p_gen.set_defaults(func=cmd_gen)

    p_run = sub.add_parser("run")
    p_run.add_argument("model")
    p_run.add_argument("--loops", type=int, default=5)
    p_run.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
