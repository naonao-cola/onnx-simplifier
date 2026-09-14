#!/usr/bin/env python3
"""Build a small neural-architecture-search space as training-step graphs,
one per candidate, using onnxsim's own reverse-mode differentiator
(``onnxsim.graph_grad``/``onnxsim.qat_graph``) instead of
``onnxruntime.training`` -- the same "one ordinary ONNX graph, no training-
enabled runtime anywhere" approach ``generate_distillation_step_graph.py``
and ``generate_federated_lora_step_graph.py`` already use for their own
tasks (see ../README.md's respective sections).

**Why a step graph is the right unit for NAS.** A search loop wants to run
every candidate architecture's own training loop and compare the results,
so each candidate needs its own compiled step graph -- there is no single
step graph that works for several different architectures at once, since
``onnxsim.graph_grad.build_backward`` bakes each op's static tensor shapes
in as it differentiates (see that module's own docstring on why: broadcast
axes are only recoverable from concrete shapes). So unlike distillation or
federated LoRA, where one step graph is built once and reused across many
*steps*, NAS calls this generator once *per candidate*, up front, and hands
the browser-side search loop the whole batch of resulting step graphs to
train and compare -- see ``../wasm/nas_search/search.mjs``.

**The search space and the loss**, deliberately the simplest thing that
still makes "some architectures fit better than others" observable: a plain
feedforward MLP (alternating ``MatMul``+``Add``+``Sigmoid``, one
``MatMul``+``Add`` for the final output layer, no activation on the
output), regressed against a fixed synthetic target with plain MSE
(``mean((y_hat - y) ** 2)``, built inline rather than via
``onnxsim.qat_graph.GraphBuilder.mean_square`` so this function has each
intermediate's own name in hand for the shape map ``build_backward``
needs) and trained with Adam on every weight -- the same composition
``scripts/apple/train_mlp_step_coreml.py``'s own ``build_step`` uses,
generalized from one hidden layer to an arbitrary list of hidden widths (a
candidate's "shape" in the architecture-search sense, see the repo's own
discussion of what building a gradient graph does and does not change about
a model's shape). Every op here (``MatMul``, ``Add``,
``Sub``, ``Mul``, ``Sigmoid``, ``ReduceMean``) is in both
``onnxsim.graph_grad.BACKWARD_OPS`` and ``onnxsim.qat_graph.EP_FRIENDLY_OPS``
already, so no execution-provider coverage work was needed to make these
candidates WebGPU/WebNN/NPU-trainable -- see ``../wasm/nas_search/`` for
running them there.

Like ``generate_federated_lora_step_graph.py`` (and unlike the distillation
step graph), every candidate here has a **fixed** batch size baked in at
generation time: this loss's own ``ReduceMean`` is not dynamic-batch-safe
(see that generator's own docstring for the same restriction), and NAS
compares candidates against the same fixed-size batches anyway, so nothing
is lost by fixing it up front.

Usage:
    python3 generate_nas_step_graphs.py -o /tmp/nas_candidates --batch-size 32 \\
        --dim 16 --out 8
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx

from onnxsim import graph_grad, qat_graph

# The search space: {candidate name: hidden-layer widths}. Deliberately
# spans clearly-too-small to comfortably-enough capacity for the default
# --dim/--out below, so a search loop training each candidate for only a
# handful of steps on the same synthetic problem can already tell them
# apart by final loss -- see ../wasm/nas_search/search.test.mjs, which
# checks exactly that ordering rather than only "it runs".
SEARCH_SPACE: Dict[str, List[int]] = {
    "tiny": [4],
    "small": [16],
    "medium": [32, 16],
    "wide": [64],
}

PARAM_PREFIX = "w"


def _layer_names(hidden_sizes: Sequence[int]) -> List[str]:
    """One name per ``Linear`` layer, hidden layers then the output layer."""
    return [f"{PARAM_PREFIX}{i}" for i in range(len(hidden_sizes) + 1)]


def build_candidate_step(
    hidden_sizes: Sequence[int],
    batch: int,
    dim: int,
    out: int,
) -> Tuple[qat_graph.StepGraph, Dict[str, Tuple[int, ...]]]:
    """A feedforward MLP Adam step: ``dim -> hidden_sizes... -> out``, MSE
    loss, Adam on every weight/bias -- one candidate architecture's whole
    trainable step, as a single ONNX graph.

    Returns ``(step, shapes)``, ``shapes`` mapping every weight/bias name to
    its own shape (what the caller needs to build the initial state and the
    manifest).
    """
    b = qat_graph.GraphBuilder()
    layers = _layer_names(hidden_sizes)
    widths = [dim, *hidden_sizes, out]

    # Every tensor the forward slice touches, keyed by the name each
    # GraphBuilder call returns -- graph_grad.build_backward needs all of
    # these, not just the weights, since a broadcast's axes are only
    # recoverable from concrete shapes (see graph_grad.py's own docstring).
    shapes: Dict[str, Tuple[int, ...]] = {"x": (batch, dim), "y": (batch, out)}
    activation = "x"
    for i, name in enumerate(layers):
        w, bias = name, f"{name}b"
        in_width, out_width = widths[i], widths[i + 1]
        shapes[w] = (in_width, out_width)
        shapes[bias] = (out_width,)
        z_shape = (batch, out_width)
        matmul_out = b.matmul(activation, w)
        shapes[matmul_out] = z_shape
        z = b.add(matmul_out, bias)
        shapes[z] = z_shape
        if i < len(layers) - 1:
            activation = b.sigmoid(z)
            shapes[activation] = z_shape
        else:
            activation = z
    y_hat = activation
    out_shape = shapes[y_hat]

    diff = b.sub(y_hat, "y")
    shapes[diff] = out_shape
    sq = b.mul(diff, diff)
    shapes[sq] = out_shape
    loss = b.op("ReduceMean", [sq], keepdims=0)
    shapes[loss] = ()
    fwd_nodes = list(b.nodes)

    grads = graph_grad.build_backward(
        b,
        fwd_nodes,
        shapes,
        grad_outputs={loss: b.const(1.0, hint="loss_seed")},
        targets=[n for name in layers for n in (name, f"{name}b")],
    )

    state: Dict[str, Tuple[Sequence[int], str]] = {}
    for name in layers:
        for p in (name, f"{name}b"):
            w_next, m_next, v_next = qat_graph.adam_update(
                b,
                p,
                grads[p],
                f"m_{p}",
                f"v_{p}",
                "lr",
                "m_correction",
                "v_correction",
            )
            shape = shapes[p]
            state[p] = (shape, w_next)
            state[f"m_{p}"] = (shape, m_next)
            state[f"v_{p}"] = (shape, v_next)

    step = qat_graph.make_step_graph(
        b,
        constants={
            "x": ((batch, dim), onnx.TensorProto.FLOAT),
            "y": ((batch, out), onnx.TensorProto.FLOAT),
        },
        state=state,
        scalars=["lr", "m_correction", "v_correction"],
        loss=loss,
    )
    onnx.checker.check_model(step.model)
    return step, shapes


def initial_state(
    shapes: Dict[str, Tuple[int, ...]], layers: Sequence[str], seed: int = 0
) -> Dict[str, np.ndarray]:
    """Fresh Glorot-ish weights, zero biases, zero Adam moments -- the same
    scheme ``train_mlp_step_coreml.py.initial_state`` uses, generalized to an
    arbitrary layer count.
    """
    rng = np.random.default_rng(seed)
    state: Dict[str, np.ndarray] = {}
    for name in layers:
        w_shape = shapes[name]
        b_shape = shapes[f"{name}b"]
        state[name] = (rng.standard_normal(w_shape) / np.sqrt(w_shape[0])).astype(
            np.float32
        )
        state[f"{name}b"] = np.zeros(b_shape, np.float32)
        for p, shape in ((name, w_shape), (f"{name}b", b_shape)):
            state[f"m_{p}"] = np.zeros(shape, np.float32)
            state[f"v_{p}"] = np.zeros(shape, np.float32)
    return state


def write_manifest_and_initial_state(
    step: qat_graph.StepGraph,
    state: Dict[str, np.ndarray],
    layers: Sequence[str],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    manifest_path: str,
    initial_state_path: str,
) -> None:
    """Same flat, line-oriented manifest format
    ``generate_federated_lora_step_graph.py`` writes (``input_name``/
    ``input_shape``/``teacher_name``/``teacher_shape``/``lr_name``/
    ``loss_name``/``state``/``weight`` lines) -- deliberately identical, not
    just similar, so ``../wasm/nas_search/search.mjs`` can run every
    candidate through the *existing*
    ``../wasm/federated_lora/step_graph_runner.mjs`` unchanged instead of
    duplicating a second copy of ``StepGraphSession``/``parseManifest``/
    ``loadInitialState`` for what is, from the runner's point of view, the
    exact same shape of step graph (fixed-batch input/target/loss/state/
    weight).
    """
    weight_names = [n for name in layers for n in (name, f"{name}b")]
    with open(manifest_path, "w") as f:
        f.write("input_name x\n")
        f.write(f"input_shape {' '.join(str(d) for d in input_shape)}\n")
        f.write("teacher_name y\n")
        f.write(f"teacher_shape {' '.join(str(d) for d in output_shape)}\n")
        f.write("lr_name lr\n")
        f.write(f"loss_name {step.loss_name}\n")
        for name, out_name in step.state.items():
            shape = list(state[name].shape)
            f.write(f"state {name} {out_name} {' '.join(str(d) for d in shape)}\n")
        for name in weight_names:
            shape = list(state[name].shape)
            f.write(f"weight {name} {' '.join(str(d) for d in shape)}\n")

    with open(initial_state_path, "wb") as f:
        for name in weight_names:
            state[name].astype(np.float32).tofile(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output-dir", required=True, help="directory to write candidates into")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--dim", type=int, default=16, help="input width")
    p.add_argument("--out", type=int, default=8, help="output width")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    space_manifest = {"batch_size": args.batch_size, "dim": args.dim, "out": args.out, "candidates": []}

    for name, hidden_sizes in SEARCH_SPACE.items():
        step, shapes = build_candidate_step(hidden_sizes, args.batch_size, args.dim, args.out)
        layers = _layer_names(hidden_sizes)
        state = initial_state(shapes, layers, seed=args.seed)

        step_path = os.path.join(args.output_dir, f"{name}.step_graph.onnx")
        manifest_path = f"{step_path}.manifest.txt"
        initial_state_path = f"{step_path}.initial_state.bin"

        onnx.save(step.model, step_path)
        write_manifest_and_initial_state(
            step,
            state,
            layers,
            (args.batch_size, args.dim),
            (args.batch_size, args.out),
            manifest_path,
            initial_state_path,
        )

        param_count = sum(int(np.prod(shapes[n])) for n in layers) + sum(
            int(np.prod(shapes[f"{n}b"])) for n in layers
        )
        # Every op type this candidate's finished (simplified) step graph
        # actually contains -- read back from the model rather than
        # hand-enumerated, so a caller checking execution-provider placement
        # (see scripts/convertmodel/test/webgpu_nas_search.test.mjs) has a
        # list that cannot drift from what this run actually emitted. Every
        # candidate in SEARCH_SPACE happens to use the same op-type
        # vocabulary regardless of width/depth (more hidden units or layers
        # repeats the same ops, it does not introduce new ones), so `ops`
        # alone cannot tell candidates apart -- num_nodes (which does grow
        # with depth) is what a caller picking "the most structurally
        # complex candidate" to placement-check should sort on instead.
        ops = sorted({node.op_type for node in step.model.graph.node})
        space_manifest["candidates"].append(
            {
                "name": name,
                "hidden_sizes": list(hidden_sizes),
                "param_count": param_count,
                "num_nodes": len(step.model.graph.node),
                "step_graph": os.path.basename(step_path),
                "manifest": os.path.basename(manifest_path),
                "initial_state": os.path.basename(initial_state_path),
                "ops": ops,
            }
        )
        print(
            f"wrote {step_path} ({len(step.model.graph.node)} nodes, "
            f"{param_count} params, hidden={list(hidden_sizes)})"
        )

    search_space_path = os.path.join(args.output_dir, "search_space.json")
    with open(search_space_path, "w") as f:
        json.dump(space_manifest, f, indent=2)
    print(f"wrote {search_space_path}")


if __name__ == "__main__":
    main()
