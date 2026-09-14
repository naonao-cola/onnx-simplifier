#!/usr/bin/env python3
"""Build a DARTS-style differentiable-architecture-search supernet as one
training-step graph, using onnxsim's own reverse-mode differentiator
(``onnxsim.graph_grad``/``onnxsim.qat_graph``) instead of
``onnxruntime.training`` -- the same "one ordinary ONNX graph, no training-
enabled runtime anywhere" approach ``generate_nas_step_graphs.py`` (discrete
candidates, one step graph each) already uses, applied to DARTS's own
continuous-relaxation idea instead.

**What DARTS actually is, and why it fits a step graph better than the
discrete search does.** Liu et al.'s DARTS (arXiv:1806.09055) replaces
"pick one of several candidate operations at this point in the network"
with "compute a *weighted sum* of every candidate operation, and make the
weights themselves trainable parameters, softmax-normalized so they sum to
1." Trained jointly with the network's own weights by ordinary gradient
descent, the architecture weights (``alpha`` below) drift toward whichever
candidate actually reduces the loss -- an architecture search that never
leaves gradient descent, i.e. never needs a discrete search loop training
and comparing separate models the way ``generate_nas_step_graphs.py``'s
``search.mjs`` does. That makes it a single *supernet* step graph, not one
step graph per candidate: every candidate operation's own weights, and the
mixing weights over them, are just more trainable state in the same graph.

This is a deliberately small, "operation choice" DARTS cell (which
nonlinearity depth to apply between two trained linear layers), not the
convolutional-cell search the paper demonstrates on CIFAR-10 -- picking a
cell shape that keeps every candidate branch the same input/output shape
avoids ever needing to combine differently-shaped branches (real DARTS
picks among ops that already share a shape contract for the same reason).
The three candidates, all parameter-free so every branch's "extra cost" is
purely which nonlinearity is applied, not extra trainable weights some
branches have and others don't:

- ``Identity`` -- no nonlinearity (a "skip" in DARTS terms)
- ``Sigmoid`` -- the same nonlinearity the rest of this repo's step graphs
  use
- ``Sigmoid`` applied twice -- a deeper, more saturating nonlinearity

**Why this is also single-level DARTS, not the paper's bi-level
optimization.** The paper alternates: update ``alpha`` against a held-out
validation batch, update the network weights against a training batch,
back and forth, specifically to stop ``alpha`` from just overfitting the
training data the network weights are also fitting. This generator instead
updates everything -- the two linear layers and ``alpha`` -- from the same
Adam step against the same batch, the simplified variant most from-scratch
DARTS write-ups reach for first. It is not a claim that this is exactly the
paper's own procedure, and that omission (not one specific dataset needing
a validation split to hold out) is why: a step graph has no concept of
"two batches, alternating which state updates from which," and giving it
one would roughly double this generator's own complexity for a demo this
size.

Every op here (``MatMul``, ``Add``, ``Sub``, ``Mul``, ``Sigmoid``, ``Exp``,
``ReduceSum``, ``Div``, ``Gather``, ``Identity``, ``ReduceMean``) is in both
``onnxsim.graph_grad.BACKWARD_OPS``'s rule table and
``onnxsim.qat_graph.EP_FRIENDLY_OPS`` already -- notably, softmax over
``alpha`` is hand-built from ``Exp``/``ReduceSum``/``Div`` rather than the
fused ``Softmax`` op (which has a gradient rule but is not in
``EP_FRIENDLY_OPS``), specifically so this supernet stays inside the same
WebGPU/WebNN/NPU-verified op set ``generate_nas_step_graphs.py``'s
candidates do, with no new execution-provider coverage work needed.

**Reading off the winning architecture.** ``alpha``'s state output at any
step, run through the same hand-built softmax on the host (or just
``argmax`` -- softmax is monotonic, so the ordering is the same either
way), names which candidate currently dominates the mixture. See
``tools/onnx-finetune/wasm/nas_search/darts_supernet.test.mjs`` for a
worked example: on the same purely-linear synthetic regression
``generate_nas_step_graphs.py``'s own search space is trained against,
``Identity`` should win -- two Linear layers with an ``Identity`` in
between still compose into one unconstrained linear map, an exact fit
(modulo noise) for a linear target, while routing any weight onto the
``Sigmoid`` branches only adds unwanted nonlinearity.

Usage:
    python3 generate_darts_supernet_step_graph.py -o /tmp/darts_supernet \\
        --batch-size 32 --dim 16 --hidden 16 --out 8
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import onnx

from onnxsim import graph_grad, qat_graph

PARAMS = ("L1_w", "L1_b", "L2_w", "L2_b", "alpha")
NUM_CANDIDATES = 3  # Identity, Sigmoid, Sigmoid-of-Sigmoid -- see module docstring.


def _int64_const(b: qat_graph.GraphBuilder, value: int, hint: str) -> str:
    """A rank-0 int64 initializer, for ``Gather``'s ``indices`` input --
    ``GraphBuilder.const`` is float32-only (see its own docstring), and
    ``graph_grad._Backward.int64_const`` is a backward-rule-construction
    helper, not something forward-graph code like this has an instance of.
    """
    name = b.name(hint)
    b.initializer.append(onnx.numpy_helper.from_array(np.array(value, dtype=np.int64), name))
    return name


def build_darts_supernet_step(
    batch: int,
    dim: int,
    hidden: int,
    out: int,
) -> Tuple[qat_graph.StepGraph, Dict[str, Tuple[int, ...]]]:
    """``dim -> hidden -> out``, with a 3-way DARTS-mixed nonlinearity
    between the two linear layers, MSE loss, Adam on every weight
    (including the architecture weights ``alpha``) -- see this module's own
    docstring for the full design.

    Returns ``(step, shapes)``, ``shapes`` mapping every trainable tensor's
    name to its own shape (what the caller needs to build the initial state
    and the manifest).
    """
    b = qat_graph.GraphBuilder()
    shapes: Dict[str, Tuple[int, ...]] = {
        "x": (batch, dim),
        "y": (batch, out),
        "L1_w": (dim, hidden),
        "L1_b": (hidden,),
        "L2_w": (hidden, out),
        "L2_b": (out,),
        "alpha": (NUM_CANDIDATES,),
    }

    z1_mm = b.matmul("x", "L1_w")
    shapes[z1_mm] = (batch, hidden)
    z1 = b.add(z1_mm, "L1_b")
    shapes[z1] = (batch, hidden)

    c0 = b.op("Identity", [z1])
    shapes[c0] = (batch, hidden)
    c1 = b.sigmoid(z1)
    shapes[c1] = (batch, hidden)
    c2 = b.sigmoid(c1)
    shapes[c2] = (batch, hidden)
    candidates = [c0, c1, c2]

    alpha_exp = b.op("Exp", ["alpha"])
    shapes[alpha_exp] = (NUM_CANDIDATES,)
    alpha_sum = b.op("ReduceSum", [alpha_exp], keepdims=0)
    shapes[alpha_sum] = ()
    alpha_weights = b.div(alpha_exp, alpha_sum)
    shapes[alpha_weights] = (NUM_CANDIDATES,)

    terms = []
    for i, candidate in enumerate(candidates):
        idx = _int64_const(b, i, f"alpha_idx{i}")
        shapes[idx] = ()
        weight = b.op("Gather", [alpha_weights, idx])
        shapes[weight] = ()
        term = b.mul(candidate, weight)
        shapes[term] = (batch, hidden)
        terms.append(term)
    h = b.add(terms[0], terms[1])
    shapes[h] = (batch, hidden)
    h = b.add(h, terms[2])
    shapes[h] = (batch, hidden)

    y_hat_mm = b.matmul(h, "L2_w")
    shapes[y_hat_mm] = (batch, out)
    y_hat = b.add(y_hat_mm, "L2_b")
    shapes[y_hat] = (batch, out)

    diff = b.sub(y_hat, "y")
    shapes[diff] = (batch, out)
    sq = b.mul(diff, diff)
    shapes[sq] = (batch, out)
    loss = b.op("ReduceMean", [sq], keepdims=0)
    shapes[loss] = ()
    fwd_nodes = list(b.nodes)

    grads = graph_grad.build_backward(
        b,
        fwd_nodes,
        shapes,
        grad_outputs={loss: b.const(1.0, hint="loss_seed")},
        targets=list(PARAMS),
    )

    state: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for p in PARAMS:
        w_next, m_next, v_next = qat_graph.adam_update(
            b, p, grads[p], f"m_{p}", f"v_{p}", "lr", "m_correction", "v_correction"
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


def initial_state(shapes: Dict[str, Tuple[int, ...]], seed: int = 0) -> Dict[str, np.ndarray]:
    """Fresh Glorot-ish weights for the two linear layers, zero biases, a
    uniform (all-zero, pre-softmax) ``alpha`` -- so the mixture starts as an
    equal 1/3, 1/3, 1/3 blend of every candidate rather than favoring one
    before training even begins -- and zero Adam moments.
    """
    rng = np.random.default_rng(seed)
    state: Dict[str, np.ndarray] = {}
    for p in ("L1_w", "L2_w"):
        shape = shapes[p]
        state[p] = (rng.standard_normal(shape) / np.sqrt(shape[0])).astype(np.float32)
    for p in ("L1_b", "L2_b"):
        state[p] = np.zeros(shapes[p], np.float32)
    state["alpha"] = np.zeros(shapes["alpha"], np.float32)
    for p in PARAMS:
        shape = shapes[p]
        state[f"m_{p}"] = np.zeros(shape, np.float32)
        state[f"v_{p}"] = np.zeros(shape, np.float32)
    return state


def write_manifest_and_initial_state(
    step: qat_graph.StepGraph,
    state: Dict[str, np.ndarray],
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    manifest_path: str,
    initial_state_path: str,
) -> None:
    """Same flat manifest format ``generate_nas_step_graphs.py``/
    ``generate_federated_lora_step_graph.py`` write -- deliberately
    identical, so the browser side can run this supernet through the
    *existing* ``../wasm/federated_lora/step_graph_runner.mjs`` unchanged,
    treating ``alpha`` as just one more ``weight`` entry (which, as far as
    that runner is concerned, it is).
    """
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
        for name in PARAMS:
            shape = list(state[name].shape)
            f.write(f"weight {name} {' '.join(str(d) for d in shape)}\n")

    with open(initial_state_path, "wb") as f:
        for name in PARAMS:
            state[name].astype(np.float32).tofile(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", required=True, help="path to write the step graph to")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--dim", type=int, default=16, help="input width")
    p.add_argument("--hidden", type=int, default=16, help="hidden width")
    p.add_argument("--out", type=int, default=8, help="output width")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    step, shapes = build_darts_supernet_step(args.batch_size, args.dim, args.hidden, args.out)
    state = initial_state(shapes, seed=args.seed)

    onnx.save(step.model, args.output)
    manifest_path = f"{args.output}.manifest.txt"
    initial_state_path = f"{args.output}.initial_state.bin"
    write_manifest_and_initial_state(
        step,
        state,
        (args.batch_size, args.dim),
        (args.batch_size, args.out),
        manifest_path,
        initial_state_path,
    )
    print(
        f"wrote {args.output} ({len(step.model.graph.node)} nodes), "
        f"{manifest_path}, {initial_state_path}"
    )


if __name__ == "__main__":
    main()
