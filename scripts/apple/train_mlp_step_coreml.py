#!/usr/bin/env python3
"""Train a small MLP on Core ML, one Adam step per predict() call.

Builds a 2-layer MLP Adam training step as an ONNX graph with onnxsim's own
training-graph tooling (`graph_grad.build_backward` for the backward pass,
`qat_graph.adam_update` + `make_step_graph` for the update and the
state-in/state-out plumbing), converts it with `onnxsim.export_coreml`, and
closes the loop in Python: each predict() call feeds the updated parameters
and Adam moments back as the next step's inputs -- the same state-feedback
shape as the KV-cache decode loop, but for weights instead of cache.

On an Apple-silicon Mac the whole step (forward + backward + Adam update)
lands on the Neural Engine, so this is on-device training through the public
Core ML stack -- the maderix/ANE project's training-throughput numbers
reached the same hardware through reverse-engineered private APIs instead
(see scripts/apple/README.md's training section for the measured comparison).

Two adaptations the Core ML target forces, both documented where they happen:

- Rank-0 scalar inputs (`lr`, Adam's bias corrections) are reshaped to
  shape-[1]: MIL has no rank-0 Placeholder. Elementwise broadcast makes the
  two forms numerically identical wherever they are used here.
- The default fp16 compute precision zeroes Adam's 1e-8 epsilon (below
  fp16's min normal) and underflows small gradients, so the step uses loss
  scaling (scaled backward seed, gradients unscaled before the moments) with
  eps=1e-3 -- the standard fp16-training recipe, and the same underflow class
  maderix documents for ANE backward matmuls.

Usage:
    python train_mlp_step_coreml.py --output mlp_step.mlpackage
    python train_mlp_step_coreml.py --steps 15 --compute-units CPU_AND_NE
    python train_mlp_step_coreml.py --qat-int8 --output mlp_step_qat.mlpackage
    python train_mlp_step_coreml.py --resident --compute-units CPU_ONLY
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnxsim import graph_grad, qat_graph

PARAMS = ("w1", "b1", "w2", "b2")
# Biases stay fp32 (standard QAT practice); only the two MatMul weights are
# fake-quantized.
QAT_PARAMS = ("w1", "w2")


def fake_quantize_weight(b: qat_graph.GraphBuilder, name: str, scale: float) -> str:
    """`name` through a symmetric int8 fake-quantization sandwich.

    Scale/zero-point are baked constants (the Core ML translator requires
    compile-time-constant QDQ parameters): `scale` is calibrated once from
    the initial weights and goes stale as they train, the standard
    simplification short of periodic re-estimation.
    """
    s = b.const(float(scale), hint=f"{name}_scale")
    zp_name = f"{name}_zp"
    b.initializer.append(numpy_helper.from_array(np.int8(0), zp_name))
    q = b.op("QuantizeLinear", [name, s, zp_name], f"{name}_q")
    dq = b.op("DequantizeLinear", [q, s, zp_name], f"{name}_dq")
    return q, dq


def build_step(
    batch: int = 256,
    dim: int = 1024,
    hidden: int = 2048,
    out: int = 1024,
    loss_scale: float = 256.0,
    eps: float = 1e-3,
    qat_int8: bool = False,
    scales: dict | None = None,
) -> qat_graph.StepGraph:
    """A 2-layer MLP Adam step: sigmoid MLP, MSE loss, Adam on every param."""
    if qat_int8 and scales is None:
        raise ValueError("qat_int8 needs per-weight scales (see calibrate_scales)")
    b = qat_graph.GraphBuilder()
    q1 = dq1 = "w1"
    q2 = dq2 = "w2"
    if qat_int8:
        q1, dq1 = fake_quantize_weight(b, "w1", scales["w1"])
        q2, dq2 = fake_quantize_weight(b, "w2", scales["w2"])
    z1m = b.matmul("x", dq1)
    z1 = b.add(z1m, "b1")
    h = b.sigmoid(z1)
    y_hatm = b.matmul(h, dq2)
    y_hat = b.add(y_hatm, "b2")
    diff = b.sub(y_hat, "y")
    sq = b.mul(diff, diff)
    loss = b.op("ReduceMean", [sq], "loss", keepdims=0)
    fwd_nodes = list(b.nodes)

    shapes: Dict[str, Tuple[int, ...]] = {
        "x": (batch, dim),
        "w1": (dim, hidden),
        "b1": (hidden,),
        "w2": (hidden, out),
        "b2": (out,),
        "y": (batch, out),
        q1: (dim, hidden),
        dq1: (dim, hidden),
        q2: (hidden, out),
        dq2: (hidden, out),
        z1m: (batch, hidden),
        z1: (batch, hidden),
        h: (batch, hidden),
        y_hatm: (batch, out),
        y_hat: (batch, out),
        diff: (batch, out),
        sq: (batch, out),
        loss: (),
    }
    seed = b.const(float(loss_scale), hint="loss_scale")
    grads = graph_grad.build_backward(
        b,
        fwd_nodes,
        shapes,
        grad_outputs={loss: seed},
        targets=list(PARAMS),
    )
    unscale = b.const(1.0 / float(loss_scale), hint="unscale")
    state = {}
    for p in PARAMS:
        g = b.mul(grads[p], unscale)
        w_next, m_next, v_next = qat_graph.adam_update(
            b,
            p,
            g,
            f"m_{p}",
            f"v_{p}",
            "lr",
            "m_correction",
            "v_correction",
            eps=eps,
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
    _widen_rank0_inputs(step.model)
    return step


def _widen_rank0_inputs(model: onnx.ModelProto) -> None:
    """Reshape rank-0 graph inputs to shape-[1] for the Core ML exporter.

    MIL has no rank-0 Placeholder, so `export_coreml` rejects scalar inputs.
    Every scalar here only ever meets elementwise broadcast arithmetic, where
    shape-[1] and rank-0 are numerically identical.
    """
    for vi in model.graph.input:
        if len(vi.type.tensor_type.shape.dim) == 0:
            vi.type.tensor_type.shape.dim.add().dim_value = 1


def make_data(batch: int, dim: int, out: int, seed: int = 0):
    """Fixed synthetic problem: fit a ReLU teacher from a fixed seed."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, dim), dtype=np.float32)
    w_true = (rng.standard_normal((dim, out)) / np.sqrt(dim)).astype(np.float32)
    y = np.maximum(x @ w_true, 0).astype(np.float32)
    return x, y


def calibrate_scales(
    state: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Per-tensor symmetric int8 scales from initial weights: max|w|/127.

    Calibrated once, from the same initial state the loop starts from (both
    default to seed 0), and baked in as constants -- scales go stale as
    weights train, the standard simplification short of periodic
    re-estimation.
    """
    return {p: float(max(np.abs(state[p]).max(), 1e-6)) / 127.0 for p in QAT_PARAMS}


def initial_state(
    shapes: Dict[str, Tuple[int, ...]], seed: int = 0
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = {}
    for p in PARAMS:
        s = shapes[p]
        state[p] = (
            (rng.standard_normal(s) / np.sqrt(s[0])).astype(np.float32)
            if len(s) > 1
            else np.zeros(s, np.float32)
        )
        state[f"m_{p}"] = np.zeros(s, np.float32)
        state[f"v_{p}"] = np.zeros(s, np.float32)
    return state


def run_loop(
    mlpackage: str,
    step: qat_graph.StepGraph,
    x: np.ndarray,
    y: np.ndarray,
    num_steps: int,
    lr: float,
    compute_units: str,
    shapes: Dict[str, Tuple[int, ...]],
) -> Tuple[List[float], float]:
    """Run the training loop through Core ML, threading state via predict().

    Returns (per-step losses, mean ms per step).
    """
    import coremltools as ct

    model = ct.models.MLModel(
        mlpackage, compute_units=getattr(ct.ComputeUnit, compute_units)
    )
    state = initial_state(shapes)
    losses: List[float] = []
    t0 = time.time()
    for t in range(num_steps):
        corr = qat_graph.adam_bias_corrections(t)
        feeds = {
            "x": x,
            "y": y,
            "lr": np.array([lr], np.float32),
            "m_correction": np.array([corr["m_correction"]], np.float32),
            "v_correction": np.array([corr["v_correction"]], np.float32),
            **state,
        }
        out = model.predict(feeds)
        for name, out_name in step.state.items():
            state[name] = out[out_name]
        losses.append(float(np.asarray(out[step.loss_name]).reshape(-1)[0]))
    return losses, (time.time() - t0) / num_steps * 1000


def run_resident_loop(
    mlpackage: str,
    step: qat_graph.StepGraph,
    x: np.ndarray,
    y: np.ndarray,
    num_steps: int,
    lr: float,
    compute_units: str,
    shapes: Dict[str, Tuple[int, ...]],
) -> Tuple[List[float], float]:
    """Run the training loop with weights resident on-device.

    The model must have been exported with ``state=dict(step.state)``: its
    parameters and moments live in Core ML states across ``predict()``
    calls, so only the batch, the target and the scalars cross the boundary
    each step (here ~2MB vs ~100MB shuttled). Initial state is written once
    up front; per-step losses come back as the only model output.
    """
    import coremltools as ct

    model = ct.models.MLModel(
        mlpackage, compute_units=getattr(ct.ComputeUnit, compute_units)
    )
    st = model.make_state()
    for name, value in initial_state(shapes).items():
        st.write_state(name, value.astype(np.float32))
    losses: List[float] = []
    t0 = time.time()
    for t in range(num_steps):
        corr = qat_graph.adam_bias_corrections(t)
        out = model.predict(
            {
                "x": x,
                "y": y,
                "lr": np.array([lr], np.float32),
                "m_correction": np.array([corr["m_correction"]], np.float32),
                "v_correction": np.array([corr["v_correction"]], np.float32),
            },
            state=st,
        )
        losses.append(float(np.asarray(out[step.loss_name]).reshape(-1)[0]))
    return losses, (time.time() - t0) / num_steps * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="mlp_step.mlpackage")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--out", type=int, default=1024)
    ap.add_argument(
        "--compute-units",
        default="CPU_AND_NE",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    ap.add_argument(
        "--qat-int8",
        action="store_true",
        help="Fake-quantize the MatMul weights to symmetric int8 in the "
        "forward pass (master fp32 weights train through straight-through "
        "estimation). Scales are calibrated once from the initial weights. "
        "Biases stay fp32.",
    )
    ap.add_argument(
        "--resident",
        action="store_true",
        help="Hold parameters and Adam moments resident on-device in Core ML "
        "states across predict() calls instead of shuttling them every step "
        "(iOS18+). Only the batch, target and scalars cross per step. ANE "
        "compilation rejects stateful programs on current macOS, so this "
        "runs CPU/GPU-side.",
    )
    args = ap.parse_args()

    import onnxsim

    shapes = {
        "w1": (args.dim, args.hidden),
        "b1": (args.hidden,),
        "w2": (args.hidden, args.out),
        "b2": (args.out,),
    }
    scales = calibrate_scales(initial_state(shapes)) if args.qat_int8 else None
    step = build_step(
        args.batch,
        args.dim,
        args.hidden,
        args.out,
        scales=scales,
        qat_int8=args.qat_int8,
    )
    print(
        f"step graph: {len(step.model.graph.node)} nodes, state: {sorted(step.state)}",
        flush=True,
    )
    export_kwargs: dict = {}
    if args.resident:
        export_kwargs["state"] = dict(step.state)
    onnxsim.export_coreml(
        step.model, args.output, skip_model_load=True, **export_kwargs
    )
    print(f"Wrote {args.output}", flush=True)

    x, y = make_data(args.batch, args.dim, args.out)
    loop = run_resident_loop if args.resident else run_loop
    losses, ms = loop(
        args.output,
        step,
        x,
        y,
        args.steps,
        args.lr,
        args.compute_units,
        shapes,
    )
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {args.steps} steps")
    print(f"{ms:.1f} ms/step ({args.compute_units})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
