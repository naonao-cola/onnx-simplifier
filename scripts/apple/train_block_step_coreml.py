#!/usr/bin/env python3
"""Train one transformer block on Core ML, one Adam step per predict() call.

Same shape as `train_mlp_step_coreml.py` (build with onnxsim's training-graph
tooling, convert with `onnxsim.export_coreml`, close the loop through
`predict()` with parameter/moment state feedback), but the step is a
pre-norm-less transformer block instead of an MLP: Q/K/V projections, head
split, scaled dot-product attention, output projection, residual, SiLU FFN,
residual, MSE loss, Adam on all 12 projection params. The attention backward
(`sdpaBwd` in maderix/ANE terms) runs on the Neural Engine here along with
everything else.

The Core ML adaptations are the same as the MLP script's: rank-0 scalars
widened to shape-[1] (MIL has no rank-0 Placeholder), loss scaling with
eps=1e-3 for default fp16 compute (Adam's 1e-8 zeroes out in fp16).

Usage:
    python train_block_step_coreml.py --output block_step.mlpackage
    python train_block_step_coreml.py --steps 8 --compute-units CPU_AND_NE
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

PARAMS = (
    "wq",
    "bq",
    "wk",
    "bk",
    "wv",
    "bv",
    "wo",
    "bo",
    "wu",
    "bu",
    "wd",
    "bd",
)


def build_block_step(
    batch: int = 32,
    seq: int = 64,
    dim: int = 512,
    heads: int = 8,
    head_dim: int = 64,
    ffn: int = 2048,
    out: int = 512,
    loss_scale: float = 256.0,
    eps: float = 1e-3,
) -> Tuple[qat_graph.StepGraph, Dict[str, Tuple[int, ...]]]:
    """A transformer-block Adam step; returns (step graph, param shapes)."""
    pshapes = {
        "wq": (dim, dim),
        "bq": (dim,),
        "wk": (dim, dim),
        "bk": (dim,),
        "wv": (dim, dim),
        "bv": (dim,),
        "wo": (dim, dim),
        "bo": (dim,),
        "wu": (dim, ffn),
        "bu": (ffn,),
        "wd": (ffn, dim),
        "bd": (dim,),
    }
    b = qat_graph.GraphBuilder()
    b.initializer.append(
        numpy_helper.from_array(
            np.array([batch, seq, heads, head_dim], dtype=np.int64), "shape_qkv"
        )
    )
    b.initializer.append(
        numpy_helper.from_array(
            np.array([batch, seq, dim], dtype=np.int64), "shape_merge"
        )
    )
    b.initializer.append(
        numpy_helper.from_array(np.float32(1.0 / np.sqrt(head_dim)), "scale")
    )
    shapes: Dict[str, Tuple[int, ...]] = {
        "x": (batch, seq, dim),
        "y": (batch, seq, out),
        "shape_qkv": (4,),
        "shape_merge": (3,),
        "scale": (),
    }
    shapes.update(pshapes)

    def R(name: str, shape: Tuple[int, ...]) -> str:
        shapes[name] = shape
        return name

    def split_heads(t: str, tag: str):
        r = R(
            b.op("Reshape", [t, "shape_qkv"], f"r_{tag}"), (batch, seq, heads, head_dim)
        )
        return R(
            b.op("Transpose", [r], f"t_{tag}", perm=[0, 2, 1, 3]),
            (batch, heads, seq, head_dim),
        )

    xwq = R(b.matmul("x", "wq"), (batch, seq, dim))
    zq = R(b.add(xwq, "bq"), (batch, seq, dim))
    q = split_heads(zq, "q")
    xwk = R(b.matmul("x", "wk"), (batch, seq, dim))
    zk = R(b.add(xwk, "bk"), (batch, seq, dim))
    k = split_heads(zk, "k")
    xwv = R(b.matmul("x", "wv"), (batch, seq, dim))
    zv = R(b.add(xwv, "bv"), (batch, seq, dim))
    v = split_heads(zv, "v")
    kt = R(
        b.op("Transpose", [k], "kt", perm=[0, 1, 3, 2]), (batch, heads, head_dim, seq)
    )
    qkt = R(b.matmul(q, kt), (batch, heads, seq, seq))
    scores = R(b.mul(qkt, "scale"), (batch, heads, seq, seq))
    p = R(b.op("Softmax", [scores], "p", axis=-1), (batch, heads, seq, seq))
    ctx = R(b.matmul(p, v), (batch, heads, seq, head_dim))
    ctm = R(
        b.op("Transpose", [ctx], "ctm", perm=[0, 2, 1, 3]),
        (batch, seq, heads, head_dim),
    )
    mrg = R(b.op("Reshape", [ctm, "shape_merge"], "mrg"), (batch, seq, dim))
    mwo = R(b.matmul(mrg, "wo"), (batch, seq, dim))
    awo = R(b.add(mwo, "bo"), (batch, seq, dim))
    a = R(b.add("x", awo), (batch, seq, dim))
    awu = R(b.matmul(a, "wu"), (batch, seq, ffn))
    u = R(b.add(awu, "bu"), (batch, seq, ffn))
    sig = R(b.sigmoid(u), (batch, seq, ffn))
    g = R(b.mul(u, sig), (batch, seq, ffn))
    gwd = R(b.matmul(g, "wd"), (batch, seq, dim))
    agd = R(b.add(gwd, "bd"), (batch, seq, dim))
    y_hat = R(b.add(a, agd), (batch, seq, out))
    diff = R(b.sub(y_hat, "y"), (batch, seq, out))
    sq = R(b.mul(diff, diff), (batch, seq, out))
    loss = R(b.op("ReduceMean", [sq], "loss", keepdims=0), ())
    fwd_nodes = list(b.nodes)

    seed = b.const(float(loss_scale), hint="loss_scale")
    grads = graph_grad.build_backward(b, fwd_nodes, shapes, {loss: seed}, list(PARAMS))
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
        shape = pshapes[p]
        state[p] = (shape, w_next)
        state[f"m_{p}"] = (shape, m_next)
        state[f"v_{p}"] = (shape, v_next)
    step = qat_graph.make_step_graph(
        b,
        constants={
            "x": ((batch, seq, dim), onnx.TensorProto.FLOAT),
            "y": ((batch, seq, out), onnx.TensorProto.FLOAT),
        },
        state=state,
        scalars=["lr", "m_correction", "v_correction"],
        loss=loss,
    )
    _widen_rank0_inputs(step.model)
    return step, pshapes


def _widen_rank0_inputs(model: onnx.ModelProto) -> None:
    """Reshape rank-0 graph inputs to shape-[1] for the Core ML exporter
    (MIL has no rank-0 Placeholder; broadcast-identical where used)."""
    for vi in model.graph.input:
        if len(vi.type.tensor_type.shape.dim) == 0:
            vi.type.tensor_type.shape.dim.add().dim_value = 1


def make_data(batch: int, seq: int, dim: int, out: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, seq, dim), dtype=np.float32)
    y = rng.standard_normal((batch, seq, out), dtype=np.float32)
    return x, y


def initial_state(
    pshapes: Dict[str, Tuple[int, ...]], seed: int = 1
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = {}
    for p, s in pshapes.items():
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
    init: Dict[str, np.ndarray],
) -> Tuple[List[float], float]:
    import coremltools as ct

    model = ct.models.MLModel(
        mlpackage, compute_units=getattr(ct.ComputeUnit, compute_units)
    )
    loss_name = step.loss_name
    omap = dict(step.state)
    state = {k: v.copy() for k, v in init.items()}
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
        for k, v in omap.items():
            state[k] = out[v]
        losses.append(float(np.asarray(out[loss_name]).reshape(-1)[0]))
    return losses, (time.time() - t0) / num_steps * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="block_step.mlpackage")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--out", type=int, default=512)
    ap.add_argument(
        "--compute-units",
        default="CPU_AND_NE",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    args = ap.parse_args()

    import onnxsim

    step, pshapes = build_block_step(
        args.batch,
        args.seq,
        args.dim,
        args.heads,
        args.head_dim,
        args.ffn,
        args.out,
    )
    print(
        f"step graph: {len(step.model.graph.node)} nodes, "
        f"{len(step.state)} state tensors",
        flush=True,
    )
    onnxsim.export_coreml(step.model, args.output, skip_model_load=True)
    print(f"Wrote {args.output}", flush=True)

    init = initial_state(pshapes)
    x, y = make_data(args.batch, args.seq, args.dim, args.out)
    losses, ms = run_loop(
        args.output,
        step,
        x,
        y,
        args.steps,
        args.lr,
        args.compute_units,
        init,
    )
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {args.steps} steps")
    print(f"{ms:.1f} ms/step ({args.compute_units})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
