#!/usr/bin/env python3
"""Run the ResNet-18 resident training graph entirely on the host.

This is the Pulsar2-free counterpart to the AXCL resident runner.  ONNX
Runtime executes the same step graph; each state output is fed back as the
corresponding state input on the next iteration.  It is useful for validating
the training loop, loss trajectory, and graph I/O before vendor compilation.

Usage::

    run_resnet18_training_host.py OUT_DIR --steps 10 --batch 1 --lr 1.0

The input data is deterministic synthetic data by default.  This keeps the
command self-contained; a real dataset can be wired in by replacing
``make_batch`` without changing the state loop.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resnet18_train_step as builder  # noqa: E402


def make_batch(batch: int, seed: int = 7):
    """Return a deterministic small image batch and one-hot labels."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((batch, 3, 64, 64)) * 0.1).astype(np.float32)
    y = np.zeros((batch, 1000), dtype=np.float32)
    y[np.arange(batch), np.arange(batch) % 1000] = 1.0
    return x, y


def run(out_dir: str, batch: int, steps: int, lr: float, seed: int = 7):
    """Build/load and execute a host-side resident training loop."""
    if steps < 1:
        raise ValueError("steps must be positive")
    step_path = os.path.join(out_dir, f"resnet18_step_b{batch}.onnx")
    if not os.path.exists(step_path):
        step_path, state = builder.build_step(out_dir, batch)
    else:
        state = None

    model = onnx.load(step_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = {item.name for item in session.get_inputs()}
    output_names = [item.name for item in session.get_outputs()]
    state_inputs = list(builder.TRAIN_PARAMS)
    state_outputs = (
        [state[name] for name in state_inputs]
        if state is not None
        else [name for name in output_names if name not in {"loss", "loss_scaled"}]
    )
    missing = [
        name
        for name in state_inputs + state_outputs
        if name not in input_names | set(output_names)
    ]
    if missing:
        raise RuntimeError(f"step graph is missing expected state names: {missing}")

    fwd_path = os.path.join(out_dir, "resnet18d_fwd.onnx")
    if not os.path.exists(fwd_path):
        raise FileNotFoundError(fwd_path)
    fwd = onnx.load(fwd_path)
    initializers = {item.name: item for item in fwd.graph.initializer}
    state_values = {
        name: onnx.numpy_helper.to_array(initializers[name]).copy()
        for name in state_inputs
    }
    x, y = make_batch(batch, seed)

    for step in range(steps):
        feeds = {
            "x": x,
            "y": y,
            "lr": np.array([lr], dtype=np.float32),
            "grad_seed": np.array([1.0], dtype=np.float32),
            **state_values,
        }
        values = dict(zip(output_names, session.run(output_names, feeds)))
        loss = float(np.asarray(values["loss"]).reshape(-1)[0])
        print(f"step={step:04d} loss={loss:.8f}")
        state_values = {
            name: np.asarray(values[next_name]).copy()
            for name, next_name in zip(state_inputs, state_outputs)
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    run(args.out_dir, args.batch, args.steps, args.lr, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
