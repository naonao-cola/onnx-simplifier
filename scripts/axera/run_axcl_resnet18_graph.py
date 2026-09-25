#!/usr/bin/env python3
"""Run a compiled resident ResNet18 training graph through AXCL.

The model is expected to expose ``x``, ``y``, ``lr``, ``grad_seed`` and the
four trainable state inputs used by ``build_resnet18_train_step.py``.  State
outputs are fed back by output order, matching the host resident runner.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import axcl_session  # noqa: E402
import build_resnet18_train_step as builder  # noqa: E402
from run_resnet18_training_host import make_batch  # noqa: E402


def _initial_state(forward_path: str) -> dict[str, np.ndarray]:
    forward = onnx.load(forward_path)
    initializers = {item.name: onnx.numpy_helper.to_array(item) for item in forward.graph.initializer}
    missing = [name for name in builder.TRAIN_PARAMS if name not in initializers]
    if missing:
        raise ValueError(f"forward graph is missing trainable state: {missing}")
    return {name: initializers[name].copy() for name in builder.TRAIN_PARAMS}


def run(model_path: str, forward_path: str, steps: int, lr: float, seed: int) -> None:
    model_onnx = onnx.load(model_path)
    x_shape = next(i for i in model_onnx.graph.input if i.name == "x").type.tensor_type.shape
    batch = x_shape.dim[0].dim_value
    x, y = make_batch(batch, seed)
    state = _initial_state(forward_path)
    initial_state = {name: value.copy() for name, value in state.items()}

    with axcl_session.AXSession() as session:
        model = session.load(model_path)
        try:
            input_names = [item.name for item in model.inputs]
            output_names = [item.name for item in model.outputs]
            state_outputs = [name for name in output_names if name != "loss"]
            if len(state_outputs) != len(builder.TRAIN_PARAMS):
                raise ValueError(
                    f"expected {len(builder.TRAIN_PARAMS)} state outputs, got {state_outputs}"
                )

            def feeds():
                values = {
                    "x": x,
                    "y": y,
                    "lr": np.array([lr], dtype=np.float32),
                    "grad_seed": np.array([1.0], dtype=np.float32),
                    **state,
                }
                return [np.asarray(values[name], dtype=np.float32) for name in input_names]

            for _ in range(1):
                values = dict(zip(output_names, session.run(model, feeds())))
                state = {
                    name: np.asarray(values[out_name]).copy()
                    for name, out_name in zip(builder.TRAIN_PARAMS, state_outputs)
                }
            initial_state = {name: value.copy() for name, value in state.items()}

            start = time.perf_counter()
            losses = []
            for _ in range(steps):
                values = dict(zip(output_names, session.run(model, feeds())))
                losses.append(float(np.asarray(values["loss"]).reshape(-1)[0]))
                state = {
                    name: np.asarray(values[out_name]).copy()
                    for name, out_name in zip(builder.TRAIN_PARAMS, state_outputs)
                }
            elapsed = time.perf_counter() - start
            print(f"batch={batch} steps={steps} total_s={elapsed:.6f} step_ms={elapsed * 1000 / steps:.3f}")
            print(f"loss_first={losses[0]:.10f} loss_last={losses[-1]:.10f}")
            state_delta = max(
                float(np.max(np.abs(state[name] - initial_state[name])))
                for name in builder.TRAIN_PARAMS
            )
            print(f"max_state_delta={state_delta:.8e}")
            print(f"axcl_exec_ms={session.exec_us / 1000:.3f}")
        finally:
            session.unload(model)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("forward")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    run(args.model, args.forward, args.steps, args.lr, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
