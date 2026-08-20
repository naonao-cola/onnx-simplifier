#!/usr/bin/env python3
"""Write synthetic (input, target) raw float32 binaries for the toy model.

target = sum(input) (a trivial function the toy MLP can actually learn),
so a working onnx-finetune run should show the loss drop substantially.
"""
import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input-dim", type=int, default=4)
p.add_argument("--output-dim", type=int, default=1)
p.add_argument("--num-samples", type=int, default=2048)
p.add_argument("--input-out", default="train_input.bin")
p.add_argument("--target-out", default="train_target.bin")
args = p.parse_args()

rng = np.random.default_rng(1)
x = rng.standard_normal((args.num_samples, args.input_dim)).astype(np.float32)
y = np.sum(x, axis=1, keepdims=True).astype(np.float32)
if args.output_dim > 1:
    y = np.tile(y, (1, args.output_dim))

x.tofile(args.input_out)
y.tofile(args.target_out)
print(f"wrote {args.num_samples} samples -> {args.input_out}, {args.target_out}")
