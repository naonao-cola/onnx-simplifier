#!/usr/bin/env python3
"""Generates the before/after ONNX fixture pair `renesas-integration.yml`'s
`real-tvm-v08-frontend` job feeds to the real TVM v0.8 ONNX frontend.

Why a separate script + CI job instead of running `onnxsim.simplify()`
directly inside the TVM-build job: TVM v0.8 needs Python <=3.8
(`docs/install/from_source.rst` in apache/tvm's own `v0.8` tag: "Avoid using
Python 3.9.X+ which is not supported"), while this repo's own
`pyproject.toml` requires Python >=3.11 -- the two cannot be installed into
one interpreter. So `renesas-integration.yml` runs this script in a normal
(Python >=3.11) job to produce plain `.onnx` files, uploads them as a build
artifact, and only the *files* cross into the separate Python-3.8 job that
builds real TVM v0.8 and imports them with `tvm.relay.frontend.from_onnx()`
-- that job never needs onnxsim itself, only the public `onnx` package.

The model is a small `Conv -> BatchNormalization -> Relu` graph -- the same
shape of case `scripts/axelera/voyager_backend.py`'s docstring reports
onnxsim folding into `Conv -> Relu` (BN fused into Conv's weights) when
checked against Axelera's real quantizer. Doing the equivalent check here
against real TVM v0.8 confirms the same onnxsim transformation doesn't
break DRP-AI TVM's ONNX import step either -- both op_types are in
`DRP_AI_TVM_IMPORTABLE_OPS` before *and* after, so this is mainly a sanity
check that the simplified graph is still well-formed from TVM's frontend's
point of view, not a coverage change (`BatchNormalization` was already
importable).
"""

import argparse
import os

import numpy as np
import onnx
from onnx import numpy_helper, parser

import onnxsim


def _conv_bn_relu_model() -> onnx.ModelProto:
    model = parser.parse_model("""
        <ir_version: 10, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y)
        {
          c = Conv<kernel_shape = [3, 3]>(x, w)
          b = BatchNormalization(c, scale, bias, mean, var)
          y = Relu(b)
        }
        """)
    rng = np.random.default_rng(0)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                rng.standard_normal((4, 3, 3, 3)).astype(np.float32), name="w"
            ),
            numpy_helper.from_array(np.ones(4, np.float32), name="scale"),
            numpy_helper.from_array(np.zeros(4, np.float32), name="bias"),
            numpy_helper.from_array(
                rng.standard_normal(4).astype(np.float32), name="mean"
            ),
            numpy_helper.from_array(
                np.abs(rng.standard_normal(4)).astype(np.float32) + 0.1, name="var"
            ),
        ]
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    before = _conv_bn_relu_model()
    onnx.save(before, os.path.join(args.output_dir, "conv_bn_relu_before.onnx"))

    after, ok = onnxsim.simplify(before)
    assert ok, "onnxsim.simplify() reported failure on the fixture model"
    onnx.save(after, os.path.join(args.output_dir, "conv_bn_relu_after.onnx"))

    before_ops = sorted(n.op_type for n in before.graph.node)
    after_ops = sorted(n.op_type for n in after.graph.node)
    print(f"before: {before_ops}")
    print(f"after:  {after_ops}")
    assert "BatchNormalization" not in after_ops, (
        "expected onnxsim to fold BatchNormalization into the preceding "
        f"Conv, but it's still present: {after_ops}"
    )


if __name__ == "__main__":
    main()
