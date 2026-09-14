#!/usr/bin/env python3
"""Parametric Edge TPU peak-performance benchmark models (pure onnx+numpy).

A small suite of synthetic graphs spanning the roofline from link-bound to
compute-bound, built to be compiled with ``peak_benchmark.py`` (int8/uint8,
``io_layout="nhwc"``) and timed on-device with ``benchmark_device.py``:

- ``pointwise-8`` / ``pointwise-48``: 1x1 conv stacks (highest MAC/byte --
  the closest any model gets to the 4 TOPS peak; the 48-layer one is the
  compute-bound probe at ~0.8 GMAC with params still on-chip).
- ``dense3x3-6``: 3x3 dense conv stack.
- ``mbblock-4``: MobileNet-style inverted residuals (depthwise-heavy,
  realistic mobile workload).
- ``fc-4k``: flatten + large Gemm (memory-bound anchor).
- ``cliff-64x32``: the 64ch x 32x32 single conv that refuses compilation
  with the NCHW entry transpose but maps fully channel-last.

Conv weights use He scaling so activations stay bounded through deep stacks
(like trained nets); unscaled random weights explode and break int8
quantization mid-graph, which would benchmark the converter's float fallback
instead of the TPU.

``gmac`` counts multiply-accumulates, exact by construction (stride-1,
same-padded convs, no branching): ``cout * h * w * cin * k * k`` per conv,
``m * n * k`` per Gemm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_OPSET = 17
_IR = 8


@dataclass
class BenchmarkModel:
    """One benchmark workload: the graph, its exact MAC count and I/O shapes."""

    name: str
    model: onnx.ModelProto
    gmac: float
    input_shape: List[int]  # NCHW public order
    param_bytes: int = 0
    output_shape: List[int] = field(default_factory=list)  # NCHW order
    io_bytes: int = 0  # uint8 input + output elements (steady state)


def _rand(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def _checked(model: onnx.ModelProto) -> onnx.ModelProto:
    onnx.checker.check_model(model)
    return model


class _Builder:
    """Accumulates nodes/initializers/MACs for one chain-structured model."""

    def __init__(self, seed: int = 0):
        self.nodes: list = []
        self.initializers: list = []
        self.macs = 0
        self.rng = np.random.RandomState(seed)

    def conv(
        self,
        x: str,
        *,
        cin: int,
        cout: int,
        h: int,
        k: int = 3,
        group: int = 1,
        bias: bool = True,
        tag: str,
    ) -> str:
        """Append Conv(+bias) + Relu; returns the output name."""
        scale = (2.0 / ((cin // group) * k * k)) ** 0.5
        w = numpy_helper.from_array(
            (self.rng.randn(cout, cin // group, k, k) * np.float32(scale)).astype(
                np.float32
            ),
            name=f"w{tag}",
        )
        self.initializers.append(w)
        inputs = [x, f"w{tag}"]
        if bias:
            b = numpy_helper.from_array(np.zeros(cout, np.float32), name=f"b{tag}")
            self.initializers.append(b)
            inputs.append(f"b{tag}")
        pad = k // 2
        self.nodes.append(
            helper.make_node(
                "Conv",
                inputs,
                [f"t{tag}"],
                kernel_shape=[k, k],
                pads=[pad, pad, pad, pad],
                group=group,
            )
        )
        self.nodes.append(helper.make_node("Relu", [f"t{tag}"], [f"p{tag}"]))
        self.macs += cout * h * h * (cin // group) * k * k
        return f"p{tag}"

    def finish(
        self, name: str, input_shape: List[int], output: str, out_shape: List[int]
    ) -> BenchmarkModel:
        nodes = self.nodes + [helper.make_node("Identity", [output], ["y"])]
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)]
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)]
        graph = helper.make_graph(nodes, name, inputs, outputs, self.initializers)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", _OPSET)], ir_version=_IR
        )
        param_bytes = sum(int(numpy_helper.to_array(t).size) for t in self.initializers)
        return BenchmarkModel(
            name=name,
            model=_checked(model),
            gmac=self.macs / 1e9,
            input_shape=input_shape,
            param_bytes=param_bytes,
            output_shape=out_shape,
            io_bytes=int(np.prod(input_shape, dtype=np.int64))
            + int(np.prod(out_shape, dtype=np.int64)),
        )


def _pointwise(channels: int, size: int, layers: int, name: str) -> BenchmarkModel:
    b = _Builder(seed=0)
    prev = "x"
    for i in range(layers):
        prev = b.conv(prev, cin=channels, cout=channels, h=size, k=1, tag=f"{name}{i}")
    return b.finish(name, [1, channels, size, size], prev, [1, channels, size, size])


def pointwise_8() -> BenchmarkModel:
    """8x pointwise 256ch/16px (link-bound reference)."""
    return _pointwise(256, 16, 8, "pointwise-8")


def dense3x3_6() -> BenchmarkModel:
    """6x dense 3x3 128ch/16px (approaches compute-bound on USB3)."""
    b = _Builder(seed=0)
    prev = "x"
    for i in range(6):
        prev = b.conv(prev, cin=128, cout=128, h=16, k=3, tag=f"d{i}")
    return b.finish("dense3x3-6", [1, 128, 16, 16], prev, [1, 128, 16, 16])


def mbblock_4() -> BenchmarkModel:
    """4x MobileNet-style inverted residuals at 32x32 (realistic workload)."""
    b = _Builder(seed=0)
    c, h = 32, 32
    prev = "x"
    for i in range(4):
        e = c * 4
        prev = b.conv(prev, cin=c, cout=e, h=h, k=1, tag=f"e{i}")
        prev = b.conv(prev, cin=e, cout=e, h=h, k=3, group=e, bias=False, tag=f"dw{i}")
        prev = b.conv(prev, cin=e, cout=c, h=h, k=1, tag=f"p{i}")
    return b.finish("mbblock-4", [1, c, h, h], prev, [1, c, h, h])


def fc_4k() -> BenchmarkModel:
    """Flatten + 4096x1024 Gemm (memory-bound anchor)."""
    rng = np.random.RandomState(0)
    gw = numpy_helper.from_array(rng.randn(1024, 4096).astype(np.float32), name="gw")
    gb = numpy_helper.from_array(rng.randn(1024).astype(np.float32), name="gb")
    nodes = [
        helper.make_node("Flatten", ["x"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "gw", "gb"], ["y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "fc-4k",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1024])],
        [gw, gb],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], ir_version=_IR
    )
    return BenchmarkModel(
        name="fc-4k",
        model=_checked(model),
        gmac=1024 * 4096 / 1e9,
        input_shape=[1, 64, 8, 8],
        param_bytes=1024 * 4096 + 1024,
        output_shape=[1, 1024],
        io_bytes=4096 + 1024,
    )


def cliff_64x32() -> BenchmarkModel:
    """Single 64ch x 32x32 conv: fails NCHW compilation, maps fully as NHWC."""
    b = _Builder(seed=0)
    prev = b.conv("x", cin=64, cout=64, h=32, k=3, tag="cliff")
    return b.finish("cliff-64x32", [1, 64, 32, 32], prev, [1, 64, 32, 32])


def pointwise_48() -> BenchmarkModel:
    """48x pointwise 256ch/16px (~0.8 GMAC, params still on-chip): the
    compute-bound probe that should approach the sustained ceiling."""
    return _pointwise(256, 16, 48, "pointwise-48")


def all_models() -> Dict[str, BenchmarkModel]:
    builders = [pointwise_8, dense3x3_6, mbblock_4, fc_4k, cliff_64x32, pointwise_48]
    return {m.name: m for m in (fn() for fn in builders)}


if __name__ == "__main__":
    for name, m in all_models().items():
        print(
            f"{name:14s} {m.gmac:7.3f} GMAC  in={m.input_shape}  "
            f"out={m.output_shape}  params={m.param_bytes / 1e6:.2f}MB  "
            f"io={m.io_bytes / 1e3:.1f}KB  nodes={len(m.model.graph.node)}"
        )
