"""Small, explicit schedule IR for the Pulsar-free AX graph path.

This is deliberately a control-plane replacement, not an invented MCode
format.  It records the tensors, fused template kernels, and dependencies that
the measured graph scheduler has actually validated.  A later emitter can
lower this IR to AXCL model buffers and command queues; unsupported topology
or missing static shapes is rejected before any device work is attempted.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping

import graph_generator
import onnx


@dataclasses.dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    elem_type: int
    kind: str


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    name: str
    chain: str
    inputs: tuple[str, ...]
    output: str
    template: str


@dataclasses.dataclass(frozen=True)
class ScheduleIR:
    inputs: tuple[BufferSpec, ...]
    outputs: tuple[BufferSpec, ...]
    kernels: tuple[KernelSpec, ...]

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims = value.type.tensor_type.shape.dim
    shape = tuple(int(dim.dim_value) for dim in dims)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"schedule requires a static shape for {value.name!r}")
    return shape


def _values(model: onnx.ModelProto) -> Mapping[str, onnx.ValueInfoProto]:
    return {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }


def build(model: onnx.ModelProto) -> ScheduleIR:
    """Build the validated schedule IR for one measured fused graph."""
    plan = graph_generator.schedule_graph(model)
    values = _values(model)
    input_names = {item.name for item in model.graph.input}
    output_names = {item.name for item in model.graph.output}
    io_names = input_names | output_names
    buffers = {
        value.name: BufferSpec(
            value.name,
            _shape(value),
            value.type.tensor_type.elem_type,
            "input" if value.name in input_names else "output",
        )
        for value in values.values()
        if value.name in io_names
    }
    kernels = tuple(
        KernelSpec(
            f"kernel_{index}",
            segment.chain,
            segment.inputs,
            segment.output,
            segment.chain,
        )
        for index, segment in enumerate(plan.segments)
    )
    if not kernels:
        raise ValueError("schedule contains no executable kernels")
    return ScheduleIR(
        tuple(buffers[item.name] for item in model.graph.input),
        tuple(buffers[item.name] for item in model.graph.output),
        kernels,
    )


def write(source_path: str, output_path: str) -> ScheduleIR:
    model = onnx.load(source_path, load_external_data=False)
    schedule = build(model)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(schedule.to_json(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return schedule
