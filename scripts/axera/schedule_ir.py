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
from math import prod

import graph_generator
import onnx


@dataclasses.dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    elem_type: int
    kind: str
    nbytes: int


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    name: str
    chain: str
    inputs: tuple[str, ...]
    output: str
    template: str


@dataclasses.dataclass(frozen=True)
class Allocation:
    name: str
    offset: int
    nbytes: int
    first_kernel: int
    last_kernel: int


@dataclasses.dataclass(frozen=True)
class ScheduleIR:
    inputs: tuple[BufferSpec, ...]
    outputs: tuple[BufferSpec, ...]
    kernels: tuple[KernelSpec, ...]
    dependencies: tuple[tuple[str, str], ...]
    allocations: tuple[Allocation, ...]
    memory_size: int

    def to_json(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["schema_version"] = 1
        return payload


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims = value.type.tensor_type.shape.dim
    shape = tuple(int(dim.dim_value) for dim in dims)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"schedule requires a static shape for {value.name!r}")
    return shape


def _nbytes(shape: tuple[int, ...], elem_type: int) -> int:
    try:
        dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"schedule has unsupported tensor type {elem_type}") from error
    return prod(shape) * dtype.itemsize


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def _allocate(
    buffers: Mapping[str, BufferSpec],
    kernels: tuple[KernelSpec, ...],
    input_names: set[str],
    output_names: set[str],
) -> tuple[tuple[Allocation, ...], int]:
    """Assign reusable aligned storage using kernel use lifetimes."""
    uses = {
        name: [index for index, kernel in enumerate(kernels) if name in kernel.inputs]
        for name in buffers
    }
    producers = {kernel.output: index for index, kernel in enumerate(kernels)}
    lifetimes = []
    for name, buffer in buffers.items():
        references = uses[name]
        if name in input_names:
            first = 0
        elif name in producers:
            first = producers[name]
        else:
            continue
        last = max(references or [first])
        if name in output_names:
            last = max(last, len(kernels) - 1)
        lifetimes.append((first, last, name, buffer.nbytes))

    allocations: list[Allocation] = []
    active: list[Allocation] = []
    free: list[tuple[int, int]] = []
    cursor = 0
    for first, last, name, nbytes in sorted(lifetimes):
        still_active = []
        for allocation in active:
            if allocation.last_kernel < first:
                free.append((allocation.offset, _align(allocation.nbytes)))
            else:
                still_active.append(allocation)
        active = still_active
        size = _align(nbytes)
        free.sort(key=lambda item: (item[1], item[0]))
        slot = next(
            (
                (index, offset, available)
                for index, (offset, available) in enumerate(free)
                if available >= size
            ),
            None,
        )
        if slot is None:
            offset = _align(cursor)
            cursor = offset + size
        else:
            index, offset, available = slot
            del free[index]
            if available > size:
                free.append((offset + size, available - size))
        allocation = Allocation(name, offset, nbytes, first, last)
        allocations.append(allocation)
        active.append(allocation)
    return tuple(allocations), _align(cursor)


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
    buffers = {
        value.name: BufferSpec(
            value.name,
            _shape(value),
            value.type.tensor_type.elem_type,
            (
                "input"
                if value.name in input_names
                else "output"
                if value.name in output_names
                else "intermediate"
            ),
            _nbytes(_shape(value), value.type.tensor_type.elem_type),
        )
        for value in values.values()
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
    known_buffers = set(buffers)
    produced: dict[str, str] = {}
    dependencies: list[tuple[str, str]] = []
    for kernel in kernels:
        if kernel.output in produced:
            raise ValueError(f"schedule has multiple producers for {kernel.output!r}")
        for input_name in kernel.inputs:
            if input_name not in known_buffers and input_name not in produced:
                raise ValueError(
                    f"kernel {kernel.name!r} uses unknown buffer {input_name!r}"
                )
            producer = produced.get(input_name)
            if producer is not None:
                dependencies.append((producer, kernel.name))
        produced[kernel.output] = kernel.name
    for output in model.graph.output:
        if output.name not in produced and output.name not in input_names:
            raise ValueError(f"schedule output {output.name!r} has no producer")
    allocations, memory_size = _allocate(buffers, kernels, input_names, output_names)
    return ScheduleIR(
        tuple(buffers[item.name] for item in model.graph.input),
        tuple(buffers[item.name] for item in model.graph.output),
        kernels,
        tuple(dependencies),
        allocations,
        memory_size,
    )


def write(source_path: str, output_path: str) -> ScheduleIR:
    model = onnx.load(source_path, load_external_data=False)
    schedule = build(model)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(schedule.to_json(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return schedule
