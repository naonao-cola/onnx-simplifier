"""Pulsar-free generation from a validated same-topology AX template.

The AX compiler emits a fused ``neu mode`` program, so an ONNX graph cannot be
reconstructed from the compiled model alone.  This module makes the safe
reuse boundary explicit: retain the source graph beside one validated AX
template, compare structural signatures, and copy the template for any
same-topology request.  Shape, IO, node attributes, and initializer layout
must match; tensor values may differ only when a higher-level emitter has a
validated patch for them.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any

import onnx


@dataclasses.dataclass(frozen=True)
class GraphSignature:
    inputs: tuple[tuple[str, int, tuple[int, ...]], ...]
    outputs: tuple[tuple[str, int, tuple[int, ...]], ...]
    nodes: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[tuple[str, Any], ...]], ...]
    initializers: tuple[tuple[str, int, tuple[int, ...]], ...]


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(int(dim.dim_value) for dim in value.type.tensor_type.shape.dim)


def _attrs(node: onnx.NodeProto) -> tuple[tuple[str, Any], ...]:
    out = []
    for attr in node.attribute:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif hasattr(value, "tolist"):
            value = value.tolist()
        elif isinstance(value, list):
            value = tuple(value)
        out.append((attr.name, value))
    return tuple(sorted(out))


def signature(model: onnx.ModelProto) -> GraphSignature:
    """Return a value-independent structural signature for an ONNX graph."""
    return GraphSignature(
        tuple((v.name, v.type.tensor_type.elem_type, _shape(v)) for v in model.graph.input),
        tuple((v.name, v.type.tensor_type.elem_type, _shape(v)) for v in model.graph.output),
        tuple(
            (n.op_type, tuple(n.input), tuple(n.output), _attrs(n))
            for n in model.graph.node
        ),
        tuple((i.name, i.data_type, tuple(i.dims)) for i in model.graph.initializer),
    )


def _io_signature(model: onnx.ModelProto) -> tuple[tuple, tuple]:
    """Return the compiled model's externally visible IO contract."""
    inputs = tuple(
        sorted((v.name, v.type.tensor_type.elem_type, _shape(v)) for v in model.graph.input)
    )
    outputs = tuple(
        sorted((v.name, v.type.tensor_type.elem_type, _shape(v)) for v in model.graph.output)
    )
    return inputs, outputs


def _load(path: str) -> onnx.ModelProto:
    return onnx.load(path, load_external_data=False)


def generate(
    source_path: str,
    template_source_path: str,
    template_axmodel_path: str,
    output_path: str,
) -> GraphSignature:
    """Emit ``output_path`` from a validated template without Pulsar2."""
    source = _load(source_path)
    template_source = _load(template_source_path)
    template = _load(template_axmodel_path)
    expected = signature(template_source)
    actual = signature(source)
    if actual != expected:
        raise ValueError("source graph does not match the validated AX template topology")
    if _io_signature(template) != _io_signature(template_source):
        raise ValueError("compiled AX template IO does not match its source graph")
    if len(template.graph.node) != 1 or template.graph.node[0].op_type != "neu mode":
        raise ValueError("template must contain exactly one fused neu mode node")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(template, output_path)
    return actual


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("template_source")
    parser.add_argument("template_axmodel")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    generate(args.source, args.template_source, args.template_axmodel, args.output)
    print(f"generated={args.output} pulsar2=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
