#!/usr/bin/env python3
"""Find quantization redundancy that is safe to act on before compilation.

The AX8850 compiler may insert QDQ around lowered kernels, so this utility
operates on the ONNX graph we control and is also useful on a compiler-dumped
graph.  It deliberately treats a ``QuantizeLinear -> DequantizeLinear`` pair
as *not* removable: rounding and clipping are observable even when the scale
and zero point match.  Only byte-for-byte identical Q/DQ computations are
safe to share.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import onnx


def _node_key(node: onnx.NodeProto) -> tuple[Any, ...]:
    return (
        node.domain,
        node.op_type,
        tuple(node.input),
        tuple(
            (attribute.name, attribute.SerializeToString())
            for attribute in sorted(node.attribute, key=lambda item: item.name)
        ),
    )


def analyze(model: onnx.ModelProto) -> dict[str, Any]:
    """Return safe duplicate candidates and quantization diagnostics.

    ``safe_duplicate_removals`` is the number of duplicate nodes that can be
    removed by redirecting their consumers to the first identical node.  The
    returned report is JSON serializable and deterministic for review/CI.
    """
    quantizers = [
        node
        for node in model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    ]
    groups: dict[tuple[Any, ...], list[onnx.NodeProto]] = defaultdict(list)
    for node in quantizers:
        if len(node.output) == 1 and all(node.input):
            groups[_node_key(node)].append(node)

    duplicate_groups = []
    for nodes in groups.values():
        if len(nodes) < 2:
            continue
        duplicate_groups.append(
            {
                "op_type": nodes[0].op_type,
                "canonical": nodes[0].output[0],
                "duplicates": [node.output[0] for node in nodes[1:]],
                "inputs": list(nodes[0].input),
            }
        )
    duplicate_groups.sort(key=lambda item: item["canonical"])

    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
    roundtrips = []
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.output) != 1:
            continue
        users = consumers.get(node.output[0], [])
        if len(users) != 1 or users[0].op_type != "DequantizeLinear":
            continue
        dequant = users[0]
        if len(node.input) < 3 or len(dequant.input) < 3:
            continue
        if tuple(node.input[1:]) != tuple(dequant.input[1:]):
            continue
        roundtrips.append(
            {
                "quantize": node.name or node.output[0],
                "dequantize": dequant.name or dequant.output[0],
                "reason": "rounding_or_clipping_changes_values",
            }
        )

    return {
        "quantize_nodes": sum(node.op_type == "QuantizeLinear" for node in quantizers),
        "dequantize_nodes": sum(
            node.op_type == "DequantizeLinear" for node in quantizers
        ),
        "duplicate_groups": duplicate_groups,
        "safe_duplicate_removals": sum(
            len(group["duplicates"]) for group in duplicate_groups
        ),
        "unsafe_roundtrips": roundtrips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="ONNX model to inspect")
    args = parser.parse_args()
    print(json.dumps(analyze(onnx.load(args.model)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
