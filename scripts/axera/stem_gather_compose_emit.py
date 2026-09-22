"""Emitter for the ResNet18 stem Gather, composed with its real graph
neighbours, at real scale.

`docs/axera-stem-gather.md` built a standalone (no consumer) 7-chunk
`Gather`+`Concat` template for the stem's `[16,1,3,50176] -> [16,1,3,614656]`
im2col unroll, because Pulsar2 rejects the single-`Gather` form as one
on-chip-memory job. `docs/axera-gather-compose-real-scale.md` then found that
attaching the real `Mul`-by-padding-mask consumer *after* that `Concat`, at
full scale, hit a second, unrelated wall: the elementwise engine addresses a
tensor's last axis with a 16-bit offset (a hard limit at 65,536 elements,
confirmed in `docs/axera-stem-gather-rechunk.md`), so a single `Mul` over the
full 614,656-element concatenated tensor cannot be lowered at all, and only a
32,768-element slice (no `Concat`, no full scale) was verified.

This module is the template that resolves that: instead of `Gather*7 ->
Concat -> Mul`, the graph is `(Gather -> Mul)*14 -> Concat`, splitting into 14
chunks of 43,904 indices each (`614656 = 2^8*7^4`; 43,904 is the largest
divisor of 614,656 that is <= 65,536, so it both tiles evenly and keeps every
`Mul`'s addressed axis under the limit). It compiles -- slowly, about 79
minutes, one compiler stage (`calc output dependencies`) taking 49m51s of
that -- and was verified correct on the AX8850 at full 614,656-element scale,
including index retargeting. See `docs/axera-stem-gather-rechunk.md` for the
build/verification record.

The `npu_params` layout is unchanged from the standalone template: the first
614,656 little-endian uint32 words are the (now 14-chunk) index vector, in
order, followed by a tail. `memory_emit.py`'s existing index-retargeting
operation (rewrite the leading N words, keep everything else) applies
unmodified; this module only adds a validated entry point scoped to this one
graph shape, mirroring `gather_compose_real_check.py`'s `patch_indices` (which
it reuses) with a structural check specific to the 14-chunk template.
"""

from __future__ import annotations

import gzip
import os
import struct
from collections.abc import Sequence

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.join(
    _HERE, "fixtures", "gather_compose_real", "stem_rechunk14_reference.axmodel.gz"
)

_CHUNK = 43904
_CHUNKS = 14
_TOTAL = _CHUNK * _CHUNKS
assert _TOTAL == 614656


def _load_template() -> onnx.ModelProto:
    with gzip.open(_TEMPLATE, "rb") as f:
        return onnx.load_model_from_string(f.read())


def _validate(model: onnx.ModelProto) -> None:
    # This is a compiled model: Pulsar2 fuses the whole 29-node ONNX graph
    # (14 Gather + 14 Mul + 1 Concat) into one 'neu mode' NPU node, the same
    # way every other emitter in this project validates its reference.
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("reference must contain exactly one compiled 'neu mode' node")
    if list(model.graph.node[0].input) != ["x"] or list(model.graph.node[0].output) != [
        "y"
    ]:
        raise ValueError("reference NPU node must map input 'x' to output 'y'")
    in_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    if in_shape != [16, 1, 3, 50176] or out_shape != [16, 1, 3, _TOTAL]:
        raise ValueError(f"unexpected shapes: in={in_shape} out={out_shape}")
    table = next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
    if len(table) % 4 or len(table) // 4 < _TOTAL:
        raise ValueError(f"npu_params too small for {_TOTAL} index words")


def emit(output_path: str, *, indices: Sequence[int]) -> str:
    """Retarget the composed stem Gather+Mul+Concat template to new indices.

    ``indices`` must have exactly 614,656 entries in ``[0, 50176)`` (the stem
    im2col's index count and source width). Only the leading index words of
    ``npu_params`` are rewritten; the MCode, the 14 mask constants, and the
    quantization scales are kept from the reference -- the same operation
    ``memory_emit.py``'s single-Gather emitters perform, applied here to the
    validated 14-chunk composed template.

    This was verified correct on-device for in-range inputs (the reference's
    own calibration range, uniform [-2.5, 2.5]) with a completely different
    (shuffled) index vector: retargeting produced the same error as the
    reference computing its own indices, because the real consumer is a
    ``{0,1}``-mask ``Mul``, which cannot expand the value range past the
    Gather's own output (see ``docs/axera-stem-gather-rechunk.md``). It was
    **not** verified safe for an aggregating consumer (``MatMul``, ``Conv``,
    a reduction) -- ``docs/axera-compose.md``'s calibration-range warning
    still applies there.
    """
    target = list(indices)
    if len(target) != _TOTAL:
        raise ValueError(
            f"indices must have exactly {_TOTAL} entries, got {len(target)}"
        )
    if any(not isinstance(v, int) or isinstance(v, bool) for v in target):
        raise ValueError("indices must contain integers")
    if any(not 0 <= v < 50176 for v in target):
        raise ValueError("indices must be in [0, 50176)")

    model = _load_template()
    _validate(model)
    table = next(i for i in model.graph.initializer if i.name == "npu_params")
    words = list(struct.unpack(f"<{len(table.raw_data) // 4}I", bytes(table.raw_data)))
    words[:_TOTAL] = target
    table.raw_data = struct.pack(f"<{len(words)}I", *words)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
