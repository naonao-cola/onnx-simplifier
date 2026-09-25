"""Graph-independent builder for the currently decoded AX MCode records.

This is an instruction-stream layer, not an ONNX compiler: the record forms
and byte encoding are known, while the arithmetic meaning of most fields is
not.  Kernel emitters can therefore build a validated stream without calling
Pulsar2, then hand it to the model/template packager once the surrounding
model metadata and memory bindings are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import mcode
import onnx


def _byte(value: int, label: str) -> int:
    value = int(value)
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{label} must fit in one byte: {value}")
    return value


@dataclass
class MCodeProgram:
    """A mutable, validated sequence of decoded MCode records."""

    records: list[dict] = field(default_factory=list)

    def verb(self, verb: int, field: int, bank: int, operand: bytes) -> MCodeProgram:
        """Append a normal 8-byte or compact 7-byte verb record."""
        verb = _byte(verb, "verb")
        if verb not in mcode.VERBS6:
            raise ValueError(f"unsupported MCode verb: 0x{verb:02x}")
        field = _byte(field, "field")
        bank = _byte(bank, "bank")
        operand = bytes(operand)
        if len(operand) not in (3, 4):
            raise ValueError("verb operand must contain 3 or 4 bytes")
        self.records.append(
            {"kind": "V", "verb": verb, "field": field, "bank": bank, "operand": operand}
        )
        return self

    def short(
        self,
        prefix_width: int,
        payload: bytes,
        tag: int,
        register: int,
        extra: bytes = b"",
    ) -> MCodeProgram:
        """Append a width-rule short unit (``S`` record)."""
        prefix_width = int(prefix_width)
        if prefix_width < 0 or prefix_width > 4:
            raise ValueError("short-unit width must be between 0 and 4")
        payload = bytes(payload)
        if len(payload) != prefix_width + 1:
            raise ValueError("short-unit payload length must equal width + 1")
        tag = _byte(tag, "short-unit tag")
        register = _byte(register, "short-unit register")
        extra = bytes(extra)
        if tag == 0x9F and len(extra) != 1:
            raise ValueError("tag 0x9f requires one extra byte")
        if tag != 0x9F and extra:
            raise ValueError("only tag 0x9f supports an extra byte")
        self.records.append(
            {
                "kind": "S",
                "p": prefix_width,
                "payload": payload,
                "tag": tag,
                "reg": register,
                "extra": extra,
            }
        )
        return self

    def bare(self, tag: int, register: int, extra: bytes = b"") -> MCodeProgram:
        """Append a payload-less bare pair (``B`` record)."""
        tag = _byte(tag, "bare-pair tag")
        register = _byte(register, "bare-pair register")
        extra = bytes(extra)
        if extra:
            raise ValueError("bare pairs do not accept extra bytes")
        self.records.append({"kind": "B", "tag": tag, "reg": register, "extra": b""})
        return self

    def raw(self, *values: int) -> MCodeProgram:
        """Append bytes that have not yet been assigned a decoded form."""
        self.records.extend({"kind": "raw", "byte": _byte(v, "raw byte")} for v in values)
        return self

    def extend(self, records: Iterable[dict]) -> MCodeProgram:
        """Append already-decoded records, preserving their field structure."""
        self.records.extend(dict(record) for record in records)
        return self

    def encode(self) -> bytes:
        """Encode the program and reject malformed records before emission."""
        for record in self.records:
            if not isinstance(record, dict) or "kind" not in record:
                raise ValueError("every MCode record must be a dictionary with kind")
        return mcode.encode(self.records)


def replace_template_segment(template: bytes, segment: int, program: MCodeProgram) -> bytes:
    """Replace one compressed segment in an AX MCode image.

    ``MCodeProgram`` produces the decoded instruction bytes.  The AX image
    stores those bytes in the repository's short-unit codec, so this helper
    compresses the new stream and delegates header/tail relocation to the
    already validated relayout implementation.  The template's opaque
    container metadata remains intact.
    """
    import short_unit_codec
    import step_recalibrate

    compressed = short_unit_codec.encode(program.encode())
    result = step_recalibrate.relayout(template, segment, compressed)
    violations = mcode.check(result)
    if violations:
        raise ValueError("generated MCode failed structural validation: " + violations[0])
    return result


def replace_model_segment(
    model: onnx.ModelProto,
    mcode_initializer: str,
    segment: int,
    program: MCodeProgram,
) -> onnx.ModelProto:
    """Return ``model`` with one generated segment packaged into its MCode.

    Only the MCode blob and its length declarations are changed.  Quantization
    tables, graph metadata, and opaque vendor attributes are copied exactly;
    callers remain responsible for proving that the program's register
    bindings match those metadata.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializer = next(
        (item for item in out.graph.initializer if item.name == mcode_initializer),
        None,
    )
    if initializer is None:
        raise ValueError(f"MCode initializer not found: {mcode_initializer!r}")
    patched = replace_template_segment(bytes(initializer.raw_data), segment, program)
    initializer.raw_data = patched
    del initializer.dims[:]
    initializer.dims.append(len(patched))
    declarations = [*out.graph.value_info, *out.graph.input]
    for value in declarations:
        if value.name != mcode_initializer:
            continue
        dims = value.type.tensor_type.shape.dim
        if len(dims) != 1:
            raise ValueError(f"MCode declaration is not one-dimensional: {value.name!r}")
        dims[0].dim_value = len(patched)
    return out
