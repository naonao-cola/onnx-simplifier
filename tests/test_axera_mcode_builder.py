import os
import sys

import pytest

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import mcode_builder  # noqa: E402


def test_builder_emits_decoded_record_stream():
    program = (
        mcode_builder.MCodeProgram()
        .verb(0xA1, 0x40, 0x02, b"\x01\x02\x03\x04")
        .short(1, b"\x98\x02", 0x83, 0x40)
        .bare(0x83, 0x0E)
        .raw(0)
    )
    assert program.encode() == bytes.fromhex("a1 00 40 02 01 02 03 04 01 98 02 83 40 83 0e 00")


def test_builder_rejects_invalid_widths_and_operands():
    with pytest.raises(ValueError, match="operand"):
        mcode_builder.MCodeProgram().verb(0xA1, 0, 0, b"\x00")
    with pytest.raises(ValueError, match="payload length"):
        mcode_builder.MCodeProgram().short(1, b"\x00", 0x83, 0)
    with pytest.raises(ValueError, match="extra byte"):
        mcode_builder.MCodeProgram().short(1, b"\x00\x00", 0x83, 0, b"\x01")


def test_builder_can_extend_a_decoded_stream():
    program = mcode_builder.MCodeProgram().extend(
        [{"kind": "raw", "byte": 0x42}, {"kind": "raw", "byte": 0x00}]
    )
    assert program.encode() == b"\x42\x00"
