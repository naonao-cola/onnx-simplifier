"""Tests for the dependency-free half of onnx_to_tflite_micro.py.

emit_c_header() has no onnx2tf/TensorFlow dependency, so it's tested
directly against arbitrary bytes -- convert_to_tflite() (the onnx2tf
subprocess call) is exercised only by manual/local testing; see that
function's own docstring.
"""

import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from onnx_to_tflite_micro import emit_c_header, main


def _extract_bytes(header: str, var_name: str) -> bytes:
    m = re.search(rf"{var_name}\[\] = \{{(.*?)\}};", header, re.DOTALL)
    assert m, "array body not found"
    hex_bytes = re.findall(r"0x([0-9a-f]{2})", m.group(1))
    return bytes(int(h, 16) for h in hex_bytes)


def test_round_trips_arbitrary_bytes():
    data = bytes(range(256)) * 3  # exercises every byte value, several rows
    header = emit_c_header(data, "g_model")
    assert _extract_bytes(header, "g_model") == data


def test_empty_input():
    header = emit_c_header(b"", "g_model")
    assert _extract_bytes(header, "g_model") == b""
    assert "g_model_len = 0" in header


def test_declares_length_and_alignment():
    header = emit_c_header(b"\x01\x02\x03", "my_model")
    assert "alignas(16)" in header
    assert "const unsigned char my_model[]" in header
    assert "const unsigned int my_model_len = 3;" in header


def test_rejects_non_identifier_var_name():
    with pytest.raises(ValueError):
        emit_c_header(b"\x00", "not a valid name")


def _fake_convert_to_tflite(fake_bytes, tmp_path):
    fake_tflite = tmp_path / "fake.tflite"
    fake_tflite.write_bytes(fake_bytes)
    return fake_tflite


def test_main_writes_plain_tflite_for_dot_tflite_output(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not a real onnx file -- convert_to_tflite is mocked")
    out_path = tmp_path / "model.tflite"
    fake_bytes = b"\x01\x02\x03tflite-flatbuffer-bytes"
    with patch("onnx_to_tflite_micro.convert_to_tflite",
               return_value=_fake_convert_to_tflite(fake_bytes, tmp_path)):
        assert main([str(onnx_path), str(out_path)]) == 0
    assert out_path.read_bytes() == fake_bytes


def test_main_writes_c_header_for_dot_h_output(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not a real onnx file -- convert_to_tflite is mocked")
    out_path = tmp_path / "model_data.h"
    fake_bytes = b"\x01\x02\x03\x04"
    with patch("onnx_to_tflite_micro.convert_to_tflite",
               return_value=_fake_convert_to_tflite(fake_bytes, tmp_path)):
        assert main([str(onnx_path), str(out_path), "--var-name", "my_model"]) == 0
    header = out_path.read_text()
    assert "const unsigned char my_model[]" in header
    assert _extract_bytes(header, "my_model") == fake_bytes
