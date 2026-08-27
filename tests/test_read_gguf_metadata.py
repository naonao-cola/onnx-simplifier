"""Tests for ``onnxsim.read_gguf_metadata`` -- reading a GGUF checkpoint's
architecture hyperparameters (``general.architecture``, ``<arch>.block_count``,
``<arch>.attention.head_count``, ...) and per-tensor name/shape/dtype list,
without reading any tensor byte data.

``onnxsim.import_gguf_weights`` (see test_import_gguf_weights.py) parses this
very same GGUF header section but only ever looks at ``general.alignment``
before moving on to loading tensor *values* into an existing graph's
initializers -- it has no way to answer "what architecture is this
checkpoint, and what shape are its tensors?". ``read_gguf_metadata`` is that
other half: the input a from-scratch ONNX graph *builder* would need to
decide what graph structure to construct in the first place, before
``import_gguf_weights`` can fill in its values.

The C++-level decode logic (every scalar value type, sign extension, float
bit-reinterpretation, array skipping) is exhaustively covered by
onnxsim/read_gguf_metadata_test.cpp; this suite instead focuses on the
Python-facing contract: the returned dict's shape, a realistic
architecture-metadata checkpoint, and error behavior.
"""

import struct

import pytest

import onnxsim

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_BOOL = 7
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGUF_METADATA_VALUE_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12


def _string_bytes(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_STRING)
        + _string_bytes(value)
    )


def _kv_uint32(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_UINT32)
        + struct.pack("<I", value)
    )


def _kv_float32(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_FLOAT32)
        + struct.pack("<f", value)
    )


def _kv_bool(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_BOOL)
        + struct.pack("<B", 1 if value else 0)
    )


def _kv_string_array(key, values):
    body = struct.pack("<I", GGUF_METADATA_VALUE_TYPE_STRING) + struct.pack(
        "<Q", len(values)
    )
    for v in values:
        body += _string_bytes(v)
    return _string_bytes(key) + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_ARRAY) + body


def _tensor_info(name, ggml_type, ne):
    """``ne`` in GGML's own innermost-dimension-first order (the reverse of
    the ONNX shape ``read_gguf_metadata`` reports for it)."""
    info = _string_bytes(name)
    info += struct.pack("<I", len(ne))
    for d in ne:
        info += struct.pack("<Q", d)
    info += struct.pack("<I", ggml_type)
    info += struct.pack("<Q", 0)  # offset -- never read by read_gguf_metadata
    return info


def _write_gguf(path, kv_chunks, tensor_chunks):
    header = struct.pack(
        "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensor_chunks), len(kv_chunks)
    )
    with open(path, "wb") as f:
        f.write(header)
        for c in kv_chunks:
            f.write(c)
        for c in tensor_chunks:
            f.write(c)
        # Deliberately no tensor-data section at all: read_gguf_metadata must
        # never need one.


def test_realistic_llama_architecture_metadata(tmp_path):
    path = str(tmp_path / "model.gguf")
    kv = [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("llama.block_count", 32),
        _kv_uint32("llama.embedding_length", 4096),
        _kv_uint32("llama.attention.head_count", 32),
        _kv_uint32("llama.attention.head_count_kv", 8),
        _kv_float32("llama.rope.freq_base", 500000.0),
        _kv_float32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        _kv_bool("llama.attention.use_bias", False),
        # A large-ish string array (the shape tokenizer.ggml.tokens really
        # takes) must not surface in "kv" -- see the module docstring.
        _kv_string_array("tokenizer.ggml.tokens", ["<unk>", "<s>", "</s>", "hello"]),
    ]
    tensors = [
        _tensor_info("token_embd.weight", GGML_TYPE_F32, [4096, 32000]),
        _tensor_info("blk.0.attn_q.weight", GGML_TYPE_Q4_K, [4096, 4096]),
    ]
    _write_gguf(path, kv, tensors)

    meta = onnxsim.read_gguf_metadata(path)

    assert meta["kv"]["general.architecture"] == "llama"
    assert meta["kv"]["llama.block_count"] == 32
    assert meta["kv"]["llama.embedding_length"] == 4096
    assert meta["kv"]["llama.attention.head_count"] == 32
    assert meta["kv"]["llama.attention.head_count_kv"] == 8
    assert meta["kv"]["llama.rope.freq_base"] == pytest.approx(500000.0)
    assert meta["kv"]["llama.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-5)
    assert meta["kv"]["llama.attention.use_bias"] is False
    assert "tokenizer.ggml.tokens" not in meta["kv"]

    assert meta["tensors"] == [
        {
            "name": "token_embd.weight",
            "shape": [32000, 4096],
            "ggml_type": GGML_TYPE_F32,
        },
        {
            "name": "blk.0.attn_q.weight",
            "shape": [4096, 4096],
            "ggml_type": GGML_TYPE_Q4_K,
        },
    ]


def test_empty_file_has_no_kv_or_tensors(tmp_path):
    path = str(tmp_path / "empty.gguf")
    _write_gguf(path, [], [])
    meta = onnxsim.read_gguf_metadata(path)
    assert meta == {"kv": {}, "tensors": []}


def test_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        onnxsim.read_gguf_metadata(str(tmp_path / "does_not_exist.gguf"))


def test_bad_magic_raises(tmp_path):
    path = str(tmp_path / "bad_magic.gguf")
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0xDEADBEEF, GGUF_VERSION, 0, 0))
    with pytest.raises(RuntimeError):
        onnxsim.read_gguf_metadata(path)
