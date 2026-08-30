"""Tests for ``onnxsim.import_gguf_weights`` -- hydrating an existing ONNX
graph's initializers, by name, from a plain (non-onnxsim) GGUF checkpoint.
Unlike ``import_gguf``, this needs no embedded onnxsim model, and is the
intended way to bring a third-party GGUF's weight *values* into a graph you
already have.

Covers the GGML "K-quant" block formats (Q8_0, Q4_K, Q5_K, Q6_K) real
quantized checkpoints (e.g. Unsloth's GGUF exports) actually use for the
bulk of their weights: this module writes real, byte-accurate GGUF v3 files
containing hand-encoded K-quant blocks with known values, computing each
expected dequantized float independently (a from-scratch transcription of
GGML's published block layout/dequant formula, not a reuse of the C++
decoder under test -- see ggml_kquant.h) and checking onnxsim's decoded
result against it.
"""

import struct

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# ggml_type codes this suite constructs (see onnxsim/gguf_dtype.h).
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q4_0 = 2  # legacy family onnxsim does NOT decode -- must be skipped


def _align_up(n, align=32):
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def _write_gguf(path, tensors):
    """Write a minimal, real GGUF v3 file. ``tensors`` is a list of
    ``(name, ggml_type, ne, raw_bytes)`` -- ``ne`` in GGML's own
    innermost-dimension-first order (the reverse of the ONNX shape it
    corresponds to)."""
    infos = b""
    data_chunks = []
    offset = 0
    for name, ggml_type, ne, raw in tensors:
        name_b = name.encode("utf-8")
        infos += struct.pack("<Q", len(name_b)) + name_b
        infos += struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", ggml_type)
        infos += struct.pack("<Q", offset)
        data_chunks.append((offset, raw))
        offset = _align_up(offset + len(raw))

    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), 0)
    header_end = len(header) + len(infos)
    data_section_start = _align_up(header_end)

    with open(path, "wb") as f:
        f.write(header)
        f.write(infos)
        f.write(b"\x00" * (data_section_start - header_end))
        pos = data_section_start
        for rel_offset, raw in data_chunks:
            abs_offset = data_section_start + rel_offset
            f.write(b"\x00" * (abs_offset - pos))
            f.write(raw)
            pos = abs_offset + len(raw)


def _f16_bits(f):
    return np.float16(f).view(np.uint16).item()


def _f16_to_f32(bits):
    return float(np.uint16(bits).view(np.float16))


def _make_q8_0_block(rng):
    d = round(float(rng.uniform(0.01, 5.0)), 4)
    qs = [int(rng.integers(-127, 128)) for _ in range(32)]
    d_bits = _f16_bits(d)
    raw = struct.pack("<H", d_bits) + bytes(q & 0xFF for q in qs)
    expected = [q * _f16_to_f32(d_bits) for q in qs]
    return raw, expected


def _get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)
    return d, m


def _make_q4_k_block(rng):
    d = round(float(rng.uniform(0.01, 2.0)), 4)
    dmin = round(float(rng.uniform(0.01, 1.0)), 4)
    d_bits, dmin_bits = _f16_bits(d), _f16_bits(dmin)
    scales = [int(rng.integers(0, 256)) for _ in range(12)]
    qs = [int(rng.integers(0, 256)) for _ in range(128)]
    raw = struct.pack("<HH", d_bits, dmin_bits) + bytes(scales) + bytes(qs)

    d_f, dmin_f = _f16_to_f32(d_bits), _f16_to_f32(dmin_bits)
    expected = [0.0] * 256
    is_, q_off, y = 0, 0, 0
    for _j in range(0, 256, 64):
        sc, m = _get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d_f * sc, dmin_f * m
        sc, m = _get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d_f * sc, dmin_f * m
        for idx in range(32):
            expected[y] = d1 * (qs[q_off + idx] & 0xF) - m1
            y += 1
        for idx in range(32):
            expected[y] = d2 * (qs[q_off + idx] >> 4) - m2
            y += 1
        q_off += 32
        is_ += 2
    return raw, expected


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _identity_model(name, shape):
    # A minimal single-initializer graph: Identity(W) -> Y, with W the
    # initializer import_gguf_weights should hydrate. Seeded with zeros so a
    # test failing to actually hydrate is caught (not accidentally correct).
    # `name` is always quoted since a real llama.cpp GGUF tensor name (e.g.
    # "blk.0.ffn_gate_exps.weight") contains dots the text format's plain
    # (unquoted) identifier syntax doesn't accept as a node input.
    dims = ",".join(str(d) for d in shape)
    weight = onnx.numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name)
    return _model(
        f"""
        g () => (float[{dims}] Y)
        {{
          Y = Identity("{name}")
        }}
        """,
        initializer=[weight],
    )


def test_import_q8_0_weights(tmp_path):
    rng = np.random.default_rng(0)
    raw, expected = _make_q8_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    # ONNX shape [32] -> ggml ne [32] (rank 1, order is irrelevant).
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q8_0, [32], raw)])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q4_k_weights(tmp_path):
    rng = np.random.default_rng(1)
    raw, expected = _make_q4_k_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q4_K, [256], raw)])

    model = _identity_model("W", [256])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_multiple_tensors_two_dims(tmp_path):
    # Exercises the ne[]-order reversal (GGML innermost-first vs ONNX
    # outermost-first) with a non-square shape, and multiple Q8_0 blocks
    # concatenated in one tensor (2 rows x 64 cols = 2 blocks of 32).
    rng = np.random.default_rng(2)
    raw0, expected0 = _make_q8_0_block(rng)
    raw1, expected1 = _make_q8_0_block(rng)
    raw = raw0 + raw1
    expected = expected0 + expected1
    gguf_path = str(tmp_path / "model.gguf")
    # ONNX shape [2, 32] -> ggml ne [32, 2] (innermost-first).
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q8_0, [32, 2], raw)])

    model = _identity_model("W", [2, 32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(
        got.reshape(-1), np.array(expected, dtype=np.float32), rtol=1e-5
    )


def test_import_skips_unmatched_and_unsupported(tmp_path):
    rng = np.random.default_rng(3)
    raw, _ = _make_q8_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    legacy_raw = b"\x00" * 18  # Q4_0 block: 2 (d) + 16 (packed nibbles).
    _write_gguf(
        gguf_path,
        [
            ("W", GGML_TYPE_Q8_0, [32], raw),
            ("not_in_graph", GGML_TYPE_Q8_0, [32], raw),
            ("legacy", GGML_TYPE_Q4_0, [32], legacy_raw),
        ],
    )

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    # "not_in_graph" is a perfectly loadable Q8_0 tensor -- it's simply not
    # in `skipped`, since that list is TensorPool::LoadGGUF's own format-
    # level skip list (unsupported ggml_type), not a name-matching report.
    # It also never becomes a `model` initializer (this call only hydrates
    # initializers the graph already has, never adds new ones).
    assert set(skipped) == {"legacy"}
    names = {i.name for i in result.graph.initializer}
    assert names == {"W"}
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT


def test_import_raw_dtype_passthrough(tmp_path):
    # A raw (already-unquantized) F32 tensor hydrates unchanged, same as
    # HydrateTensorProto -- no dequantization involved.
    rng = np.random.default_rng(4)
    values = rng.standard_normal(8).astype(np.float32)
    gguf_path = str(tmp_path / "model.gguf")
    # GGUF is always little-endian on disk, regardless of host byte order.
    _write_gguf(gguf_path, [("W", GGML_TYPE_F32, [8], values.astype("<f4").tobytes())])

    model = _identity_model("W", [8])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_array_equal(got, values)


def test_import_skips_shape_mismatched_tensor(tmp_path):
    # The C++ side decodes/copies purely from the GGUF entry's own shape,
    # with no awareness of the target initializer's declared dims (see
    # onnx_simplifier.py's import_gguf_weights, `_tensor_proto_nbytes`) --
    # so a same-named match whose decoded byte count doesn't fit the
    # initializer's declared shape (here: the file's "W" is 8 raw floats,
    # but the graph declares W as shape [32], i.e. 32 floats) must be
    # reported in `skipped` and leave the initializer untouched, rather
    # than silently writing a too-short raw_data into a [32]-shaped
    # initializer.
    rng = np.random.default_rng(7)
    values = rng.standard_normal(8).astype(np.float32)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_F32, [8], values.astype("<f4").tobytes())])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == ["W"]
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert list(w.dims) == [32]
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_array_equal(got, np.zeros(32, dtype=np.float32))


def test_import_moe_expert_tensor_3d_matches_llama_cpp_layout(tmp_path):
    # llama.cpp's own GGUF convention for a MoE model's per-expert gate
    # projection is a single 3D tensor named "blk.N.ffn_gate_exps.weight"
    # (fc1 in com.microsoft.MoE's own naming -- see contrib_schemas.cpp),
    # GGML shape (ne, innermost-first) [hidden, inter, num_experts].
    # Reversing that (onnxsim's existing, rank-agnostic rule) gives ONNX
    # shape [num_experts, inter, hidden] -- exactly com.microsoft.MoE's
    # fc1_experts_weights layout, with no extra transpose needed. This is a
    # real 3D, multi-block round trip (4 Q8_0 blocks spanning the
    # expert/intermediate axes), unlike the rank-1/2 cases the rest of this
    # file covers.
    num_experts, inter, hidden = 2, 2, 32
    rng = np.random.default_rng(8)
    blocks = [_make_q8_0_block(rng) for _ in range(num_experts * inter)]
    raw = b"".join(b for b, _ in blocks)

    gguf_path = str(tmp_path / "model.gguf")
    name = "blk.0.ffn_gate_exps.weight"
    _write_gguf(gguf_path, [(name, GGML_TYPE_Q8_0, [hidden, inter, num_experts], raw)])

    model = _identity_model(name, [num_experts, inter, hidden])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == name)
    assert list(w.dims) == [num_experts, inter, hidden]
    got = onnx.numpy_helper.to_array(w)

    # Spot-check real 3D indexing (not just the flattened order): expert e's
    # block i should land at got[e, i, :], in GGML's flattening order.
    for e in range(num_experts):
        for i in range(inter):
            _, block_expected = blocks[e * inter + i]
            np.testing.assert_allclose(
                got[e, i, :], np.array(block_expected, dtype=np.float32), rtol=1e-5
            )


def test_import_gguf_weights_hydrates_a_moe_node_with_llama_cpp_names(tmp_path):
    # End-to-end: a com.microsoft.MoE node whose fc1/fc2/fc3 initializers
    # are named exactly the way llama.cpp's own GGUF export names a real
    # Mixtral/Qwen3-MoE/gpt-oss checkpoint's expert weights (gate=fc1,
    # up=fc3, down=fc2 -- see contrib_schemas.cpp's BuildMoEFunctionBody
    # comment) hydrates correctly from a real GGUF file, and the resulting
    # model still passes onnxsim.simplify()'s shape inference -- tying
    # import_gguf_weights and the MoE schema registration together, the
    # concrete case docs/import-gguf-weights.md's compatibility note is
    # about.
    num_experts, inter, hidden = 2, 2, 32

    def make_tensor(seed_offset):
        r = np.random.default_rng(9 + seed_offset)
        blocks = [_make_q8_0_block(r) for _ in range(num_experts * inter)]
        raw = b"".join(b for b, _ in blocks)
        expected = np.array([v for _, vals in blocks for v in vals], dtype=np.float32)
        return raw, expected

    gate_raw, gate_expected = make_tensor(1)
    up_raw, up_expected = make_tensor(2)
    down_raw, down_expected = make_tensor(3)

    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(
        gguf_path,
        [
            (
                "blk.0.ffn_gate_exps.weight",
                GGML_TYPE_Q8_0,
                [hidden, inter, num_experts],
                gate_raw,
            ),
            (
                "blk.0.ffn_up_exps.weight",
                GGML_TYPE_Q8_0,
                [hidden, inter, num_experts],
                up_raw,
            ),
            (
                "blk.0.ffn_down_exps.weight",
                GGML_TYPE_Q8_0,
                [inter, hidden, num_experts],
                down_raw,
            ),
        ],
    )

    num_tokens = 4
    # com.microsoft.MoE needs its own opset_import entry, which this file's
    # shared `_model` helper (only "" domain) doesn't provide -- built
    # directly, the same way tests/test_moe_contrib_schema.py's own `_model`
    # helper does.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18, "com.microsoft": 1]
        >
        agraph (float[{num_tokens},{hidden}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[?,?] output)
        {{
          output = com.microsoft.MoE
              <k: int = 1, activation_type: string = "silu",
               normalize_routing_weights: int = 0>
              (input, router_probs, "blk.0.ffn_gate_exps.weight", ,
               "blk.0.ffn_down_exps.weight", , "blk.0.ffn_up_exps.weight")
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(
                np.zeros((num_experts, inter, hidden), dtype=np.float32),
                "blk.0.ffn_gate_exps.weight",
            ),
            onnx.numpy_helper.from_array(
                np.zeros((num_experts, hidden, inter), dtype=np.float32),
                "blk.0.ffn_down_exps.weight",
            ),
            onnx.numpy_helper.from_array(
                np.zeros((num_experts, inter, hidden), dtype=np.float32),
                "blk.0.ffn_up_exps.weight",
            ),
        ]
    )

    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)
    assert skipped == []

    by_name = {i.name: i for i in result.graph.initializer}
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(by_name["blk.0.ffn_gate_exps.weight"]).reshape(-1),
        gate_expected,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(by_name["blk.0.ffn_up_exps.weight"]).reshape(-1),
        up_expected,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(by_name["blk.0.ffn_down_exps.weight"]).reshape(-1),
        down_expected,
        rtol=1e-5,
    )

    simplified, ok = onnxsim.simplify(
        result, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    dims = [
        d.dim_value if d.HasField("dim_value") else d.dim_param for d in out_shape.dim
    ]
    assert dims == [num_tokens, hidden]


def test_import_leaves_input_model_unchanged_and_preserves_unmatched(tmp_path):
    # Regression test for the C.import_gguf_weights binding change: matched
    # tensors' bytes now come back as a separate TensorPool instead of being
    # written into a fully re-serialized model (the same double protobuf
    # round-trip load_model/import_safetensors/import_gguf were fixed for),
    # so the caller's own `model` object must come back untouched, and an
    # initializer with NO match in the GGUF file must keep its original
    # value in `result` -- both are easy to get wrong when the matched and
    # unmatched tensors take different code paths on the way back.
    rng = np.random.default_rng(5)
    raw, expected = _make_q8_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("matched", GGML_TYPE_Q8_0, [32], raw)])

    unmatched_value = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
    model = _model(
        """
        g () => (float[32] Y, float[4] Z)
        {
          Y = Identity(matched)
          Z = Identity(unmatched)
        }
        """,
        initializer=[
            onnx.numpy_helper.from_array(np.zeros(32, dtype=np.float32), "matched"),
            onnx.numpy_helper.from_array(unmatched_value, "unmatched"),
        ],
    )
    before = onnx.ModelProto()
    before.CopyFrom(model)

    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert model == before, "import_gguf_weights must not mutate its input"
    assert skipped == []

    matched_init = next(i for i in result.graph.initializer if i.name == "matched")
    assert matched_init.data_type == onnx.TensorProto.FLOAT
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(matched_init),
        np.array(expected, dtype=np.float32),
        rtol=1e-5,
    )

    unmatched_init = next(i for i in result.graph.initializer if i.name == "unmatched")
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(unmatched_init), unmatched_value
    )
