"""Tests for ``onnxsim.import_gguf_weights`` -- hydrating an existing ONNX
graph's initializers, by name, from a plain (non-onnxsim) GGUF checkpoint.
Unlike ``import_gguf``, this needs no embedded onnxsim model, and is the
intended way to bring a third-party GGUF's weight *values* into a graph you
already have.

Covers the GGML "K-quant" block formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0)
real quantized checkpoints (e.g. Unsloth's GGUF exports) actually use for the
bulk of their weights; the legacy family (Q4_0, Q4_1, Q5_0, Q5_1) llama.cpp's
own mixed-precision quantizers still pick for particular tensor roles even
in an otherwise K-quant checkpoint (confirmed empirically against several
real, official gpt-oss-20b GGUF releases -- several of the most popular
size-optimized quantizations, e.g. Q4_K_S/Q4_0/UD-Q4_K_XL, use these for
their embedding/attention tensors); and MXFP4 (the OCP Microscaling FP4
format official gpt-oss GGUF releases use natively for their MoE expert
weights): this module writes real, byte-accurate GGUF v3 files containing
hand-encoded blocks with known values, computing each expected dequantized
float independently (a from-scratch transcription of GGML's published
block layout/dequant formula, not a reuse of the C++ decoder under test --
see ggml_kquant.h/ggml_legacy_quant.h/ggml_mxfp4.h) and checking onnxsim's
decoded result against it.
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
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_MXFP4 = 39
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_K = 15  # NOT decoded by onnxsim -- must be skipped


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


def _make_q2_k_block(rng):
    # One 256-element Q2_K super-block: 16 bytes of packed 4-bit (scale, min)
    # pairs (one pair per 16-element sub-block), 64 bytes of packed 2-bit
    # quant codes, then two fp16 super-block scales (d, dmin) -- transcribed
    # directly from ggml-quants.c's dequantize_row_q2_K.
    d = round(float(rng.uniform(0.01, 2.0)), 4)
    dmin = round(float(rng.uniform(0.01, 1.0)), 4)
    d_bits, dmin_bits = _f16_bits(d), _f16_bits(dmin)
    scales = [int(rng.integers(0, 256)) for _ in range(16)]
    qs = [int(rng.integers(0, 256)) for _ in range(64)]
    raw = bytes(scales) + bytes(qs) + struct.pack("<HH", d_bits, dmin_bits)

    d_f, dmin_f = _f16_to_f32(d_bits), _f16_to_f32(dmin_bits)
    expected = [0.0] * 256
    is_, y = 0, 0
    q_off = 0
    for _n in range(0, 256, 128):
        shift = 0
        for _j in range(4):
            sc = scales[is_]
            is_ += 1
            dl, ml = d_f * (sc & 0xF), dmin_f * (sc >> 4)
            for idx in range(16):
                expected[y] = dl * ((qs[q_off + idx] >> shift) & 3) - ml
                y += 1
            sc = scales[is_]
            is_ += 1
            dl, ml = d_f * (sc & 0xF), dmin_f * (sc >> 4)
            for idx in range(16):
                expected[y] = dl * ((qs[q_off + 16 + idx] >> shift) & 3) - ml
                y += 1
            shift += 2
        q_off += 32
    return raw, expected


def _unpack_q3k_scales(scales12):
    # Unpacks Q3_K's 12-byte packed-6-bit scale table into 16 signed values
    # (still offset by +32, matching GGML's own `scales[is++] - 32` step) --
    # transcribed directly from ggml-quants.c's dequantize_row_q3_K, reading
    # each 4-byte word explicitly little-endian rather than via GGML's own
    # (little-endian-host-only) `memcpy` onto a native `uint32_t[3]` -- see
    # onnxsim/ggml_kquant.h's file comment for why that distinction matters.
    def le32(b):
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    aux = [le32(scales12[0:4]), le32(scales12[4:8]), le32(scales12[8:12]), 0]
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    tmp = aux[2]
    new0 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    new1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    new2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    new3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux = [new0 & 0xFFFFFFFF, new1 & 0xFFFFFFFF, new2 & 0xFFFFFFFF, new3 & 0xFFFFFFFF]
    out = []
    for k in range(16):
        byte = (aux[k // 4] >> (8 * (k % 4))) & 0xFF
        out.append(byte - 256 if byte >= 128 else byte)
    return out


def _make_q3_k_block(rng):
    # One 256-element Q3_K super-block: a 32-byte high-bit mask, 64 bytes of
    # packed 2-bit low quant codes, a 12-byte packed-6-bit scale table, then
    # one fp16 super-block scale -- transcribed directly from
    # ggml-quants.c's dequantize_row_q3_K.
    d = round(float(rng.uniform(0.01, 2.0)), 4)
    d_bits = _f16_bits(d)
    hmask = [int(rng.integers(0, 256)) for _ in range(32)]
    qs = [int(rng.integers(0, 256)) for _ in range(64)]
    scales12 = bytes(int(rng.integers(0, 256)) for _ in range(12))
    raw = bytes(hmask) + bytes(qs) + scales12 + struct.pack("<H", d_bits)

    d_f = _f16_to_f32(d_bits)
    scales = _unpack_q3k_scales(scales12)
    expected = [0.0] * 256
    is_, y, m = 0, 0, 1
    q_off = 0
    for _n in range(0, 256, 128):
        shift = 0
        for _j in range(4):
            dl = d_f * (scales[is_] - 32)
            is_ += 1
            for idx in range(16):
                low = (qs[q_off + idx] >> shift) & 3
                bit = hmask[idx] & m
                expected[y] = dl * (low - (0 if bit else 4))
                y += 1
            dl = d_f * (scales[is_] - 32)
            is_ += 1
            for idx in range(16):
                low = (qs[q_off + 16 + idx] >> shift) & 3
                bit = hmask[idx + 16] & m
                expected[y] = dl * (low - (0 if bit else 4))
                y += 1
            shift += 2
            m = (m << 1) & 0xFF
        q_off += 32
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


# GGML's kvalues_fp4/kvalues_mxfp4 table (e2m1-style magnitudes, doubled):
# code 0-7 are the non-negative magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} times
# 2, codes 8-15 their negated counterparts -- transcribed directly from the
# OCP Microscaling FP4 spec table GGML itself uses (ggml-common.h's
# kvalues_fp4), same as onnxsim/ggml_mxfp4.h's kMxfp4Values.
_MXFP4_VALUES = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]


def _e8m0_to_f32_half(e):
    # Independent (pure floating-point, not bit-pattern-construction like
    # ggml_mxfp4.h's GgmlE8m0ToFloat32Half) computation of the same value:
    # an E8M0 byte `e` nominally encodes 2**(e-127), and GGML's own
    # ggml_e8m0_to_fp32_half halves that (since _MXFP4_VALUES above is
    # already doubled) -- 2**(e-128).
    return 2.0 ** (e - 128)


def _make_mxfp4_block(rng):
    # One 32-element MXFP4 block: 1 byte E8M0 exponent + 16 bytes of packed
    # 4-bit codes, element i and i+16 sharing byte i's low/high nibble (NOT
    # the consecutive-pair packing K-quant uses).
    e = int(rng.integers(0, 255))  # 255 is GGML's only NaN encoding
    codes = [int(rng.integers(0, 16)) for _ in range(32)]
    qs = bytes((codes[i] & 0xF) | ((codes[i + 16] & 0xF) << 4) for i in range(16))
    raw = struct.pack("<B", e) + qs
    d = _e8m0_to_f32_half(e)
    expected = [_MXFP4_VALUES[c] * d for c in codes]
    return raw, expected


def _make_q4_0_block(rng):
    # One 32-element Q4_0 block: a single fp16 scale plus 16 bytes of packed
    # 4-bit codes (unsigned 0..15, representing signed -8..7 via a fixed -8
    # bias, no separate min value).
    d = round(float(rng.uniform(0.01, 5.0)), 4)
    d_bits = _f16_bits(d)
    codes = [int(rng.integers(0, 16)) for _ in range(32)]
    qs = bytes((codes[i] & 0xF) | ((codes[i + 16] & 0xF) << 4) for i in range(16))
    raw = struct.pack("<H", d_bits) + qs
    d_f = _f16_to_f32(d_bits)
    expected = [(c - 8) * d_f for c in codes]
    return raw, expected


def _make_q4_1_block(rng):
    # Like Q4_0, but with an explicit fp16 min added after scaling instead
    # of a fixed bias (codes are used unsigned, 0..15).
    d = round(float(rng.uniform(0.01, 2.0)), 4)
    m = round(float(rng.uniform(-1.0, 1.0)), 4)
    d_bits, m_bits = _f16_bits(d), _f16_bits(m)
    codes = [int(rng.integers(0, 16)) for _ in range(32)]
    qs = bytes((codes[i] & 0xF) | ((codes[i + 16] & 0xF) << 4) for i in range(16))
    raw = struct.pack("<HH", d_bits, m_bits) + qs
    d_f, m_f = _f16_to_f32(d_bits), _f16_to_f32(m_bits)
    expected = [c * d_f + m_f for c in codes]
    return raw, expected


def _make_q5_0_block(rng):
    # Like Q4_0, but each element's 5th (high) bit lives in a separate
    # 4-byte `qh` bitfield (one bit per element) rather than packed
    # alongside the other 4 bits -- the resulting 5-bit unsigned code
    # (0..31) is biased by -16 before scaling.
    d = round(float(rng.uniform(0.01, 5.0)), 4)
    d_bits = _f16_bits(d)
    codes = [int(rng.integers(0, 32)) for _ in range(32)]
    low_nibbles = [c & 0xF for c in codes]
    high_bits = [(c >> 4) & 0x1 for c in codes]
    qs = bytes(
        (low_nibbles[i] & 0xF) | ((low_nibbles[i + 16] & 0xF) << 4) for i in range(16)
    )
    # GGML's qh packing: element j's high bit lives at bit j (j < 16) or bit
    # (j - 16) + 16 (j >= 16) of the 32-bit qh word -- i.e. bit j directly
    # for the low half, bit j+16 for the high half (see
    # dequantize_row_q5_0's xh_0/xh_1 extraction, which this mirrors in the
    # opposite direction).
    qh = 0
    for j in range(16):
        if high_bits[j]:
            qh |= 1 << j
        if high_bits[j + 16]:
            qh |= 1 << (j + 16)
    raw = struct.pack("<H", d_bits) + struct.pack("<I", qh) + qs
    d_f = _f16_to_f32(d_bits)
    expected = [(c - 16) * d_f for c in codes]
    return raw, expected


def _make_q5_1_block(rng):
    # Like Q5_0, but with an explicit fp16 min added after scaling instead
    # of a fixed bias (the 5-bit code is used unsigned, 0..31), matching
    # Q4_1's own relationship to Q4_0.
    d = round(float(rng.uniform(0.01, 2.0)), 4)
    m = round(float(rng.uniform(-1.0, 1.0)), 4)
    d_bits, m_bits = _f16_bits(d), _f16_bits(m)
    codes = [int(rng.integers(0, 32)) for _ in range(32)]
    low_nibbles = [c & 0xF for c in codes]
    high_bits = [(c >> 4) & 0x1 for c in codes]
    qs = bytes(
        (low_nibbles[i] & 0xF) | ((low_nibbles[i + 16] & 0xF) << 4) for i in range(16)
    )
    qh = 0
    for j in range(16):
        if high_bits[j]:
            qh |= 1 << j
        if high_bits[j + 16]:
            qh |= 1 << (j + 16)
    raw = struct.pack("<HH", d_bits, m_bits) + struct.pack("<I", qh) + qs
    d_f, m_f = _f16_to_f32(d_bits), _f16_to_f32(m_bits)
    expected = [c * d_f + m_f for c in codes]
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


def test_import_q2_k_weights(tmp_path):
    rng = np.random.default_rng(11)
    raw, expected = _make_q2_k_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q2_K, [256], raw)])

    model = _identity_model("W", [256])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q3_k_weights(tmp_path):
    rng = np.random.default_rng(12)
    raw, expected = _make_q3_k_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q3_K, [256], raw)])

    model = _identity_model("W", [256])
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


def test_import_mxfp4_weights(tmp_path):
    rng = np.random.default_rng(10)
    raw, expected = _make_mxfp4_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_MXFP4, [32], raw)])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q4_0_weights(tmp_path):
    rng = np.random.default_rng(11)
    raw, expected = _make_q4_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q4_0, [32], raw)])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q4_1_weights(tmp_path):
    rng = np.random.default_rng(12)
    raw, expected = _make_q4_1_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q4_1, [32], raw)])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q5_0_weights(tmp_path):
    rng = np.random.default_rng(13)
    raw, expected = _make_q5_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q5_0, [32], raw)])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q5_1_weights(tmp_path):
    rng = np.random.default_rng(14)
    raw, expected = _make_q5_1_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q5_1, [32], raw)])

    model = _identity_model("W", [32])
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
    # Q8_K is not decoded by onnxsim at all, so its exact byte count doesn't
    # matter here (this tensor's bytes are never read) -- picked because
    # it's the last tensor in this file, so nothing after it needs locating.
    unsupported_raw = b"\x00" * 110
    _write_gguf(
        gguf_path,
        [
            ("W", GGML_TYPE_Q8_0, [32], raw),
            ("not_in_graph", GGML_TYPE_Q8_0, [32], raw),
            ("unsupported", GGML_TYPE_Q8_K, [256], unsupported_raw),
        ],
    )

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    # "not_in_graph" is a perfectly loadable Q8_0 tensor -- it's simply not
    # in `skipped`, since that list is TensorPool::LoadGGUF's own format-
    # level skip list (unsupported ggml_type), not a name-matching report.
    # It also never becomes a `model` initializer (this call only hydrates
    # initializers the graph already has, never adds new ones).
    assert set(skipped) == {"unsupported"}
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
