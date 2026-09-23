"""Regression coverage for ``scripts/axera/llm_build_dtype_analysis.py``.

The script covers how ``pulsar2 llm_build`` lays out each ``--weight_type``
(s4, s8, fp16, bf16, fp8_e4m3, fp8_e5m2). No Docker, device or network is
needed. The fixtures are byte slices of our own builds' ``npu_params`` and
gzip'd decode-subgraph MCode. The weights are regenerated from the seeds the
builds used. See ``docs/axera-llm-build-dtype-analysis.md``."""

import gzip
import json
import os
import sys

import numpy as np
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import llm_build_dtype_analysis as lda  # noqa: E402
import short_unit_codec as codec  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures", "llm_build_dtype_analysis")
_BLOCKS = dict(np.load(os.path.join(_FIX, "blocks.npz")))
_SUMMARY = json.load(open(os.path.join(_FIX, "summary.json")))
_W = lda.tiny_llama_weights(512)
_Q = _W["model.layers.0.self_attn.q_proj.weight"]


def _fix(name):
    return _BLOCKS[name].tobytes()


def _mcode(tag):
    with open(os.path.join(_FIX, f"decode_{tag}.mcode.gz"), "rb") as f:
        return gzip.decompress(f.read())


@pytest.mark.parametrize("weight_type", ["s8", "s4"])
def test_int_block_is_reproduced_byte_for_byte(weight_type):
    assert lda.encode_block(_Q[:32], weight_type) == _fix(
        f"q_proj_block0_{weight_type}"
    )


@pytest.mark.parametrize("weight_type", lda.FLOAT_TYPES)
def test_float_block_is_fp32_words_of_the_rounded_weight(weight_type):
    got = _fix(f"q_proj_block0_{weight_type}")
    assert lda.encode_block(_Q[:32], weight_type)[: len(got)] == got
    # Column-major float32 words: column 0 holds rows 0..31, then column 1.
    words = np.frombuffer(got, "<f4").reshape(-1, 32)
    np.testing.assert_array_equal(
        words, lda.round_float(_Q[:32, : len(words)], weight_type).T
    )


def test_float_types_differ_only_in_rounding():
    bf16 = np.frombuffer(_fix("q_proj_block0_bf16"), "<u4")
    assert (bf16 & 0xFFFF == 0).all()  # bf16 values widened to float32
    fp16 = np.frombuffer(_fix("q_proj_block0_fp16"), "<f4")
    np.testing.assert_array_equal(fp16, fp16.astype(np.float16).astype(np.float32))
    # fp8 is a plain cast with no scale: with |w| ~ 0.02, e4m3 lands in its
    # subnormal/low-exponent range, so only a few distinct values survive.
    e4 = np.frombuffer(_fix("q_proj_block0_fp8_e4m3"), "<f4")
    assert len(np.unique(e4)) < 64
    assert (
        _SUMMARY["npu_params_bytes"]["bf16"] == _SUMMARY["npu_params_bytes"]["fp8_e4m3"]
    )


def test_int_tail_is_rowsum_zeros_and_float32_row_scales():
    q, scale = lda.quantize_int(_Q[:32], 8)
    got = lda.decode_block(_fix("q_proj_block0_s8"), "s8", 256)
    np.testing.assert_array_equal(got["q"], q)
    np.testing.assert_array_equal(got["scale"], scale)
    np.testing.assert_array_equal(got["rowsum"][:, 0], -q.sum(axis=1) / 2)
    tail = _fix("q_proj_block0_s8")[-512:]
    assert tail[256:384] == bytes(128)
    assert all(tail[8 * r + 4 : 8 * r + 8] == bytes(4) for r in range(32))


def test_s8_near_ties_are_the_only_code_mismatches():
    w = _W["model.layers.0.self_attn.o_proj.weight"][:32]
    q, scale = lda.quantize_int(w, 8)
    got = lda.decode_block(_fix("o_proj_block0_s8"), "s8", 256)
    np.testing.assert_array_equal(got["scale"], scale)
    bad = got["q"] != q
    assert bad.sum() == 1
    ratio = (w / scale[:, None])[bad]
    assert np.all(np.abs(ratio - np.floor(ratio) - 0.5) < 1e-5)
    assert np.all(got["q"][bad] == q[bad] + 1)  # rounded up past the tie


def test_s4_wide_input_splits_into_544_column_blocks():
    assert lda.column_blocks(2048) == [416, 544, 544, 544]
    assert lda.column_blocks(4096) == [288] + [544] * 7  # the README's s8 split
    assert lda.column_blocks(512) == [512]
    down = lda.tiny_llama_weights(2048)["model.layers.0.mlp.down_proj.weight"]
    data = _fix("down_proj_wide_block0_s4")
    assert lda.encode_block(down[:32], "s4") == data
    got = lda.decode_block(data, "s4", 2048)
    q, scale = lda.quantize_int(down[:32], 4)
    np.testing.assert_array_equal(got["q"], q)
    # One scale per full row, repeated in all four column-block tails: no
    # per-k-group scales.
    raw = np.frombuffer(scale.astype("<f4").tobytes(), np.uint8).tobytes()
    assert data.count(raw) == 4
    sums = [
        q[:, a:b].sum(axis=1)
        for a, b in ((0, 416), (416, 960), (960, 1504), (1504, 2048))
    ]
    np.testing.assert_array_equal(got["rowsum"], -np.stack(sums, axis=1) / 2)


def test_hypothesis_nibble_pairs_are_s4_and_s8_nibble_planes():
    # s4: byte = (code[2j+1] << 4) | code[2j], code = q + 8.
    q4, _ = lda.quantize_int(_Q[:32], 4)
    s4 = _fix("q_proj_block0_s4")
    assert s4[0] == ((q4[0, 1] + 8) << 4) | (q4[0, 0] + 8)
    # s8: the same pair shape, but one byte per nibble plane of q + 128, the
    # planes 18 bytes apart. The high plane is (q >> 4) + 8, the external
    # "+8 offset".
    q8, _ = lda.quantize_int(_Q[:32], 8)
    s8 = _fix("q_proj_block0_s8")
    c = q8[0] + 128
    assert s8[0] == ((c[1] & 15) << 4) | (c[0] & 15)
    assert s8[18] == ((c[1] >> 4) << 4) | (c[0] >> 4)
    assert (c[0] >> 4) == (q8[0, 0] >> 4) + 8


def test_hypothesis_metadata_half_of_72_byte_unit_is_row_r_plus_16():
    q8, _ = lda.quantize_int(_Q[:32], 8)
    unit = _fix("q_proj_block0_s8")[:72]
    c = q8[16, :36] + 128
    assert unit[36:54] == bytes(((c[1::2] & 15) << 4 | (c[0::2] & 15)).astype(np.uint8))
    assert unit[54:72] == bytes(((c[1::2] >> 4) << 4 | (c[0::2] >> 4)).astype(np.uint8))


@pytest.mark.parametrize("weight_type", lda.WEIGHT_TYPES)
def test_hypothesis_no_duplicate_weight_copy(weight_type):
    offsets = _SUMMARY["offsets"][weight_type]
    spans = []
    for name, offs in offsets.items():
        cin = _W[f"model.layers.0.{name}.weight"].shape[1]
        size = lda.block_bytes(weight_type, cin)
        # The one s8 block holding near-tie codes is at the implied offset.
        offs = [o if o is not None else offs[i + 1] - size for i, o in enumerate(offs)]
        spans += [(o, o + size) for o in offs]
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    weights = sum(b - a for a, b in spans)
    # What is left: the 1/sqrt(head_dim) table (+4 bytes on int types), a
    # 127 x 32 float32 RoPE table, a 256-byte constant, and on float types a
    # 128-byte gap.
    rest = _SUMMARY["npu_params_bytes"][weight_type] - weights
    assert rest == (17152 if weight_type in lda.FLOAT_TYPES else 17028)


@pytest.mark.parametrize("weight_type", lda.WEIGHT_TYPES)
def test_params_start_with_attention_scale(weight_type):
    head = np.frombuffer(_fix(f"head_{weight_type}")[:512], "<f4")
    np.testing.assert_array_equal(head, np.float32(1 / np.sqrt(32)))


def test_codec_parses_llm_build_mcode():
    for tag in ("bf16", "bf16_rebuild", "fp16", "s8", "s4"):
        segs = lda.mcode_summary(_mcode(tag))
        assert len(segs) == 15 and all(s["records"] for s in segs)
        raw = codec.decode_segments(_mcode(tag))[0]
        assert codec.decode(codec.encode(raw)) == raw


def _changed(a, b):
    return {
        (d["segment"], v, r)
        for d in lda.mcode_record_diff(_mcode(a), _mcode(b))
        for v, r in d["changed_streams"]
    }


def test_float_types_share_one_program_up_to_rebuild_noise():
    noise = _changed("bf16", "bf16_rebuild")
    assert noise  # the same build twice already differs here
    assert _changed("bf16", "fp16") <= {
        (0, 0xA2, 0x0000),
        (9, 0xA8, 0x0170),
        (12, 0xA8, 0x0230),
    }
    assert noise <= {(0, 0xA2, 0x0000), (9, 0xA8, 0x0170), (12, 0xA8, 0x0230)}
    counts = [s["records"] for s in lda.mcode_summary(_mcode("bf16"))]
    assert counts == [s["records"] for s in lda.mcode_summary(_mcode("fp16"))]


def test_int_types_run_different_programs():
    s8 = [s["records"] for s in lda.mcode_summary(_mcode("s8"))]
    s4 = [s["records"] for s in lda.mcode_summary(_mcode("s4"))]
    bf16 = [s["records"] for s in lda.mcode_summary(_mcode("bf16"))]
    assert s8 != s4 and s8 != bf16
    assert len(_changed("s8", "s4")) > 100
