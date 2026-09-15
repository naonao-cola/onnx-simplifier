"""The mcode checker, run on committed blobs from real Pulsar2 builds.

Everything else in this project's Axera work needs a Pulsar2 Docker image or
an AX650N card. This file does not: it reads seven mcodes compiled earlier and
checked into `scripts/axera/fixtures/`, so the codec and the structural rules
in `scripts/axera/mcode.py` are exercised on every push, on a stock runner.

What is checked is *form*, not arithmetic -- see that module's docstring for
why there is no evaluator. The rules are the ones confirmed on hardware: the
segment table is a loader manifest whose word counts must tile the stream, a7
marks every segment boundary but the first, the verb and tag sets are closed,
and the codec round-trips byte for byte.
"""

import gzip
import os
import random
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")

# name -> (stream length, segment count, op programs). The op-program count is
# the number of `a1 00 40 02` writes, the convolution engine's channel-extent
# register: one per convolution the compiler emitted, so it counts the layers
# a graph was split into rather than the layers it was written with.
_BLOBS = {
    "conv64_k5_d2": (2824, 5, 2),
    "conv128_k7_d12": (10544, 5, 44),
    "piper_vocoder": (82216, 5, 199),
    # A real *training*-step compile (wav2vec2's feature extractor, PR #1370)
    # -- Gather/MatMul-heavy backward pass and in-graph SGD update, not a
    # forward-only inference graph like the three above. Added specifically
    # to confirm the structural rules (derived from inference-only fixtures)
    # generalise rather than being over-fit -- see
    # docs/axera-mcode-training-graph-coverage.md for the full analysis
    # across four real training graphs (this one, resnet18, resnet50,
    # Whisper), all of which pass cleanly with the same tag-frequency
    # profile as the inference-only fixtures above.
    "w2v2fe_training_step": (202976, 5, 533),
    # Three single-purpose probes, compiled with Pulsar2 7.0-lite: a
    # group-32 depthwise convolution, a last-axis LayerNormalization, and a
    # MatMul/Softmax/MatMul attention fragment with folded K/V. None shares
    # an op with the fixtures above beyond MatMul, and the layernorm one
    # carries no `a1 00 40 02` op program at all -- yet all three use only
    # the closed verb/tag sets and pass every rule. See the README's "Three
    # new families, no new forms" section.
    "dwconv_g32": (3176, 5, 3),
    "layernorm_last_axis": (2528, 5, 0),
    "attn_qkv_softmax": (5552, 5, 5),
    # Training-graph and backward-slice streams, compiled with Pulsar2
    # 7.0-lite and run on a real AX650N: a full distillation training step
    # (forward + KD loss + backward + Adam, the first training-graph stream
    # in this corpus), an INT8 ResNet18 classifier, the KD soft-loss head,
    # an Adam update with the FP32-Sub override, a Reshape->Gather
    # backward slice, and the Reshape->Mul / Reshape->MatMul pair that
    # pinpoint the scheduler's standalone-reshape crash boundary (see the
    # README's "Training-graph streams" section). The pair sits just under
    # the coverage floor with characterized 1-2 byte singles -- pinned by
    # test_known_gap_streams_hold_status_quo below, not by the clean test.
    "toy_training_step": (31032, 5, 24),
    "resnet18_int8": (50984, 5, 186),
    "loss_head_kd": (5480, 5, 1),
    "adam_update_fp32": (2288, 5, 2),
    "reshape_gather_bwd": (2600, 5, 1),
    # A unary-negation probe: the first stream emitted against by
    # scripts/axera/tiny_emit.py (tinygrad-traced MUL-by-minus-one).
    "neg_1x8": (2432, 5, 1),
    # A two-input elementwise Mul: the first multi-input stream in the
    # corpus, which exposed the 297-byte header assumption in
    # tail_vector (its pointer sits at 328). See "Tails beyond the fixed
    # header" below.
    "mul_1x8": (2264, 5, 0),
}


def _blob(name):
    with gzip.open(os.path.join(_FIXTURES, name + ".mcode.gz"), "rb") as f:
        return f.read()


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_real_mcode_passes_every_structural_rule(name):
    """The three committed streams -- a small convolution, a widely dilated
    one, and the real Piper vocoder -- violate none of the rules confirmed on
    an AX650N."""
    blob = _blob(name)
    length, segments, ops = _BLOBS[name]
    assert len(blob) == length
    assert mcode.check(blob) == []
    assert len(mcode.segments(blob)[1]) == segments
    assert sum(len(v) for v in mcode.op_programs(blob).values()) == ops


# Per-stream structured-share floors where the default 0.90 below does not
# fit: mul_1x8's two-input header is proportionally more zero padding.
_SHARE_FLOOR = {
    "mul_1x8": 0.88,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_the_codec_round_trips_byte_for_byte(name):
    """Decoding and re-encoding reproduces the stream exactly. This is the
    property the whole codec rests on: an edit that keeps it is an edit that
    changed only what it meant to."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert mcode.encode(records) == blob[lo:hi]
    # Nearly all of it comes from a recognised form rather than a raw escape.
    # The floor is 0.90 rather than higher because tiny streams carry
    # proportionally more zero padding, which counts as raw. The two-input
    # Mul stream carries a longer header still, so it gets its own floor.
    assert mcode.structured_share(records) >= _SHARE_FLOOR.get(name, 0.90)


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_a_deleted_byte_is_caught(name):
    """Dropping one byte shifts everything after it. That is the failure a
    hand-written or hand-edited stream actually makes, and the checker has to
    see it rather than quietly decode nonsense."""
    blob = _blob(name)
    lo, _ = mcode.stream_bounds(blob)
    cut = lo + 40
    assert mcode.check(blob[:cut] + blob[cut + 1 :] + b"\x00") != []


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_bulk_corruption_is_caught(name):
    """A block of bytes belonging to no form is caught twice over: by the
    coverage floor and by the run-length rule. No real stream -- CNN or
    `llm_build`, up to 1.8 MB -- has an unexplained non-zero run over nine
    bytes long."""
    blob = bytearray(_blob(name))
    lo, _ = mcode.stream_bounds(bytes(blob))
    blob[lo + 64 : lo + 192] = bytes(range(128))
    assert mcode.check(bytes(blob)) != []


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_a_stream_without_its_manifest_is_caught(name):
    """The segment table is a loader manifest, and it lives at the end. A
    stream cut off before it -- or anywhere that loses it -- cannot be walked
    at all, which is what the runtime itself refuses."""
    blob = _blob(name)
    _, hi = mcode.stream_bounds(blob)
    assert mcode.check(blob[:hi]) != []
    assert mcode.check(blob[: len(blob) // 2]) != []


def test_the_checker_reports_rather_than_raises():
    """Fed bytes that are not an mcode at all, `check` returns violations. A
    validator that raises is one a caller has to wrap, and a corrupt blob is
    exactly when it gets called."""
    rng = random.Random(0)
    for size in (0, 1, 64, 4096):
        blob = bytes(rng.randrange(256) for _ in range(size))
        assert mcode.check(blob) != []
    # A real stream with its tail vector scribbled over is the near miss.
    blob = bytearray(_blob("conv64_k5_d2"))
    vec = mcode.tail_vector(bytes(blob))
    blob[vec : vec + 24] = b"\xff" * 24
    assert mcode.check(bytes(blob)) != []


def test_every_segment_but_the_first_is_marked_by_a7():
    """`a7` is the synchronisation verb. It sits within four bytes of every
    segment boundary except the stream's first -- the verb can begin just
    before the word boundary its segment starts on."""
    for name in sorted(_BLOBS):
        blob = _blob(name)
        lo, hi = mcode.stream_bounds(blob)
        records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
        at = {r["at"] for r in records if r["kind"] == "V" and r["verb"] == 0xA7}
        _, segments = mcode.segments(blob)
        for pos, _, _ in segments[1:]:
            assert any(pos + d in at for d in range(-4, 5)), (name, pos)


# Quintets per committed stream -- `[04][a][b][a][b]` units, counted with the
# form admitted (see `test_quintet_programs_its_pair_twice`).
_Q_COUNTS = {
    "conv64_k5_d2": 0,
    "conv128_k7_d12": 8,
    "piper_vocoder": 86,
    "w2v2fe_training_step": 76,
    "dwconv_g32": 0,
    "layernorm_last_axis": 0,
    "attn_qkv_softmax": 0,
    "toy_training_step": 0,
    "resnet18_int8": 0,
    "loss_head_kd": 0,
    "adam_update_fp32": 0,
    "reshape_gather_bwd": 0,
    "neg_1x8": 0,
    "mul_1x8": 0,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_quintet_programs_its_pair_twice(name):
    """A five-byte `[04][a][b][a][b]` unit: the last two payload bytes repeat
    the middle two. Found by clustering the unexplained bytes of 68 real
    streams from an AX650N -- the `05`-led six-byte gaps resolve into clean
    parses once these are admitted -- and validated the way every other form
    was: it fires thousands of times on real streams, a handful on shuffled
    ones, and only where no other form matches, so no stream regresses. See
    the README's "A five-byte form that programs its pair twice" section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    quintets = [t for t in toks if t[1] == "Q"]
    assert len(quintets) == _Q_COUNTS[name], (name, len(quintets))
    for o, _, a, b, _ in quintets:
        assert (blob[o], blob[o + 1], blob[o + 2], blob[o + 3], blob[o + 4]) == (
            0x04,
            a,
            b,
            a,
            b,
        )
        # Neither slot is a verb byte: a quintet can never split a verb the
        # way a short unit ending in `a1 00` splits one beginning there.
        assert a not in mcode.VERBS6 and b not in mcode.VERBS6, (name, o)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [(r["a"], r["b"]) for r in records if r["kind"] == "Q"] == [
        (a, b) for _, _, a, b, _ in quintets
    ]

    # Admitting the form buys coverage, and only where it fires.
    without = dict(mcode.FULL_RULE)
    without["quintet"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if _Q_COUNTS[name]:
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The shuffle control: far fewer quintets by chance than by structure.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = sum(
        1
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] == "Q"
    )
    assert null <= max(2, len(quintets) // 4), (name, len(quintets), null)


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_lookahead_never_regresses_coverage(name):
    """Adjudicating overlaps can only move bytes from raw escapes into
    recognised forms or leave them where they were: per-stream coverage with
    the lookahead on is never below the greedy walk's."""
    blob = _blob(name)
    plain = dict(mcode.FULL_RULE)
    plain["lookahead"] = 0
    covered_plain, _ = mcode.nonzero_coverage(blob, **plain)
    covered_with, _ = mcode.nonzero_coverage(blob)
    assert covered_with >= covered_plain, (name, covered_plain, covered_with)


def test_lookahead_adjudicates_a1_tag_verb_overlaps():
    """Where a short unit ending in `a1 00` overlaps a verb beginning there,
    the greedy walk always takes the short unit and usually strands the
    verb's field and bank bytes. With the lookahead, each overlap is walked
    both ways for 64 bytes and the cleaner parse wins -- a swallowed verb
    head is restored, while a genuine tag (whose continuation already parses)
    keeps its short unit. Both outcomes occur in the committed vocoder
    stream, pinned here by offset."""
    blob = _blob("piper_vocoder")
    lo, hi = mcode.stream_bounds(blob)
    plain = dict(mcode.FULL_RULE)
    plain["lookahead"] = 0

    plain_toks = {t[0]: t[1] for t in mcode.tokenize(blob, start=lo, end=hi, **plain)}
    ruled_toks = {
        t[0]: t[1] for t in mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    }
    # At both sites the greedy walk takes the short unit...
    assert plain_toks[1019] == "S", plain_toks[1019]
    assert plain_toks[967] == "S", plain_toks[967]
    # ...the lookahead restores the swallowed `a1 00 30 04` verb at one...
    assert ruled_toks[1019] == "?", ruled_toks[1019]
    assert blob[1019:1028] == bytes.fromhex("00 03 a1 00 30 04 18 01 80")
    following = [
        t
        for t in mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
        if t[0] == 1021
    ]
    assert following[0][1:] == ("V", 0xA1, 0x30, 0x04), following
    # ...and keeps the genuine short unit at the other.
    assert ruled_toks[967] == "S", ruled_toks[967]
    assert blob[967:979] == bytes.fromhex("03 3f 00 00 11 a1 00 b0 03 3f f0 3e")


# (P, D) token counts per committed stream -- `0b 91` prefixes under their
# anchored verb, and raw-on-both-bytes `05 90` doublets (see
# `test_bookend_prefix_pairs`).
_PD_COUNTS = {
    "conv64_k5_d2": (0, 0),
    "conv128_k7_d12": (0, 6),
    "piper_vocoder": (0, 35),
    "w2v2fe_training_step": (16, 48),
    "dwconv_g32": (0, 0),
    "layernorm_last_axis": (0, 0),
    "attn_qkv_softmax": (0, 0),
    "toy_training_step": (0, 0),
    "resnet18_int8": (0, 8),
    "loss_head_kd": (0, 0),
    "adam_update_fp32": (0, 0),
    "reshape_gather_bwd": (0, 0),
    "neg_1x8": (0, 0),
    "mul_1x8": (0, 0),
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_bookend_prefix_pairs(name):
    """Two-byte prefixes the greedy walk strands: a raw-on-both-bytes
    `05 90` doublet, and `0b 91`, always followed by an `a1 00 b0 03` verb
    (2,663 of 2,663 occurrences corpus-wide). Neither byte can open any other
    form and the guards hold the absorbed bytes to otherwise-raw ones, so
    admitting them is pure addition -- see the README's "Bookend pairs: a
    two-byte prefix class" section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    prefixes = [t for t in toks if t[1] == "P"]
    doublets = [t for t in toks if t[1] == "D"]
    assert (len(prefixes), len(doublets)) == _PD_COUNTS[name], (
        name,
        len(prefixes),
        len(doublets),
    )
    for o, _, a, b, _ in prefixes:
        assert (blob[o], blob[o + 1]) == (0x0B, 0x91) == (a, b), (name, o)
        assert blob[o + 2 : o + 6] == bytes([0xA1, 0x00, 0xB0, 0x03]), (name, o)
    for o, _, a, b, _ in doublets:
        assert (blob[o], blob[o + 1]) == (0x05, 0x90) == (a, b), (name, o)
    # Neither form can split a verb: no verb byte occurs inside either.
    for o, kind, a, b, _ in prefixes + doublets:
        assert a not in mcode.VERBS6 and b not in mcode.VERBS6, (name, kind, o)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [(r["a"], r["b"]) for r in records if r["kind"] == "P"] == [
        (a, b) for _, _, a, b, _ in prefixes
    ]
    assert [(r["a"], r["b"]) for r in records if r["kind"] == "D"] == [
        (a, b) for _, _, a, b, _ in doublets
    ]

    # Admitting the forms buys coverage, and only where they fire.
    without = dict(mcode.FULL_RULE)
    without["pair_prefix"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if sum(_PD_COUNTS[name]):
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The shuffle control: essentially no prefixes or doublets by chance.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = [
        t
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] in ("P", "D")
    ]
    assert len(null) <= max(2, (len(prefixes) + len(doublets)) // 4), (
        name,
        len(prefixes) + len(doublets),
        len(null),
    )


def _synthetic_fx_stream():
    """A hand-built stream exercising the F and X forms: a verb, a short
    unit, a bare pair, then a fixed `01 a4 00 c1 W` unit, a `05 10 e2 0e`
    prefix under its anchored verb, and the verb itself."""
    return bytes.fromhex(
        "a1 00 40 02 00000000"  # V
        "00 08 81 e8"  # S
        "83 62"  # B
        "01 a4 00 c1 23"  # F
        "05 10 e2 0e"  # X
        "a1 00 c0 81 00000000"  # the anchored verb
    )


def test_fixed_head_and_verb_prefix_codec():
    """The F (`01 a4 00 c1 W`) and X (`05 10 e2 0e` + anchored verb) forms
    tokenize, decode and re-encode exactly on a synthetic stream -- the
    committed fixtures carry neither (both were found in larger whisper and
    wav2vec2 builds), so this pins the implementation where no fixture can.
    The real-stream evidence -- 1,690 fixed heads against 1 shuffled, 1,278
    anchored prefixes -- is in the README's "A fixed head with a live tail"
    section."""
    blob = _synthetic_fx_stream()
    toks = mcode.tokenize(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    kinds = [t[1] for t in toks]
    assert kinds == ["V", "S", "B", "F", "X", "V"], kinds
    assert toks[3][2] == 0x23
    records = mcode.decode(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [(r["kind"]) for r in records] == kinds
    assert [r["w"] for r in records if r["kind"] == "F"] == [0x23]
    assert mcode.encode(records) == blob

    # The flags plumb through: off means the forms do not fire.
    plain = dict(mcode.FULL_RULE)
    plain["fixed5"] = False
    plain["vprefix"] = False
    kinds_off = [t[1] for t in mcode.tokenize(blob, start=0, end=len(blob), **plain)]
    assert "F" not in kinds_off and "X" not in kinds_off, kinds_off

    # The X anchor is load-bearing: the same four bytes without the verb
    # after them stay raw.
    unanchored = bytes.fromhex("05 10 e2 0e") + bytes.fromhex("00 08 81 e8")
    toks = mcode.tokenize(unanchored, start=0, end=len(unanchored), **mcode.FULL_RULE)
    assert [t[1] for t in toks] == ["?", "?", "?", "?", "S"], [t[1] for t in toks]

    # Neither form's bytes can hide a verb.
    assert not set(bytes.fromhex("01 a4 00 c1")) & set(mcode.VERBS6)
    assert not set(bytes.fromhex("05 10 e2 0e")) & set(mcode.VERBS6)


# Octet takes per committed stream -- the `04 40 84 18 83 R TT 40`
# template is adjudicated per site, and only the training step's stream
# takes it (once). See `test_octet_wins_only_by_adjudication`.
_E_COUNTS = {
    "conv64_k5_d2": 0,
    "conv128_k7_d12": 0,
    "piper_vocoder": 0,
    "w2v2fe_training_step": 1,
    "dwconv_g32": 0,
    "layernorm_last_axis": 0,
    "attn_qkv_softmax": 0,
    "toy_training_step": 0,
    "resnet18_int8": 0,
    "loss_head_kd": 0,
    "adam_update_fp32": 0,
    "reshape_gather_bwd": 0,
    "neg_1x8": 0,
    "mul_1x8": 0,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_octet_wins_only_by_adjudication(name):
    """The `04 40 84 18 83 R TT 40` template recurs exactly with zero
    shuffled counterparts, but its tail usually completes a genuine-looking
    short unit -- taking it unconditionally was measured to lose net bytes.
    So it is never taken blind: only by lookahead adjudication against
    skipping it, like the verb splits. Pinned here on the training step's
    single take. See the README's "Bookend pairs" section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    takes = [t for t in toks if t[1] == "E"]
    assert len(takes) == _E_COUNTS[name], (name, len(takes))
    for o, _, a, b, _ in takes:
        assert bytes(blob[o : o + 8])[:6] == bytes([0x04, 0x40, 0x84, 0x18, 0x83, a])
        assert bytes(blob[o + 6 : o + 8]) == bytes([b, 0x40]), (name, o)
        assert a not in mcode.VERBS6 and b not in mcode.VERBS6, (name, o)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [(r["a"], r["b"]) for r in records if r["kind"] == "E"] == [
        (a, b) for _, _, a, b, _ in takes
    ]

    # The take and the skip, side by side, on the training step's site.
    if takes:
        o = takes[0][0]
        assert o == 7826, o
        assert bytes(blob[o : o + 8]) == bytes(
            [0x04, 0x40, 0x84, 0x18, 0x83, 0x66, 0x01, 0x40]
        )
        plain = dict(mcode.FULL_RULE)
        plain["octet"] = False
        skipped = [t for t in mcode.tokenize(blob, start=o, end=o + 16, **plain)]
        assert [t[1] for t in skipped] == ["?", "?", "B", "B", "S", "B", "S"], [
            t[1] for t in skipped
        ]

    # Admitting it buys coverage where it takes, and nowhere else.
    without = dict(mcode.FULL_RULE)
    without["octet"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if _E_COUNTS[name]:
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The template never occurs by chance.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = sum(
        1
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] == "E"
    )
    assert null == 0, (name, null)


def _synthetic_template_stream():
    """A hand-built stream with an `01 98 02 83 R 83 0e 05` template: a
    short unit, the template (R = `0x40`), and a bare pair."""
    return bytes.fromhex(
        "00 08 81 e8"  # S
        "01 98 02 83 40 83 0e 05"  # T
        "81 96"  # B
    )


def test_template_with_fused_tail_codec():
    """The `01 98 02 83 R 83 0e 05` template recurs 5,594 times exactly with
    zero shuffled counterparts, but its head is a valid short unit -- so
    like the octet it is taken only by lookahead adjudication, never blind.
    The committed fixtures carry none, so this pins the codec on a synthetic
    stream; the corpus numbers are in the README's "An eight-byte template
    with a fused tail" section."""
    blob = _synthetic_template_stream()
    toks = mcode.tokenize(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [t[1] for t in toks] == ["S", "T", "B"], [t[1] for t in toks]
    assert toks[1][2] == 0x40
    records = mcode.decode(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [r["kind"] for r in records] == ["S", "T", "B"]
    assert [r["r"] for r in records if r["kind"] == "T"] == [0x40]
    assert mcode.encode(records) == blob

    # Without the template the head parses as the short unit it mimics.
    plain = dict(mcode.FULL_RULE)
    plain["template8"] = False
    kinds_off = [t[1] for t in mcode.tokenize(blob, start=0, end=len(blob), **plain)]
    assert kinds_off == ["S", "S", "B", "?", "B"], kinds_off

    # The template's bytes hide no verb.
    assert not set(bytes.fromhex("01 98 02 83 83 0e 05")) & set(mcode.VERBS6)


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_template8_never_regresses_coverage(name):
    """Adjudicated templates can only move bytes from raw escapes into
    recognised forms or leave the walk where it was: per-stream coverage
    with the template on is never below it off. (The fixtures carry no
    templates, so this pins equality there and guards the plumbing.)"""
    blob = _blob(name)
    plain = dict(mcode.FULL_RULE)
    plain["template8"] = False
    covered_plain, _ = mcode.nonzero_coverage(blob, **plain)
    covered_with, _ = mcode.nonzero_coverage(blob)
    assert covered_with >= covered_plain, (name, covered_plain, covered_with)


# C-token counts per committed stream -- `[H][A][B][0xc1][D]` units with
# `H` in `{0x01, 0x30}` and `D` even (see `test_c1_five_programs_a_slot`).
_C_COUNTS = {
    "conv64_k5_d2": 1,
    "conv128_k7_d12": 0,
    "piper_vocoder": 1,
    "w2v2fe_training_step": 7,
    "dwconv_g32": 1,
    "layernorm_last_axis": 0,
    "attn_qkv_softmax": 0,
    "toy_training_step": 2,
    "resnet18_int8": 0,
    "loss_head_kd": 0,
    "adam_update_fp32": 0,
    "reshape_gather_bwd": 0,
    "neg_1x8": 0,
    "mul_1x8": 0,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_c1_five_programs_a_slot(name):
    """A `[H][A][B][0xc1][D]` unit with `H` in `{0x01, 0x30}` and `D` even:
    a short-unit-shaped payload against tag `0xc1`, whose `(A, B)` roam
    over a dozen field-like pairs for `H = 0x01` (concentrated: three pairs
    are 85% of corpus takes, against a scattered shuffle) and stay fixed
    for `H = 0x30`. The even-`D` gate keeps it disjoint from the fixed
    `01 a4 00 c1 W` form, whose tail is always odd. See the README's "The
    `0xc1` five" section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    takes = [t for t in toks if t[1] == "C"]
    assert len(takes) == _C_COUNTS[name], (name, len(takes))
    for o, _, h, a, bd in takes:
        b, d = bd
        assert bytes(blob[o : o + 5]) == bytes([h, a, b, 0xC1, d]), (name, o)
        assert h in (0x01, 0x30) and d % 2 == 0, (name, o)
        for j in range(o + 1, o + 5):
            assert not (
                blob[j] in mcode.VERBS6 and blob[j + 1] == 0 and blob[j + 2] % 0x10 == 0
            ), (name, o, j)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [(r["h"], r["a"], r["b"], r["d"]) for r in records if r["kind"] == "C"] == [
        (h, a, b, d) for _, _, h, a, (b, d) in takes
    ]

    # Admitting the form buys coverage where it fires, and nowhere else.
    without = dict(mcode.FULL_RULE)
    without["c1five"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if _C_COUNTS[name]:
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The shuffle control is honest about small streams: chance shapes fire
    # too, just diffusely (scattered pairs, top count 41 corpus-wide --
    # against 1,561 on the top real pair), so the bound only guards against
    # pathological over-firing.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = sum(
        1
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] == "C"
    )
    assert null <= 2 * len(takes) + 2, (name, len(takes), null)


def _synthetic_repeat_stream():
    """A hand-built stream with a `30 03 XX 03 09` repeat unit: a short
    unit, the repeat (X = `0x1c`), and a bare pair."""
    return bytes.fromhex(
        "00 08 81 e8"  # S
        "30 03 1c 03 09"  # R
        "81 96"  # B
    )


def test_second_repeat_form_codec():
    """The `[0x30][0x03][X][0x03][0x09]` repeat unit repeats its second byte
    at the fourth, the quintet's pair-repeat with one byte instead of two:
    648 occurrences corpus-wide, zero shuffled. The committed fixtures carry
    none, so this pins the codec on a synthetic stream; the corpus numbers
    are in the README's "A second repeat form" section."""
    blob = _synthetic_repeat_stream()
    toks = mcode.tokenize(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [t[1] for t in toks] == ["S", "R", "B"], [t[1] for t in toks]
    assert toks[1][2] == 0x1C
    records = mcode.decode(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [r["kind"] for r in records] == ["S", "R", "B"]
    assert [r["x"] for r in records if r["kind"] == "R"] == [0x1C]
    assert mcode.encode(records) == blob

    # The flag plumbs through: off means the form does not fire.
    plain = dict(mcode.FULL_RULE)
    plain["repeat"] = False
    kinds_off = [t[1] for t in mcode.tokenize(blob, start=0, end=len(blob), **plain)]
    assert kinds_off == ["S", "?", "?", "?", "?", "?", "B"], kinds_off

    # The repeat's bytes hide no verb.
    assert not set(bytes.fromhex("30 03 1c 03 09")) & set(mcode.VERBS6)


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_repeat_never_regresses_coverage(name):
    """The repeat form fires only where the walk emits raw escapes, so
    per-stream coverage with it on is never below it off. (The fixtures
    carry no repeats, so this pins equality there and guards the
    plumbing.)"""
    blob = _blob(name)
    plain = dict(mcode.FULL_RULE)
    plain["repeat"] = False
    covered_plain, _ = mcode.nonzero_coverage(blob, **plain)
    covered_with, _ = mcode.nonzero_coverage(blob)
    assert covered_with >= covered_plain, (name, covered_plain, covered_with)


def _synthetic_stutter_stream():
    """A hand-built stream with a stuttered pair: a short unit ending in
    `90 03`, its `90 03` echo, and a bare pair."""
    return bytes.fromhex(
        "00 04 90 03"  # S
        "90 03"  # Y
        "81 96"  # B
    )


def test_stuttered_pair_codec():
    """A short unit ending in `90 03` followed by another `90 03`: the pair
    stutters (5,502 of them corpus-wide, eleven of anything else), and no
    other form can open at the echo. The committed fixtures carry none, so
    this pins the codec on a synthetic stream; the corpus numbers are in
    the README's "The stuttered pair" section."""
    blob = _synthetic_stutter_stream()
    toks = mcode.tokenize(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [t[1] for t in toks] == ["S", "Y", "B"], [t[1] for t in toks]
    records = mcode.decode(blob, start=0, end=len(blob), **mcode.FULL_RULE)
    assert [r["kind"] for r in records] == ["S", "Y", "B"]
    assert [(r["a"], r["b"]) for r in records if r["kind"] == "Y"] == [(0x90, 0x03)]
    assert mcode.encode(records) == blob

    # The flag plumbs through: off means the echo stays raw.
    plain = dict(mcode.FULL_RULE)
    plain["stutter"] = False
    kinds_off = [t[1] for t in mcode.tokenize(blob, start=0, end=len(blob), **plain)]
    assert kinds_off == ["S", "?", "?", "B"], kinds_off


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_stutter_never_regresses_coverage(name):
    """The echo converts only raw escapes with everything around it parsing
    identically, so per-stream coverage with it on is never below it off.
    (The fixtures carry no stutters, so this pins equality there and guards
    the plumbing.)"""
    blob = _blob(name)
    plain = dict(mcode.FULL_RULE)
    plain["stutter"] = False
    covered_plain, _ = mcode.nonzero_coverage(blob, **plain)
    covered_with, _ = mcode.nonzero_coverage(blob)
    assert covered_with >= covered_plain, (name, covered_plain, covered_with)


# N-token counts per committed stream -- six-byte `09 0c 80 fe 01 01`
# prefixes under `a1 00 d0 0c` verbs (see `test_six_byte_prefix`).
_N_COUNTS = {
    "conv64_k5_d2": 0,
    "conv128_k7_d12": 0,
    "piper_vocoder": 0,
    "w2v2fe_training_step": 7,
    "dwconv_g32": 0,
    "layernorm_last_axis": 0,
    "attn_qkv_softmax": 0,
    "toy_training_step": 0,
    "resnet18_int8": 0,
    "loss_head_kd": 0,
    "adam_update_fp32": 0,
    "reshape_gather_bwd": 0,
    "neg_1x8": 0,
    "mul_1x8": 0,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_six_byte_prefix(name):
    """A six-byte `09 0c 80 fe 01 01` prefix, always followed by an
    `a1 00 d0 0c` verb (all 2,741 occurrences corpus-wide, zero shuffled
    counterparts): a third prefix length alongside the two- and four-byte
    ones, with the same positional anchor. Its head opens no other form and
    the anchored verb parses identically after it, so it converts only raw
    escapes. See the README's "A six-byte prefix" section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    takes = [t for t in toks if t[1] == "N"]
    assert len(takes) == _N_COUNTS[name], (name, len(takes))
    for o, _, _, _, _ in takes:
        assert bytes(blob[o : o + 6]) == bytes([0x09, 0x0C, 0x80, 0xFE, 0x01, 0x01]), (
            name,
            o,
        )
        assert bytes(blob[o + 6 : o + 10]) == bytes([0xA1, 0x00, 0xD0, 0x0C]), (
            name,
            o,
        )
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [r["kind"] for r in records if r["kind"] == "N"] == ["N"] * len(takes)
    assert mcode.encode(records) == blob[lo:hi]

    # Admitting the form buys coverage where it fires, and nowhere else.
    without = dict(mcode.FULL_RULE)
    without["nprefix"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if _N_COUNTS[name]:
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The anchored template never occurs by chance.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = sum(
        1
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] == "N"
    )
    assert null == 0, (name, null)


_NEW_FAMILIES = ("dwconv_g32", "layernorm_last_axis", "attn_qkv_softmax")


@pytest.mark.parametrize("name", _NEW_FAMILIES)
def test_new_families_use_no_new_instruction_forms(name):
    """Depthwise convolution, LayerNormalization and a Softmax attention
    fragment -- three op families the fixtures never covered -- introduce no
    verb and no tag beyond the closed sets the CNN, transformer and vocoder
    builds already used. Op-type differences live in operand values, not in
    new forms. See the README's "Three new families, no new forms" section.
    Needs neither Docker nor a device."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    verbs = {t[2] for t in toks if t[1] == "V"}
    assert verbs <= mcode.VERBS6, (name, verbs)
    stags = set()
    for t in toks:
        if t[1] == "S":
            o, _, p, _, _ = t
            stags.add(blob[o + p + 2])
    assert stags <= mcode.ALL_TAGS | {0xA1}, (name, stags)
    btags = {t[2] for t in toks if t[1] == "B"}
    assert btags <= mcode.ALL_TAGS | {0xA1, 0xC1, 0xE1}, (name, btags)
    covered, _ = mcode.nonzero_coverage(blob)
    assert covered >= 0.95, (name, covered)


_TRAINING_FAMILIES = (
    "toy_training_step",
    "resnet18_int8",
    "loss_head_kd",
    "adam_update_fp32",
)


@pytest.mark.parametrize("name", _TRAINING_FAMILIES)
def test_training_graph_streams_use_no_new_instruction_forms(name):
    """A full distillation training step (forward + KD loss + backward +
    Adam), an INT8 ResNet18, the KD soft-loss head and an Adam update --
    the first training-graph streams in this corpus -- introduce no verb
    and no tag beyond the closed sets, same bar as the inference-only
    families above. Op-type differences live in operand values, not in
    new forms. Needs neither Docker nor a device."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    verbs = {t[2] for t in toks if t[1] == "V"}
    assert verbs <= mcode.VERBS6, (name, verbs)
    stags = set()
    for t in toks:
        if t[1] == "S":
            o, _, p, _, _ = t
            stags.add(blob[o + p + 2])
    assert stags <= mcode.ALL_TAGS | {0xA1}, (name, stags)
    btags = {t[2] for t in toks if t[1] == "B"}
    assert btags <= mcode.ALL_TAGS | {0xA1, 0xC1, 0xE1}, (name, btags)
    covered, _ = mcode.nonzero_coverage(blob)
    assert covered >= 0.95, (name, covered)


_GAP_STREAMS = {
    # The last sub-floor stream, with its exact unexplained runs pinned:
    # lone singles the abutting-trailer form does not cover plus the
    # still-unformed `08`. Fails loudly in either direction: a regression
    # adds runs, a future form decoding these removes them.
    "reshape_mul_gap": (
        "coverage: only 93.8% of non-zero bytes explained "
        "[(284, 285), (321, 322), (465, 468), (470, 471)]"
    ),
    "reshape_gather_bwd": (),
}


@pytest.mark.parametrize("name", sorted(_GAP_STREAMS))
def test_known_gap_streams_hold_status_quo(name):
    """The three backward-slice streams below the families bar above hold
    exactly their recorded status -- no more, no less. `reshape_gather_bwd`
    passes `check()` (0.9443 coverage, above the 0.94 floor) with the same
    unexplained singles; the pair sits just under the floor with the runs
    listed. Needs neither Docker nor a device."""
    blob = _blob(name)
    expected = _GAP_STREAMS[name]
    assert mcode.check(blob) == ([expected] if expected else [])
    covered, runs = mcode.nonzero_coverage(blob)
    for a, b in runs:
        assert (b - a) <= 12, (name, (a, b))
    if expected:
        assert covered < 0.94, (name, covered)
    else:
        assert 0.94 <= covered < 0.95, (name, covered)


# Terminal takes per committed stream -- a `0b 01` pair closing an S-unit
# rhythm (`82 08` immediately before) ahead of zero padding. Device-mapped:
# zeroing the pair faults the NPU while zeroing a lone `08` nearby runs
# bit-identical (see test_splice_gap_bytes_split_inert_vs_fault in
# test_axera_mcode_structure.py).
_L_COUNTS = {
    "conv64_k5_d2": 1,
    "conv128_k7_d12": 0,
    "piper_vocoder": 1,
    "w2v2fe_training_step": 0,
    "dwconv_g32": 1,
    "layernorm_last_axis": 1,
    "attn_qkv_softmax": 0,
    "toy_training_step": 0,
    "resnet18_int8": 0,
    "loss_head_kd": 1,
    "adam_update_fp32": 1,
    "reshape_gather_bwd": 1,
    "neg_1x8": 1,
    "mul_1x8": 1,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_terminal_pair_closes_before_padding(name):
    """A two-byte `0b 01` unit: the `82 08` tail of a complete short unit
    immediately before it, eight zero bytes immediately after. Found by
    clustering the unexplained bytes of training-graph and probe streams
    from an AX650N, and validated the way every other form was: it fires
    only where no other form matches (its head opens nothing), so no
    stream regresses, and zero shuffled counterparts across all fourteen
    fixtures. See the README's "A terminal pair before the padding"
    section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    takes = [t for t in toks if t[1] == "L"]
    assert len(takes) == _L_COUNTS[name], (name, len(takes))
    for o, _, _, _, _ in takes:
        assert bytes(blob[o : o + 2]) == bytes([0x0B, 0x01]), (name, o)
        assert bytes(blob[o - 2 : o]) == bytes([0x82, 0x08]), (name, o)
        assert bytes(blob[o + 2 : o + 10]) == b"\x00" * 8, (name, o)
        assert 0x0B not in mcode.VERBS6, (name, o)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [r["at"] for r in records if r["kind"] == "L"] == [
        o for o, _, _, _, _ in takes
    ]

    # Admitting the form buys coverage, and only where it fires.
    without = dict(mcode.FULL_RULE)
    without["terminal"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if _L_COUNTS[name]:
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The shuffle control: no terminal pairs by chance.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = sum(
        1
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] == "L"
    )
    assert null == 0, (name, null)


# Abutting-trailer takes per committed stream -- a lone value byte in
# {0x23, 0x24, 0x26, 0x2B} with two zero bytes before it, abutting the
# next segment's `a7 00` marker head. Device-mapped (zeroing faults the
# NPU); 36 exact recurrences corpus-wide against zero shuffled.
_A_COUNTS = {
    "conv64_k5_d2": 3,
    "conv128_k7_d12": 4,
    "piper_vocoder": 3,
    "w2v2fe_training_step": 3,
    "dwconv_g32": 3,
    "layernorm_last_axis": 1,
    "attn_qkv_softmax": 4,
    "toy_training_step": 4,
    "resnet18_int8": 4,
    "loss_head_kd": 1,
    "adam_update_fp32": 1,
    "reshape_gather_bwd": 2,
    "reshape_mul_gap": 1,
    "reshape_matmul_gap": 3,
    "neg_1x8": 1,
    "mul_1x8": 1,
}


@pytest.mark.parametrize("name", sorted(_BLOBS))
def test_trailer_single_abuts_next_segment(name):
    """A lone trailer single: value in `{0x23, 0x24, 0x26, 0x2B}`, two
    zero bytes immediately before it, the next segment's `a7 00` marker
    head immediately after. Found by census over every non-final segment
    of 62 streams from an AX650N (36 sites, zero shuffled counterparts),
    and validated the way every other form was: it fires only where no
    other form matches, so no stream regresses. Zeroing one faults the
    NPU -- live epilogue bytes, not padding. See the README's "Trailer
    singles abutting the next segment" section."""
    blob = _blob(name)
    lo, hi = mcode.stream_bounds(blob)
    toks = mcode.tokenize(blob, start=lo, end=hi, **mcode.FULL_RULE)
    takes = [t for t in toks if t[1] == "A"]
    assert len(takes) == _A_COUNTS[name], (name, len(takes))
    for o, _, v, _, _ in takes:
        assert blob[o] == v, (name, o)
        assert v in (0x23, 0x24, 0x26, 0x2B), (name, o)
        assert bytes(blob[o - 2 : o]) == b"\x00" * 2, (name, o)
        assert bytes(blob[o + 1 : o + 3]) == bytes([0xA7, 0x00]), (name, o)
        assert v not in mcode.VERBS6, (name, o)
    records = mcode.decode(blob, start=lo, end=hi, **mcode.FULL_RULE)
    assert [r["at"] for r in records if r["kind"] == "A"] == [
        o for o, _, _, _, _ in takes
    ]

    # Admitting the form buys coverage, and only where it fires.
    without = dict(mcode.FULL_RULE)
    without["trailer"] = False
    covered_without, _ = mcode.nonzero_coverage(blob, **without)
    covered_with, _ = mcode.nonzero_coverage(blob)
    if _A_COUNTS[name]:
        assert covered_with > covered_without, (name, covered_without, covered_with)
    else:
        assert covered_with == covered_without, (name, covered_without, covered_with)

    # The shuffle control: no abutting singles by chance.
    bulk = bytearray(blob[lo:hi])
    random.Random(0).shuffle(bulk)
    shuffled = bytes(blob[:lo]) + bytes(bulk) + bytes(blob[hi:])
    null = sum(
        1
        for t in mcode.tokenize(shuffled, start=lo, end=hi, **mcode.FULL_RULE)
        if t[1] == "A"
    )
    assert null == 0, (name, null)
