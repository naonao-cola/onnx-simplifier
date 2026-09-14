"""The mcode checker, run on committed blobs from real Pulsar2 builds.

Everything else in this project's Axera work needs a Pulsar2 Docker image or
an AX650N card. This file does not: it reads three mcodes compiled earlier and
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
    assert mcode.structured_share(records) >= 0.91


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
