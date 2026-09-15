"""The mcode codec: reading, writing and checking Axera's compiled
command-queue programs.

`.axmodel` files carry two blobs this project reverse-engineered -- the
weight table (`npu_params`) and the **mcode**, the NPU command queue, stored
under a `neu_key`-named initializer. Everything here is the confirmed-real
half of that work, lifted out of `tests/test_axera_mcode_structure.py` so it
can run without a Pulsar2 Docker image or an AX650N card. See
`scripts/axera/README.md` for the narrative and for what is *not* known.

The pieces:

* `tokenize` / `decode` / `encode` -- a lossless codec. `encode(decode(m))`
  reproduces the stream byte for byte, which is what makes an edit safe.
* `tail_tables` / `segments` / `stream_bounds` -- the FlatBuffers tail, which
  is a loader manifest: the runtime validates its per-segment word counts.
* `check` -- every structural invariant confirmed on real hardware, as a
  list of violations. This is the CI-facing entry point.

There is deliberately no evaluator here. What each verb *computes* is not
known: most operand slots hold allocator output (addresses and sizes chosen
per build), and no verb's datapath semantics have been established. `check`
validates form, not arithmetic.
"""

import json
import struct

import onnx

VERBS = {0xA1, 0xA2, 0xA3, 0xA8, 0xA9}

WIDE_TAGS = frozenset(
    {0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x89, 0x8A, 0x8B, 0x8C, 0x8D}
    | {0x94, 0x95, 0x9B, 0x9C, 0x9D, 0x9F}
)
"""The 17 short-unit tag bytes that beat a shuffled *explained-bytes* null
by >= 2x in both real models -- the README's "Fourth correction" section.
Superseded by `ALL_TAGS`: the fifth correction's conditional-parity test
showed the rejected tags were false rejections of rare forms."""

VERBS6 = frozenset(VERBS | {0xA7})
"""The five verbs plus `a7`, the segment-marker verb every stream segment
opens with and `llm_build` op programs use freely -- see the README's "The
tail is the segment table" section. Pass as `verbs=` to `tokenize`."""

ALL_TAGS = frozenset(range(0x81, 0xA0))
"""Every byte below the verb range: the full short-unit tag set. For every
tag the byte after it is even in ~100% of real units against ~60% shuffled
-- see the README's "Fifth correction" section."""

FULL_RULE = dict(
    tags=ALL_TAGS | {0xA1},
    pmax=4,
    bare=True,
    extra_byte_tags={0x9F},
    verbs=VERBS6,
    odd_tags={0xC1, 0xE1},
    companion=True,
    quintet=True,
    lookahead=64,
    pair_prefix=True,
    fixed5=True,
    vprefix=True,
    octet=True,
    template8=True,
    c1five=True,
    repeat=True,
    stutter=True,
    nprefix=True,
    terminal=True,
    trailer=True,
)
"""Every validated form: all tags, p <= 4, bare pairs, the 0x9f extra byte,
six verbs -- the README's "The tail is the segment table" section -- plus
the `[04][a][b][a][b]` quintet ("A five-byte form that programs its pair
twice"), the bookend pairs ("Bookend pairs: a two-byte prefix class"), a
six-byte prefix ("A six-byte prefix"), the fixed head with a live tail
("A fixed head with a live tail"), the two
adjudicated templates ("An eight-byte template with a fused tail"), the
`0xc1` five ("The `0xc1` five"), the second repeat form ("A second
repeat form"), the stuttered pair ("The stuttered pair") and the terminal
`0b 01` pair ("A terminal pair before the padding"), plus the abutting
trailer single ("Trailer singles abutting the next segment").
`lookahead` adjudicates the overlaps the greedy walk cannot see past
("Adjudicating the overlaps ..."); 64 bytes is where the corpus-wide
gain converges (32 is slightly worse, 128 no better)."""

TAIL_VECTOR = bytes.fromhex("05000000200000002c000000500000007400000098000000")
"""The FlatBuffers vector of five table offsets that opens an mcode blob's
tail -- see the README's "The op programs are fully tokenized" section."""


def tokenize(
    mcode,
    start=297,
    end=None,
    tags=None,
    pmax=3,
    bare=False,
    extra_byte_tags=frozenset(),
    verbs=None,
    odd_tags=frozenset(),
    companion=False,
    quintet=False,
    lookahead=0,
    pair_prefix=False,
    fixed5=False,
    vprefix=False,
    octet=False,
    template8=False,
    c1five=False,
    repeat=False,
    stutter=False,
    nprefix=False,
    terminal=False,
    trailer=False,
):
    """Tokenize an mcode blob's bulk with every validated form -- the 8/7-byte
    verb instructions, the width-rule short units `[p][p+1 bytes][tag]
    [register]` (p+4 bytes) and, with `bare=True`, the payload-less 2-byte
    `[tag][register]` pair -- stepping one unknown byte otherwise. Units
    whose tag is in `extra_byte_tags` take one extra trailing byte (the
    sixth correction: tag 0x9f). Returns `(byte_offset, kind, a, b, c)`
    tuples: kind 'V' (a=verb, b=xx, c=yy), 'S' (a=prefix, b=tag, c=first
    payload byte), 'B' (a=tag, b=register), 'W' (a companion write:
    a=X, b=field, c=bank), 'Q' (a quintet `[04][a][b][a][b]`: a five-byte
    unit repeating its last two payload bytes -- see the README's "A
    five-byte form that programs its pair twice" section), 'P' (a `0b 91`
    prefix under its anchored verb), 'D' (a raw-on-both-bytes `05 90`
    doublet -- both see the README's "Bookend pairs: a two-byte prefix
    class" section), 'N' (a six-byte `09 0c 80 fe 01 01` prefix under its
    anchored verb -- see "A six-byte prefix"), 'F' (a fixed `01 a4 00 c1 W` unit: the middle never
    varies, only the last byte does), 'X' (a four-byte `05 10 e2 0e` verb
    prefix -- both see the README's "A fixed head with a live tail"
    section), 'E' (an octet `04 40 84 18 83 R TT 40`, taken only when the
    lookahead adjudicates its overlap in its favour -- see "Bookend pairs"
    and "Adjudicating the overlaps ..."), 'T' (an 8-byte
    `01 98 02 83 R 83 0e 05` template, taken only by the same adjudication
    -- see "An eight-byte template with a fused tail"), 'C' (a
    `[H][A][B][0xc1][D]` unit with `H` in `{0x01, 0x30}` and `D` even --
    see "The `0xc1` five"), 'R' (a `[0x30][0x03][X][0x03][0x09]` repeat
    unit -- see "A second repeat form"),     'Y' (a stuttered `90 03` echo
    after a short unit ending in it -- see "The stuttered pair"), 'L' (a
    terminal `0b 01` pair closing an S-unit rhythm before zero padding --
    see "A terminal pair before the padding"), 'A' (a lone trailer single
    abutting the next segment's `a7` marker -- see "Trailer singles
    abutting the next segment") or '?'
    (a=byte).

    Where a short unit ending in `a1 00` overlaps a verb beginning there,
    the greedy walk cannot tell a genuine tag from a swallowed verb head by
    looking at those two bytes alone. With `lookahead > 0` both parses are
    walked for that many bytes and the one leaving fewer unexplained
    non-zero bytes wins (ties keep the short unit) -- see the README's
    "Adjudicating the overlaps the greedy walk cannot see past" section.
    `lookahead=0` (the default) is the historical greedy walk, byte for
    byte.

    The defaults
    (tags 0x81..0x84, p <= 3, no bare pairs, no extra bytes, stop 252 bytes
    before the end) are the original narrow rule; `tags=ALL_TAGS, pmax=4,
    bare=True, extra_byte_tags={0x9F}` is the corrected one. See the
    README's "The layout that explains all of it" and "Fourth" .. "Sixth
    correction" sections."""
    tags = set(range(0x81, 0x85)) if tags is None else set(tags)
    extra_byte_tags = set(extra_byte_tags)
    verbs = VERBS if verbs is None else frozenset(verbs)
    end = len(mcode) - 252 if end is None else end

    def is_verb(i):
        return (
            i + 3 < len(mcode)
            and mcode[i] in verbs
            and mcode[i + 1] == 0
            and mcode[i + 2] % 0x10 == 0
        )

    def companion_at(i):
        """A 7-byte `[X][field][bank][32-bit operand]` write, recognised only
        when the next 8 bytes are an `a1` verb writing the adjacent slot --
        the same bank one field higher, or the first field of the next bank.
        That anchor never fires on a shuffled stream (see the README's "a
        7-byte write that fills the slot below the next one")."""
        if not companion or i + 15 > end:
            return False
        field, bank = mcode[i + 1], mcode[i + 2]
        if field % 0x10 or mcode[i + 7] != 0xA1 or mcode[i + 8] != 0:
            return False
        nfield, nbank = mcode[i + 9], mcode[i + 10]
        if nfield % 0x10:
            return False
        return (bank == nbank and (nfield - field) % 0x100 == 0x10) or (
            field == 0xF0 and nfield == 0x00 and nbank == bank + 1
        )

    def short_len(i):
        if i >= len(mcode):
            return 0
        p = mcode[i]
        if p <= pmax and i + p + 3 < len(mcode) and mcode[i + p + 2] in tags:
            return p + 4 + (1 if mcode[i + p + 2] in extra_byte_tags else 0)
        return 0

    def quintet_at(i):
        """A 5-byte `[04][a][b][a][b]` unit -- the last two payload bytes
        repeat the middle two. It fires only where no other form matches
        (0x04 is not a verb, a tag, or a valid width prefix with a tag at
        +6 -- that check runs first), so admitting it can only convert raw
        escapes, never steal a recognised unit. Across 68 real streams no
        (a, b) pair contains a verb byte, so it has never split one either.
        See the README's "A five-byte form that programs its pair twice"
        section."""
        return (
            quintet
            and i + 4 < len(mcode)
            and mcode[i] == 0x04
            and mcode[i + 1] == mcode[i + 3]
            and mcode[i + 2] == mcode[i + 4]
        )

    def octet_at(i):
        """An 8-byte `04 40 84 18 83 R TT 40` template: fixed bookends around
        two bare pairs (R always even, `0x42..0x58`) and a small tail tag.
        The template recurs 2,252 times exactly with zero shuffled
        counterparts, but its tail `TT 40` usually completes a
        genuine-looking short unit, so it is never taken blind -- only by
        lookahead adjudication in the walk below, where it wins solely by
        leaving less unexplained behind. See the README's "Bookend pairs"
        section."""
        return (
            octet
            and i + 7 < len(mcode)
            and mcode[i] == 0x04
            and mcode[i + 1] == 0x40
            and mcode[i + 2] == 0x84
            and mcode[i + 3] == 0x18
            and mcode[i + 4] == 0x83
            and mcode[i + 7] == 0x40
            and not quintet_at(i + 6)
            and not companion_at(i + 6)
            and not companion_at(i + 7)
        )

    def template8_at(i):
        """An 8-byte `01 98 02 83 R 83 0e 05` template: a short-unit head
        (`p = 1`, payload, tag `0x83`) fused with a bare pair and a trailing
        `05`. Every one of the 5,594 occurrences corpus-wide is preceded by
        exactly these bytes, with zero shuffled counterparts -- but a plain
        short unit plus a bare pair parses the same bytes too, so like the
        octet it is taken only by lookahead adjudication, never blind. See
        the README's "A template with a fused tail" section."""
        return (
            template8
            and i + 7 < len(mcode)
            and mcode[i] == 0x01
            and mcode[i + 1] == 0x98
            and mcode[i + 2] == 0x02
            and mcode[i + 3] == 0x83
            and mcode[i + 5] == 0x83
            and mcode[i + 6] == 0x0E
            and mcode[i + 7] == 0x05
            and not companion_at(i)
            and not companion_at(i + 1)
        )

    def c1five_at(i):
        """A five-byte `[H][A][B][0xc1][D]` unit with `H` in `{0x01, 0x30}`
        and `D` even: a short-unit-shaped payload `[A][B]` against tag
        `0xc1`, whose `(A, B)` roam over a dozen field-like pairs for `H =
        0x01` and stay fixed at `(0x03, 0xc0)` for `H = 0x30`. It fires only
        where no other form matches (`0xc1` is not an admitted width tag),
        so the walk emits raw escapes there today; the even-`D` gate keeps
        it disjoint from the fixed `01 a4 00 c1 W` form, whose tail is
        always odd. See the README's "The `0xc1` five" section."""
        return (
            c1five
            and i + 4 < len(mcode)
            and mcode[i] in (0x01, 0x30)
            and mcode[i + 3] == 0xC1
            and mcode[i + 4] % 2 == 0
        )

    def repeat_at(i):
        """A five-byte `[0x30][0x03][X][0x03][0x09]` unit: fixed frame with
        one live slot, repeating its second byte at the fourth (the quintet
        repeats a pair; this repeats one byte). It fires only where no
        other form matches (`0x30` is not a verb, a tag or a width prefix),
        so the walk emits raw escapes there today. See the README's "A
        second repeat form" section."""
        return (
            repeat
            and i + 4 < len(mcode)
            and mcode[i] == 0x30
            and mcode[i + 1] == 0x03
            and mcode[i + 3] == 0x03
            and mcode[i + 4] == 0x09
        )

    def pair_prefix_at(i):
        """A two-byte prefix standing immediately before the unit it
        modifies: `05 90`, always raw on both bytes, or `0b 91`, always raw
        and always followed by an `a1 00 b0 03` verb. Neither byte can open
        and always followed by an `a1 00 b0 03` verb. Neither byte can open
        any other form (both heads are outside the verb, tag and prefix
        sets), and the guards hold the bytes they absorb to otherwise-raw
        ones -- `05 90` only when no bare pair starts at the `90` (its
        register byte is odd there), `0b 91` only under its anchored verb
        and never overlapping a companion. See the README's "Bookend pairs:
        a two-byte prefix class" section."""
        if i + 1 >= len(mcode):
            return False
        if not pair_prefix:
            return False
        if mcode[i] == 0x05 and mcode[i + 1] == 0x90:
            return (i + 2 >= end or mcode[i + 2] % 2 == 1) and not companion_at(i + 1)
        return (
            mcode[i] == 0x0B
            and mcode[i + 1] == 0x91
            and i + 5 < len(mcode)
            and mcode[i + 2] == 0xA1
            and mcode[i + 3] == 0
            and mcode[i + 4] == 0xB0
            and mcode[i + 5] == 0x03
            and not companion_at(i + 1)
        )

    def nprefix_at(i):
        """A six-byte `09 0c 80 fe 01 01` prefix, always followed by an
        `a1 00 d0 0c` verb (all 2,741 occurrences corpus-wide, zero shuffled
        counterparts). Like the two-byte prefixes, its head opens no other
        form and the anchored verb parses identically after it, so it
        converts only raw escapes. See the README's "A six-byte prefix"
        section."""
        return (
            nprefix
            and i + 11 < len(mcode)
            and mcode[i] == 0x09
            and mcode[i + 1] == 0x0C
            and mcode[i + 2] == 0x80
            and mcode[i + 3] == 0xFE
            and mcode[i + 4] == 0x01
            and mcode[i + 5] == 0x01
            and mcode[i + 6] == 0xA1
            and mcode[i + 7] == 0
            and mcode[i + 8] == 0xD0
            and mcode[i + 9] == 0x0C
        )

    def fixed5_at(i):
        """A fixed five-byte `01 a4 00 c1 W` unit: the middle never varies,
        only the last byte does (almost always `0x23`/`0x25`). It fires only
        where no other form matches -- `0x01` is not a verb, a tag or a
        width prefix with a tag at +3 -- so the walk emits a raw escape
        there today; the `c1 W` tail re-parses after it exactly as before
        (a bare odd pair), which is why admitting it only ever converts raw
        escapes. See the README's "A fixed head with a live tail" section."""
        return (
            fixed5
            and i + 4 < len(mcode)
            and mcode[i] == 0x01
            and mcode[i + 1] == 0xA4
            and mcode[i + 2] == 0
            and mcode[i + 3] == 0xC1
        )

    def vprefix_at(i):
        """A four-byte `05 10 e2 0e` prefix, always followed by an
        `a1 00 c0 81` verb (96% of those verbs take it). Like the two-byte
        prefixes, its head opens no other form and the anchored verb parses
        identically after it, so it converts only raw escapes. See the
        README's "A fixed head with a live tail" section."""
        return (
            vprefix
            and i + 7 < len(mcode)
            and mcode[i] == 0x05
            and mcode[i + 1] == 0x10
            and mcode[i + 2] == 0xE2
            and mcode[i + 3] == 0x0E
            and mcode[i + 4] == 0xA1
            and mcode[i + 5] == 0
            and mcode[i + 6] == 0xC0
            and mcode[i + 7] == 0x81
        )

    def terminal_at(i):
        """A terminal `0b 01` pair closing an S-unit rhythm before zero
        padding: the two bytes before it are a complete short unit's tail
        (`82 08`), the eight bytes after it are zero. Device-mapped on the
        card (zeroing the pair faults the NPU, zeroing a lone `08` nearby
        runs bit-identical), 42 exact recurrences corpus-wide against zero
        shuffled counterparts. Like every other form here its head opens
        no other rule (`0x0b` alone matches nothing), so it converts only
        raw escapes. See the README's "A terminal pair before the padding"
        section."""
        return (
            terminal
            and i >= 2
            and i + 9 < len(mcode)
            and mcode[i] == 0x0B
            and mcode[i + 1] == 0x01
            and mcode[i - 2] == 0x82
            and mcode[i - 1] == 0x08
            and mcode[i + 2 : i + 10] == b"\x00" * 8
        )

    def trailer_single_at(i):
        """A lone trailer single abutting the next segment's `a7` marker:
        two zero bytes immediately before it, the `a7 00` marker head
        immediately after, and a value in the corpus-observed set
        {0x23, 0x24, 0x26, 0x2B}. Device-mapped (zeroing faults the NPU),
        36 exact recurrences corpus-wide against zero shuffled
        counterparts, and its head opens no other form, so it converts
        only raw escapes. See the README's "Trailer singles abutting the
        next segment" section."""
        return (
            trailer
            and i >= 2
            and i + 3 < len(mcode)
            and mcode[i] in (0x23, 0x24, 0x26, 0x2B)
            and mcode[i - 2 : i] == b"\x00" * 2
            and mcode[i + 1] == 0xA7
            and mcode[i + 2] == 0
            and mcode[i + 3] % 0x10 == 0
        )

    def _plain_step(i):
        """One greedy step: `(token, next offset)`, no adjudication. With
        `lookahead=0` the walk below is exactly this step repeated, byte for
        byte the historical walk."""
        if companion_at(i):
            return (i, "W", mcode[i], mcode[i + 1], mcode[i + 2]), i + 7
        if is_verb(i):
            n = (
                7
                if (
                    not (is_verb(i + 8) or short_len(i + 8))
                    and (is_verb(i + 7) or short_len(i + 7))
                )
                else 8
            )
            return (i, "V", mcode[i], mcode[i + 2], mcode[i + 3]), i + n
        sl = short_len(i)
        if sl:
            return (i, "S", mcode[i], mcode[i + sl - 2], mcode[i + 1]), i + sl
        if quintet_at(i):
            return (i, "Q", mcode[i + 1], mcode[i + 2], 0), i + 5
        if pair_prefix_at(i):
            kind = "P" if mcode[i] == 0x0B else "D"
            return (i, kind, mcode[i], mcode[i + 1], 0), i + 2
        if terminal_at(i):
            return (i, "L", 0, 0, 0), i + 2
        if trailer_single_at(i):
            return (i, "A", mcode[i], 0, 0), i + 1
        if nprefix_at(i):
            return (i, "N", 0, 0, 0), i + 6
        if fixed5_at(i):
            return (i, "F", mcode[i + 4], 0, 0), i + 5
        if vprefix_at(i):
            return (i, "X", 0, 0, 0), i + 4
        if c1five_at(i):
            return (i, "C", mcode[i], mcode[i + 1], (mcode[i + 2], mcode[i + 4])), i + 5
        if repeat_at(i):
            return (i, "R", mcode[i + 2], 0, 0), i + 5
        if bare and i + 1 < end and mcode[i] in tags and mcode[i + 1] % 2 == 0:
            n = 2 + (1 if mcode[i] in extra_byte_tags else 0)
            return (i, "B", mcode[i], mcode[i + 1], 0), i + n
        if bare and i + 1 < end and mcode[i] in odd_tags and mcode[i + 1] % 2 == 1:
            # Tags with bit 6 set (0xc1, 0xe1) pair with an *odd* register byte.
            return (i, "B", mcode[i], mcode[i + 1], 0), i + 2
        return (i, "?", mcode[i], 0, 0), i + 1

    def _simulate_cost(i, stop):
        """Unexplained non-zero bytes a plain walk leaves between `i` and
        `stop` -- the score each side of an overlap is judged by."""
        cost = 0
        while i < min(stop, end):
            tok, i = _plain_step(i)
            if tok[1] == "?" and mcode[tok[0]]:
                cost += 1
        return cost

    out, i = [], start
    while i < end:
        if (
            lookahead and octet and octet_at(i)
        ):  # The octet template overlaps the short unit starting at its
            # tail: taking the template breaks that unit, skipping it leaves
            # the bookends raw. Walk both parses; the template wins only by
            # leaving less unexplained behind, ties skipping it.
            cost_take = _simulate_cost(i + 8, i + 8 + lookahead)
            cost_skip = (1 if mcode[i] else 0) + _simulate_cost(
                i + 1, i + 1 + lookahead
            )
            if cost_take < cost_skip:
                out.append((i, "E", mcode[i + 5], mcode[i + 6], 0))
                i += 8
                continue
        tok, nxt = _plain_step(i)
        if (
            lookahead
            and tok[1] == "S"
            and mcode[tok[0] + tok[2] + 2] == 0xA1
            and mcode[tok[0] + tok[2] + 3] == 0
            and is_verb(tok[0] + tok[2] + 2)
        ):
            # A short unit ending in `a1 00` where a verb begins: either the
            # tag is real or the `00` is the verb's second byte. Walk both
            # parses for `lookahead` bytes; the verb-first one wins only by
            # leaving less unexplained behind. Ties keep the short unit, so
            # this never fires without evidence.
            cost_take = _simulate_cost(nxt, nxt + lookahead)
            cost_skip = (1 if mcode[i] else 0) + _simulate_cost(
                i + 1, i + 1 + lookahead
            )
            if cost_skip < cost_take:
                out.append((i, "?", mcode[i], 0, 0))
                i += 1
                continue
        if lookahead and template8 and tok[1] == "S" and template8_at(i):
            # The template's head is a valid short unit, so the greedy walk
            # always takes it and strands the tail. Walk both parses; the
            # template wins only by leaving less unexplained behind, ties
            # keeping the short unit.
            cost_keep = _simulate_cost(nxt, nxt + lookahead)
            cost_take = _simulate_cost(i + 8, i + 8 + lookahead)
            if cost_take < cost_keep:
                out.append((i, "T", mcode[i + 4], 0, 0))
                i += 8
                continue
        if stutter and tok[1] == "S" and nxt + 1 < len(mcode):
            # A short unit ending in `90 03` followed by another `90 03`:
            # the pair stutters (5,502 of them corpus-wide, eleven of
            # anything else), and no other form can open at the echo (`0x90`
            # needs an even register after it, `0x03` is not a prefix). The
            # echo walks as a `Y` record; everything around it parses
            # exactly as before, so this converts only raw escapes. See the
            # README's "The stuttered pair" section.
            if (
                mcode[nxt - 2] == 0x90
                and mcode[nxt - 1] == 0x03
                and mcode[nxt] == 0x90
                and mcode[nxt + 1] == 0x03
                and not companion_at(nxt)
                and not short_len(nxt + 1)
                and not companion_at(nxt + 1)
            ):
                out.append(tok)
                out.append((nxt, "Y", 0x90, 0x03, 0))
                i = nxt + 2
                continue
        out.append(tok)
        i = nxt
    return out


def tail_vector(mcode):
    """Offset of the FlatBuffers table vector that opens an mcode blob's
    tail, found through the header word that points at it (a uoffset whose
    target holds a small count followed by increasing table offsets) -- the
    fixed five-entry `TAIL_VECTOR` pattern only holds for graphs with one
    input and one output. Multi-input graphs carry longer headers (the
    pointer sits past the 297-byte fallback bound); for those the bounded
    scan below finds nothing and a wider, tiling-validated scan takes over.
    See the README's "The tail is the segment table" section."""
    u32 = lambda o: struct.unpack_from("<I", mcode, o)[0]  # noqa: E731
    # The usual anchor is the convolution engine's channel-extent write (see
    # the README's "The first operand with a known meaning" section). A graph
    # with no convolution in it -- a lone LeakyRelu, Relu or Sigmoid -- never
    # programs that register, so fall back to the end of the fixed header.
    first_verb = mcode.find(b"\xa1\x00\x40\x02")
    if first_verb <= 0:
        first_verb = 297
    for o in range(0, first_verb - 3, 4):
        t = o + u32(o)
        if not (first_verb < t < len(mcode) - 8):
            continue
        n = u32(t)
        if not (1 <= n <= 64) or t + 4 + 4 * n > len(mcode):
            continue
        offs = [u32(t + 4 + 4 * k) for k in range(n)]
        if all(0 < x < 8192 for x in offs) and offs == sorted(offs):
            return t
    # Fallback for longer headers (multi-input graphs carry per-input
    # descriptors that push the pointer past the 297-byte fixed header a
    # lone Relu-style graph has): scan wide and validate by tiling, which
    # a chance byte pattern cannot satisfy -- the segments must tile the
    # blob exactly from the header to the vector.
    for o in range(0, len(mcode) - 64, 4):
        t = o + u32(o)
        if not (0 < t < len(mcode) - 8):
            continue
        n = u32(t)
        if not (1 <= n <= 64) or t + 4 + 4 * n > len(mcode):
            continue
        offs = [u32(t + 4 + 4 * k) for k in range(n)]
        if not (all(0 < x < 8192 for x in offs) and offs == sorted(offs)):
            continue
        if _tiling_ok(mcode, t):
            return t
    raise AssertionError("no header word points at a tail table vector")


def _tables_at(mcode, vec):
    """`tail_tables` internals for an already-located vector (so the wide
    fallback above can validate candidates without recursing)."""
    u32 = lambda o: struct.unpack_from("<I", mcode, o)[0]  # noqa: E731
    i32 = lambda o: struct.unpack_from("<i", mcode, o)[0]  # noqa: E731
    u16 = lambda o: struct.unpack_from("<H", mcode, o)[0]  # noqa: E731
    tables = []
    for k in range(u32(vec)):
        p = vec + 4 + 4 * k
        tpos = p + u32(p)
        vt = tpos - i32(tpos)
        vsz, tsz = u16(vt), u16(vt + 2)
        fields = {}
        for f in range((vsz - 4) // 2):
            off = u16(vt + 4 + 2 * f)
            if off:
                fields[f] = u32(tpos + off) if off + 4 <= tsz else u16(tpos + off)
        tables.append(fields)
    return tables


def _tiling_ok(mcode, vec):
    """Whether `vec` is a genuine tail vector: its tables parse and their
    word counts tile the blob exactly. False (never raising) otherwise."""
    try:
        tables = _tables_at(mcode, vec)
        words = [t.get(2, 0) for t in tables]
        header = vec - 8 * sum(words)
        return 0 <= header < vec and header + 8 * sum(words) == vec
    except Exception:  # noqa: BLE001 -- any parse failure means "not it"
        return False


def tail_tables(mcode):
    """Walk the FlatBuffers tables of an mcode blob's tail (five for a
    one-input, one-output CNN; fifteen for an `llm_build` subgraph).
    Returns, per table, a dict of `field index -> uint32 (or uint16) value`
    for present fields, read through each table's vtable."""
    vec = tail_vector(mcode)
    return vec, _tables_at(mcode, vec)


def segments(mcode):
    """The stream segments the tail table describes: `(offset, length,
    table)` per segment in stream order, which is *reverse* table order.
    Table field 2 is the segment length in 8-byte words; the segments tile
    the blob exactly from the end of the FlatBuffers header to the tail
    vector. Also returns the header length."""
    vec, tables = tail_tables(mcode)
    words = [t.get(2, 0) for t in tables]
    header = vec - 8 * sum(words)
    segs, pos = [], header
    for k in range(len(tables) - 1, -1, -1):
        segs.append((pos, 8 * words[k], tables[k]))
        pos += 8 * words[k]
    assert pos == vec
    return header, segs


def stream_bounds(mcode):
    """`(first instruction byte, one past the last)` -- the FlatBuffers header
    and tail are not instructions."""
    header, segs = segments(mcode)
    return header, segs[-1][0] + segs[-1][1]


def decode(mcode, start=None, end=None, **rule):
    """Decode a stream into records that carry *everything* needed to write it
    back out -- the counterpart of `tokenize`, which returns only what
    is needed to identify a form. Each record is a dict with a `kind`:

    * `V`: a verb (`verb`, `field`, `bank`, `operand`; 8 bytes, or 7 when the
      next form starts early and the operand is one byte shorter)
    * `W`: a companion write (`x`, `field`, `bank`, `operand`)
    * `S`: a width-rule unit (`p`, `payload`, `tag`, `reg`, `extra`)
    * `B`: a bare `[tag][register]` pair (`tag`, `reg`, `extra`)
    * `Q`: a quintet `[04][a][b][a][b]` (`a`, `b`)
    * `P`: a `0b 91` prefix under its anchored verb (`x`, `y`)
    * `D`: a `05 90` doublet, raw on both bytes (`x`, `y`)
    * `N`: a six-byte `09 0c 80 fe 01 01` prefix under its anchored verb
      (no payload)
    * `F`: a fixed `01 a4 00 c1 W` unit (`w`: the one live byte)
    * `X`: a `05 10 e2 0e` prefix under its anchored verb (no payload)
    * `E`: an octet `04 40 84 18 83 R TT 40`, taken only by adjudication
      (`R`, `TT`)
    * `T`: an 8-byte `01 98 02 83 R 83 0e 05` template, taken only by
      adjudication (`R`: the one live byte)
    * `C`: a `[H][A][B][0xc1][D]` unit with `H` in `{0x01, 0x30}` and `D`
      even (`h`, `a`, `b`, `d`)
    * `R`: a `[0x30][0x03][X][0x03][0x09]` repeat unit (`x`: the live slot)
    * `Y`: a stuttered `90 03` echo after a short unit ending in it
      (`a`, `b`)
    * `L`: a terminal `0b 01` pair closing an S-unit rhythm before zero
      padding (no payload)
    * `A`: a lone trailer single abutting the next segment's `a7`
      marker (`v`: the value byte)
    * `raw`: one byte no form accounts for (`byte`)

    Every record also carries `at`, its offset in the original stream, so
    an edit can be scoped to one segment.

    See the README's "A lossless codec" section.
    """
    lo = 297 if start is None else start
    hi = (len(mcode) - 252) if end is None else end
    out = []
    toks = tokenize(mcode, start=lo, end=hi, **rule)
    for i, t in enumerate(toks):
        o, kind = t[0], t[1]
        nxt = toks[i + 1][0] if i + 1 < len(toks) else hi
        n = nxt - o
        if kind == "V":
            out.append(
                {
                    "at": o,
                    "kind": "V",
                    "verb": t[2],
                    "field": t[3],
                    "bank": t[4],
                    "operand": mcode[o + 4 : o + n],
                }
            )
        elif kind == "W":
            out.append(
                {
                    "at": o,
                    "kind": "W",
                    "x": t[2],
                    "field": t[3],
                    "bank": t[4],
                    "operand": mcode[o + 3 : o + 7],
                }
            )
        elif kind == "S":
            p = t[2]
            # Read the tag from the stream rather than the token: for a unit
            # whose tag takes an extra byte, `tokenize`'s third slot
            # holds the register, not the tag.
            out.append(
                {
                    "at": o,
                    "kind": "S",
                    "p": p,
                    "payload": mcode[o + 1 : o + 1 + p + 1],
                    "tag": mcode[o + p + 2],
                    "reg": mcode[o + p + 3],
                    "extra": mcode[o + p + 4 : o + n],
                }
            )
        elif kind == "B":
            out.append(
                {
                    "at": o,
                    "kind": "B",
                    "tag": t[2],
                    "reg": t[3],
                    "extra": mcode[o + 2 : o + n],
                }
            )
        elif kind == "Q":
            out.append({"at": o, "kind": "Q", "a": t[2], "b": t[3]})
        elif kind in ("P", "D"):
            out.append({"at": o, "kind": kind, "a": t[2], "b": t[3]})
        elif kind == "N":
            out.append({"at": o, "kind": "N"})
        elif kind == "F":
            out.append({"at": o, "kind": "F", "w": t[2]})
        elif kind == "X":
            out.append({"at": o, "kind": "X"})
        elif kind == "E":
            out.append({"at": o, "kind": "E", "a": t[2], "b": t[3]})
        elif kind == "T":
            out.append({"at": o, "kind": "T", "r": t[2]})
        elif kind == "C":
            out.append(
                {"at": o, "kind": "C", "h": t[2], "a": t[3], "b": t[4][0], "d": t[4][1]}
            )
        elif kind == "R":
            out.append({"at": o, "kind": "R", "x": t[2]})
        elif kind == "Y":
            out.append({"at": o, "kind": "Y", "a": t[2], "b": t[3]})
        elif kind == "L":
            out.append({"at": o, "kind": "L"})
        elif kind == "A":
            out.append({"at": o, "kind": "A", "v": t[2]})
        else:
            out.append({"at": o, "kind": "raw", "byte": t[2]})
    return out


def encode(records):
    """Write decoded records back out as bytes -- the inverse of
    `decode`. Nothing here reads the original stream, so a byte-exact
    round trip proves the decode captures every bit the forms carry."""
    out = bytearray()
    for r in records:
        kind = r["kind"]
        if kind == "V":
            out += bytes([r["verb"], 0, r["field"], r["bank"]]) + r["operand"]
        elif kind == "W":
            out += bytes([r["x"], r["field"], r["bank"]]) + r["operand"]
        elif kind == "S":
            out += (
                bytes([r["p"]])
                + r["payload"]
                + bytes([r["tag"], r["reg"]])
                + r["extra"]
            )
        elif kind == "B":
            out += bytes([r["tag"], r["reg"]]) + r["extra"]
        elif kind == "Q":
            out += bytes([0x04, r["a"], r["b"], r["a"], r["b"]])
        elif kind in ("P", "D"):
            out += bytes([r["a"], r["b"]])
        elif kind == "N":
            out += bytes([0x09, 0x0C, 0x80, 0xFE, 0x01, 0x01])
        elif kind == "F":
            out += bytes([0x01, 0xA4, 0x00, 0xC1, r["w"]])
        elif kind == "X":
            out += bytes([0x05, 0x10, 0xE2, 0x0E])
        elif kind == "E":
            out += bytes([0x04, 0x40, 0x84, 0x18, 0x83, r["a"], r["b"], 0x40])
        elif kind == "T":
            out += bytes([0x01, 0x98, 0x02, 0x83, r["r"], 0x83, 0x0E, 0x05])
        elif kind == "C":
            out += bytes([r["h"], r["a"], r["b"], 0xC1, r["d"]])
        elif kind == "R":
            out += bytes([0x30, 0x03, r["x"], 0x03, 0x09])
        elif kind == "Y":
            out += bytes([r["a"], r["b"]])
        elif kind == "L":
            out += bytes([0x0B, 0x01])
        elif kind == "A":
            out += bytes([r["v"]])
        else:
            out += bytes([r["byte"]])
    return bytes(out)


def structured_share(records):
    """Fraction of the encoded bytes that come from a recognised form rather
    than a raw escape -- how much of a stream we could write from structure."""
    total = len(encode(records))
    raw = sum(1 for r in records if r["kind"] == "raw")
    return (total - raw) / total


def nonzero_coverage(mcode, **rule_overrides):
    """Fraction of the stream's *non-zero* bytes the rule accounts for, plus
    the unexplained non-zero runs as `(start, end)` pairs. Zero bytes are
    segment padding and are not counted either way."""
    rule = dict(FULL_RULE)
    rule.update(rule_overrides)
    lo, hi = stream_bounds(mcode)
    toks = tokenize(mcode, start=lo, end=hi, **rule)
    nonzero = sum(1 for i in range(lo, hi) if mcode[i])
    runs, last = [], None
    for o, kind, *_ in toks:
        if kind == "?" and mcode[o]:
            if last == o:
                runs[-1][1] = o + 1
            else:
                runs.append([o, o + 1])
            last = o + 1
    unexplained = sum(b - a for a, b in runs)
    return (nonzero - unexplained) / nonzero, [tuple(r) for r in runs]


def segment_coverage(mcode, seg):
    """Explained fraction of one stream segment under `FULL_RULE`, with its
    trailing zero padding trimmed; also the count of `a1 40 02` verbs (op
    programs) and of `a7` verbs inside it."""
    pos, length, _ = seg
    end = pos + length
    while end > pos and mcode[end - 1] == 0:
        end -= 1
    toks = tokenize(mcode, start=pos, end=end, **FULL_RULE)
    # The segment's first 8 bytes can hold the tail of the `a7` marker verb
    # that starts 4 bytes *before* the word boundary (`1e 00 00 00 00`);
    # those are the marker's operand, not unexplained content. Zero bytes
    # left over are padding (an `llm_build` op segment ends with 8 zero bytes
    # before the next segment's marker), not content either.
    unknown = sum(1 for t in toks if t[1] == "?" and t[0] >= pos + 8 and mcode[t[0]])
    programs = sum(1 for t in toks if t[1:] == ("V", 0xA1, 0x40, 0x02))
    a7 = sum(1 for t in toks if t[1] == "V" and t[2] == 0xA7)
    return 1 - unknown / max(1, end - pos), programs, a7


def mcodes_of(axmodel_path):
    """Every `neu mode` node's mcode in an `.axmodel`, as `(node name,
    bytes)` -- `llm_build` per-layer files carry two (decode and prefill)."""
    m = onnx.load(axmodel_path)
    out = []
    for node in m.graph.node:
        if node.op_type != "neu mode":
            continue
        info = json.loads(
            next(a for a in node.attribute if a.name == "npu_graph_info").s.decode()
        )
        for d in info["dotneus"]:
            init = next(i for i in m.graph.initializer if i.name == d["neu_key"])
            out.append((node.name, bytes(init.raw_data)))
    return out


TAGS = ALL_TAGS | {0xA1} | {t | 0x40 for t in ALL_TAGS | {0xA1}}
"""Every byte that can open a short unit: the tags below the verb range, `a1`
(a tag as well as a verb), and the odd-register form of each -- bit 6 of a tag
selects the odd register. See the README's "0xa1 is also a tag" section."""


def check(mcode, strict=True):
    """Every structural invariant this project confirmed on an AX650N, as a
    list of human-readable violations. An empty list means the blob is
    well-formed as far as the format is understood.

    This is *not* an evaluator -- see the module docstring. It answers "could
    the runtime load and walk this?", which is the question that catches a
    corrupted or hand-edited stream, and it needs neither Docker nor a card.
    """
    bad = []

    try:
        header, segs = segments(mcode)
    except Exception as exc:  # noqa: BLE001 -- report, do not raise
        return [f"tail: no readable segment table ({exc})"]

    # 1. The segment table is a loader manifest. Its word counts are
    #    load-bearing: the runtime rejects a blob whose segments do not tile
    #    the stream exactly, from the end of the header to the tail vector.
    vec = tail_vector(mcode)
    end = segs[-1][0] + segs[-1][1]
    if end != vec:
        bad.append(f"segments: tile to {end}, tail vector is at {vec}")
    for pos, length, _ in segs:
        if length % 8:
            bad.append(f"segment at {pos}: length {length} is not a whole word")
        if pos < header or pos + length > len(mcode):
            bad.append(f"segment at {pos}: length {length} leaves the blob")

    if not header < end <= len(mcode):
        # Nothing below can be read if the manifest does not describe this
        # blob; report what is already known and stop.
        return bad + [f"segments: span {header}..{end} is not inside the blob"]

    try:
        records = decode(mcode, start=header, end=end, **FULL_RULE)
    except Exception as exc:  # noqa: BLE001 -- report, do not raise
        return bad + [f"codec: the stream does not decode ({exc!r})"]

    # 2. The codec round-trips. A stream that does not is one this decoder
    #    misread, so nothing below it can be trusted.
    if encode(records) != mcode[header:end]:
        bad.append("codec: re-encoding the decoded records is not byte-exact")

    # 3. `a7` is the synchronisation verb, and every segment boundary but the
    #    first carries one within four bytes -- the verb can start just before
    #    the word boundary its segment begins on. Measured on 60 real streams.
    starts = {r["at"] for r in records if r["kind"] == "V" and r["verb"] == 0xA7}
    for pos, _, _ in segs[1:]:
        if not any(pos + d in starts for d in range(-4, 5)):
            bad.append(f"segment at {pos}: no a7 within four bytes of the boundary")

    # 4. Only the six known verbs, and only tags below the verb range -- plus
    #    `a1`, which is a tag as well as a verb, and the odd-register forms
    #    (bit 6 set) of both.
    for r in records:
        if r["kind"] == "V" and r["verb"] not in VERBS6:
            bad.append(f"at {r['at']}: unknown verb {r['verb']:#04x}")
        if r["kind"] in ("S", "B") and r["tag"] not in TAGS:
            bad.append(f"at {r['at']}: unknown tag {r['tag']:#04x}")

    # 5. Almost every non-zero byte belongs to a recognised form. Zero bytes
    #    are segment padding and are not counted either way. The floor is the
    #    worst of the 60 streams this was measured on (0.9409).
    covered, runs = nonzero_coverage(mcode)
    if strict and covered < 0.94:
        bad.append(
            f"coverage: only {covered:.1%} of non-zero bytes explained {runs[:4]}"
        )

    # 6. The unexplained bytes are scattered, never bulk. Over 70 real streams
    #    -- CNN and `llm_build`, up to 1.8 MB -- no unexplained non-zero run
    #    exceeds nine bytes. A long run is a stream this decoder lost sync in,
    #    which a percentage alone will not show on a large blob.
    long_runs = [(a, b) for a, b in runs if b - a > 12]
    if strict and long_runs:
        bad.append(f"stream: unexplained runs longer than eight bytes {long_runs[:4]}")

    return bad


def op_programs(mcode):
    """The offsets of the op-program verbs (`a1 00 40 02`, the convolution
    engine's channel-extent write) in each segment, as `{segment offset:
    [verb offsets]}`. An `a7` brackets every one of them."""
    header, segs = segments(mcode)
    out = {}
    for pos, length, _ in segs:
        out[pos] = [
            i
            for i in range(pos, pos + max(length - 3, 0))
            if mcode[i] == 0xA1
            and mcode[i + 1] == 0
            and mcode[i + 2] == 0x40
            and mcode[i + 3] == 0x02
        ]
    return out


def weight_table_of(axmodel_path):
    """The `npu_params` weight table of a compiled model, as bytes."""
    model = onnx.load(axmodel_path)
    return bytes(
        next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
    )
