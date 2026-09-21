"""Field-fitting predictor for the untiled AX650 ``Transpose`` MCode.

For one form class of ``Transpose(x[1,1,R,C], perm 0,1,3,2)`` (untiled, float32)
every byte of the compiled MCode that changes with the shape is a low or high
byte of a simple quantity of ``(R, C)``: the tensor byte size, the row size
padded to 8 elements, ``ceil(C/8)`` multiples, and so on. This module *fits*
that relation from a training set of Pulsar2 builds and *predicts* the MCode of
another shape from one template build, so it can be scored byte for byte
against real builds. It does not know where a form class ends: a prediction is
only meaningful for shapes that a held-out comparison has shown to be in the
same class as the template (see ``docs/axera-transpose-untiled.md``).

The compiler-noise window (bytes 301..325, which flips between rebuilds of the
same shape) is copied from the template and ignored when scoring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

NOISE_START = 301
NOISE_END = 326


def _ceil8(c: int) -> int:
    return (c + 7) // 8


# name -> value of the quantity for a (R, C) shape. Extended only when a fit
# needs a new quantity; every one of these was seen as a real field.
FEATURES: dict[str, Callable[[int, int], int]] = {
    "4RC-1": lambda r, c: 4 * r * c - 1,
    "4RC": lambda r, c: 4 * r * c,
    "4C-1": lambda r, c: 4 * c - 1,
    "4C": lambda r, c: 4 * c,
    "32q": lambda r, c: 32 * _ceil8(c),
    "4q": lambda r, c: 4 * _ceil8(c),
    "2q": lambda r, c: 2 * _ceil8(c),
    "2q-2": lambda r, c: 2 * _ceil8(c) - 2,
    "q-1": lambda r, c: _ceil8(c) - 1,
    "q": lambda r, c: _ceil8(c),
    "Rq-1": lambda r, c: r * _ceil8(c) - 1,
    "Rq": lambda r, c: r * _ceil8(c),
    "R-1": lambda r, c: r - 1,
    "R": lambda r, c: r,
    "C-1": lambda r, c: c - 1,
    "C": lambda r, c: c,
    "4R-1": lambda r, c: 4 * r - 1,
    "4R": lambda r, c: 4 * r,
}

Shape = tuple[int, int]
# position -> ("const", byte) or (feature name, byte index 0 = low)
Field = tuple[str, int]


def _byte(value: int, index: int) -> int:
    return (value >> (8 * index)) & 0xFF


def fit(
    builds: dict[Shape, bytes], carries: bool = True, tie_break: bool = False
) -> dict[int, Field]:
    """Explain every position that varies across ``builds`` (equal-length blobs).

    Returns ``{position: (feature, byte_index)}``. Raises ``ValueError`` if a
    varying position matches no feature, or if two features match ambiguously
    (the training set should then be widened so it separates them).

    With ``tie_break`` an ambiguous position takes the first matching feature in
    ``FEATURES`` order instead of raising. Ambiguity means the features agree on
    every training shape (for example the high byte of ``4RC-1`` and ``4RC`` for
    unaligned ``C``); only held-out scoring can show they also agree elsewhere.

    With ``carries`` a fitted low-byte field also claims the next position as its
    high byte when every training shape's byte there equals the feature's high
    byte. That covers a value that only crosses 255 outside the training shapes;
    it is a guess that held-out scoring has to confirm.
    """
    shapes = sorted(builds)
    length = len(builds[shapes[0]])
    if any(len(blob) != length for blob in builds.values()):
        raise ValueError("training builds differ in length; not one form class")
    fields: dict[int, Field] = {}
    for pos in range(length):
        if NOISE_START <= pos < NOISE_END:
            continue
        values = [builds[s][pos] for s in shapes]
        if len(set(values)) == 1:
            continue
        matches = [
            (name, idx)
            for name, fn in FEATURES.items()
            for idx in range(3)
            if all(_byte(fn(*s), idx) == builds[s][pos] for s in shapes)
        ]
        if not matches:
            raise ValueError(f"position {pos} matches no feature: {values}")
        if len(matches) > 1 and not tie_break:
            raise ValueError(f"position {pos} is ambiguous: {matches}")
        fields[pos] = matches[0]
    if carries:
        for pos, (name, idx) in list(fields.items()):
            nxt = pos + 1
            if idx != 0 or nxt in fields or NOISE_START <= nxt < NOISE_END:
                continue
            fn = FEATURES[name]
            if all(_byte(fn(*s), 1) == builds[s][nxt] for s in shapes):
                fields[nxt] = (name, 1)
    return fields


def predict(template: bytes, fields: dict[int, Field], shape: Shape) -> bytes:
    """Patch ``template`` with the fitted fields evaluated at ``shape``."""
    out = bytearray(template)
    for pos, (name, idx) in fields.items():
        out[pos] = _byte(FEATURES[name](*shape), idx)
    return bytes(out)


def mismatches(predicted: bytes, actual: bytes) -> list[int]:
    """Positions where a prediction differs from a real build (noise excluded)."""
    if len(predicted) != len(actual):
        return [-1]
    return [
        i
        for i in range(len(actual))
        if predicted[i] != actual[i] and not NOISE_START <= i < NOISE_END
    ]


def score(
    template: bytes,
    fields: dict[int, Field],
    held_out: dict[Shape, bytes],
) -> tuple[int, list[tuple[Shape, Sequence[int]]]]:
    """Return ``(byte-exact count, [(shape, mismatching positions), ...])``."""
    exact = 0
    bad: list[tuple[Shape, Sequence[int]]] = []
    for shape, actual in sorted(held_out.items()):
        diff = mismatches(predict(template, fields, shape), actual)
        if diff:
            bad.append((shape, diff))
        else:
            exact += 1
    return exact, bad
