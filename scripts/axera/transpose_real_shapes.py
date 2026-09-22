"""A verified template library for the real ResNet18 training step's
``Transpose`` ops -- a coverage report, not another shape-to-MCode decode.

Four independent attempts (``docs/axera-transpose-mcode.md``,
``docs/axera-transpose-tiled.md``, ``docs/axera-teng2-sqrt-blocks.md``,
``docs/axera-teng2-tiled-repeat.md``) found that a compiled ``Transpose``'s
MCode cannot be predicted from its shape by formula. That is the wrong
target for this op: unlike ``Slice``/``Gather``, a ``Transpose`` has no
data-dependent parameter to retarget -- its ``npu_params`` is either all
zero or a pure function of shape (already decoded in
``transpose_tiled_params.py``), and its MCode is fixed once the shape and
perm are fixed. So a Pulsar2-compiled reference for one exact
``(shape, perm)`` pair *is* the complete, reusable artifact for every graph
instance that uses that exact shape -- no math needed, just a verified
lookup.

The real ResNet18 training step
(``/home/takecheeze/npu-scratch/t6-r18fold/step.onnx``, 1104 nodes) has 41
``Transpose`` nodes across 23 distinct ``(input_shape, perm)`` pairs. All 23
are compiled here (Pulsar2 7.0-lite, AX650, MinMax calibration, uniform
+/-0.9 samples) and verified: every one ran on a real AX8850 (``axcl-vm``)
against ``numpy.transpose`` with **exact** results (``max_err = 0.0`` for
all 23, including the two largest at 28.9M and 29.5M elements) -- Transpose
moves FP32 bytes without quantizing them, so unlike Relu/Sqrt/Gather there
is no rounding to allow for. That covers all 41 of the step's Transpose
node-instances, since every instance is one of these 23 shapes.

See ``docs/axera-transpose-real-shapes.md`` for the full report.
"""

from __future__ import annotations

import gzip
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_DIR = os.path.join(_HERE, "fixtures", "transpose_real")
_FIXTURE_RE = re.compile(r"^transpose_real_(\d+(?:x\d+)*)_perm(\d+)\.axmodel\.gz$")


def _discover() -> dict[tuple[tuple[int, ...], tuple[int, ...]], str]:
    found = {}
    for name in sorted(os.listdir(_FIXTURE_DIR)):
        match = _FIXTURE_RE.match(name)
        if not match:
            continue
        shape = tuple(int(d) for d in match.group(1).split("x"))
        perm = tuple(int(d) for d in match.group(2))
        found[(shape, perm)] = name
    return found


TEMPLATES = _discover()
"""``{(shape, perm): fixture_filename}`` for every verified real-shape template."""


def template_path(shape, perm) -> str:
    """The fixture path for a verified ``(shape, perm)`` pair.

    Raises ``ValueError`` if this exact pair has no verified template --
    this module does no retargeting or shape generalization, only an
    indexed lookup of what was actually compiled and checked on device.
    """
    key = (tuple(shape), tuple(perm))
    if key not in TEMPLATES:
        raise ValueError(
            f"no verified Transpose template for shape={shape} perm={perm}; "
            f"known: {sorted(TEMPLATES)}"
        )
    return os.path.join(_FIXTURE_DIR, TEMPLATES[key])


def load_template_bytes(shape, perm) -> bytes:
    """The raw (ungzipped) compiled ``.axmodel`` bytes for a verified pair."""
    with gzip.open(template_path(shape, perm), "rb") as f:
        return f.read()
