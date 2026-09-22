"""Characterization of Add's `cv3` compute-engine segment: content is
calibration-invariant; size is not a function of shape this project's
existing formulas explain. No predictor -- this is a decode, not an
emitter. See ``docs/axera-add-cv3-decode.md``.

`cv3` is `mcode.segments(mc)[1][3]` -- the fourth of the five per-engine
segments a compiled model's MCode splits into (`docs/axera-step-attribution.md`
names the five tracks `conv0`, `conv1`, `teng2`, `cv3`, `sdma4`; a Relu/Add
model with no `Conv` still has all five, with `conv0`/`conv1` collapsed to
small fixed headers).

Two things are now established that were not before this module:

* **`cv3`'s content does not depend on calibration at all.** Three builds of
  the same `Add([1,16,16,16], [1,16,16,16])` shape with different calibration
  ranges -- symmetric `x,z` in `[-0.9,0.9]`, asymmetric `x` in `[-0.9,0.9]`/
  `z` in `[-0.1,0.1]` (the range PR #1756's `npu_params` header formula was
  derived from), and a second, differently-asymmetric `z` in `[-0.3,0.05]` --
  produced byte-identical `cv3` segments, 0/256 bytes differing across all
  three. `cv3` is not where the Q15 requantization literals PR #1756 found in
  `npu_params`'s header live, and it does not encode any other
  calibration-derived value either, at least not at this shape.
* **`cv3`'s length is not explained by `dma_tile_predict.py`'s `entries`/
  `chunk` model**, the same tile-count rule that fully predicts `npu_params`.
  It is not a function of `N*C` alone, of `H*W` alone, of their product
  (total tensor bytes), or of the predicted tile `entries` count alone --
  each of those four candidate single variables was directly falsified by a
  same-value counter-example in the measured table below (e.g. two shapes
  with `N*C=16` differ, 256 vs. 32 bytes, depending on `H*W`; two shapes with
  `entries=4` differ, 256 vs. 288, depending on shape). Within the untiled
  `npu_params` regime (`entries=1`, every shape below `dma_tile_predict.py`'s
  262,144-byte threshold) there is a SECOND, finer size transition
  `dma_tile_predict.py` has no concept of: shapes with total tensor bytes
  <=16,384 measure `cv3_len=256`; shapes with total tensor bytes >=65,536
  measure `cv3_len=32`. The exact boundary is bracketed to
  `(16384, 65536)` and not narrowed further.

`MEASURED` below is every `(shape, cv3_len)` pair checked, from the fixtures
in `fixtures/add_cv3_decode/` plus the pre-existing ones in
`fixtures/add_tiles/` (PR #1756). No formula reproduces this column from the
shape columns; it is recorded as data, not fit to anything.
"""

from __future__ import annotations

import gzip
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402
import onnx  # noqa: E402

_TILE_FIXTURES = os.path.join(_HERE, "fixtures", "add_tiles")

# (shape, cv3_len, entries-per-dma_tile_predict-or-None-if-not-applicable, source fixture)
# entries is recorded for reference only; see the module docstring for why it
# does not explain cv3_len.
MEASURED = (
    ((1, 16, 8, 8), 256, 1, "add_tiles/add_1x16x8x8.axmodel.gz"),
    ((1, 16, 16, 16), 256, 1, "add_tiles/add_asym_1x16x16x16.axmodel.gz"),
    ((1, 16, 16, 16), 256, 1, "add_cv3_decode/sym_1x16x16x16.axmodel.gz"),
    ((1, 16, 16, 16), 256, 1, "add_cv3_decode/asym2_1x16x16x16.axmodel.gz"),
    ((1, 16, 32, 32), 32, 1, "add_cv3_decode/nc16_row1024b.axmodel.gz"),
    ((1, 16, 38, 38), 32, 1, "add_cv3_decode/nc16_row1444.axmodel.gz"),
    ((1, 16, 32, 64), 32, 1, "add_cv3_decode/nc16_row2048.axmodel.gz"),
    ((1, 16, 56, 56), 32, 1, "add_cv3_decode/row3136_c16.axmodel.gz"),
    ((1, 16, 64, 64), 32, 1, "add_cv3_decode/nc16_row4096.axmodel.gz"),
    ((1, 32, 8, 8), 256, 1, "add_cv3_decode/nc32_row64.axmodel.gz"),
    ((1, 32, 32, 32), 32, 1, "add_tiles/add_1x32x32x32.axmodel.gz"),
    ((1, 32, 56, 56), 256, 4, "add_cv3_decode/row3136_c32.axmodel.gz"),
    ((1, 64, 56, 56), 256, 4, "add_tiles/add_1x64x56x56.axmodel.gz"),
    ((1, 128, 56, 56), 256, 4, "add_cv3_decode/row3136_c128.axmodel.gz"),
    ((4, 64, 56, 56), 288, 8, "add_tiles/add_4x64x56x56.axmodel.gz"),
    ((8, 64, 28, 28), 288, 4, "add_tiles/add_holdout_8x64x28x28.axmodel.gz"),
    ((16, 64, 56, 56), 512, 32, "add_tiles/add_16x64x56x56.axmodel.gz"),
    ((16, 128, 28, 28), 256, 16, "add_tiles/add_16x128x28x28.axmodel.gz"),
)


_FIXTURES_ROOT = os.path.dirname(_TILE_FIXTURES)


def _resolve(rel_path: str) -> str:
    """`rel_path` is `"<subdir>/<file>"` relative to `fixtures/`."""
    return os.path.join(_FIXTURES_ROOT, rel_path)


def load_cv3(rel_fixture_path: str) -> bytes:
    """The raw `cv3` segment bytes of a gzipped fixture under `fixtures/`."""
    with gzip.open(_resolve(rel_fixture_path), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    mc = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )
    _, segs = mcode.segments(mc)
    off, length, _ = segs[3]
    return mc[off : off + length]
