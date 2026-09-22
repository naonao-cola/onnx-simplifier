# Refining `teng2_cluster_patch`: no third point needed, but a different wall found

`docs/axera-teng2-sqrt-blocks.md` (PR #1757) built `teng2_cluster_patch.py`,
which extrapolates a third cluster member's MCode from two adjacent, already-
compiled references. It was exact for one held-out triple and left a small
residual (2 and 20 bytes, of ~3000) for two others, attributing this to "a
carry effect invisible to a two-point diff" and suggesting a third reference
as the fix. This tests that suggestion directly and finds something narrower
and more useful.

## No third point was actually needed

`extrapolate`'s field-detection sizes each field to exactly the run of bytes
that *differ* between the two given references. When a byte's value happens
not to cross a 256-boundary between those two references, the field is
recorded as one byte wide -- and the *next* step's real carry into the
following byte is invisible to a model built from only that width.

Reproducing the `C=153,154,155` case exactly (the three fixtures were already
committed by PR #1757 -- no rebuild needed) confirms this precisely: the
`153->154` field at byte offset 1644 is one byte wide (delta `+28`), and the
`154->155` field at the same offset is *two* bytes wide (still delta `+28`,
now visibly carrying into offset 1645). `extrapolate`, built only from the
`153->154` pair, uses the narrower one-byte field and gets both 1644 and the
analogous case at 2237 wrong by exactly the missed carry.

**The fix needs no new build.** Reading a few extra high-order bytes at each
detected field -- clamped so it never reaches the next field's start or the
buffer end -- lets the same unbounded-integer addition `extrapolate` already
does carry correctly, using only the original two references. Widening by up
to 4 bytes:

- Closes the `C=153,154,155` residual to **zero** (was 2 bytes), using only
  `ref153`/`ref154`.
- Regresses nothing: the already-exact `C=157,158,159` cluster stays exact
  with the wider window.

Device-verified (AX8850, `axcl-vm`, serialized under the shared lock, control
run before and health run after, both clean): the widened prediction for
`C=155` matches the native `C=155` build's own output exactly --
`max_err=0.0071` against `numpy.sqrt`, identical across all 486,080 elements,
both before and after the fix. This is the same figure the *control* run
reports, i.e. the prediction is indistinguishable from a real compile.

## A genuinely new build shows the wider window is not the whole story

`docs/axera-teng2-sqrt-blocks.md`'s third example, `C=165,166,167`, was never
built past `ref165`/`ref166` -- it was cited only as "20 residual differing
bytes," estimated from the pattern of the other two. Building the real
`C=167` (Pulsar2 7.0-lite, `Sqrt(x[1,C,56,56])`, matching the original
sweep's calibration exactly) for the first time finds the true residual is
**31 bytes**, not 20, and widening only closes 2 of them.

The other 29 are not a carry. Comparing the `165->166` and `166->167` field
lists at the same offset range:

```
165->166:  2088(w1,+14)  2093(w1,+1)  2099(w2,+28)
166->167:  2087(w29, one huge non-arithmetic "delta")
```

and the raw bytes directly:

```
166[2085:2099] = 8b 16 01 f8 0f 82 4e 00 27 90 46 8a 18 01
167[2085:2099] = 8b 16 00 06 81 26 02 e0 01 28 90 46 8a 18
```

`166`'s `82 4e 00 27 90 46 8a 18 01` and `167`'s corresponding
`26 02 e0 01 28 90 46 8a 18 01` are the same trailing content shifted right by
one byte, with one new byte (`28`) inserted just before it. This is a real
**length-changing re-encoding** between `C=166` and `C=167` -- the same
variable-length-immediate phenomenon `docs/axera-transpose-mcode.md`
documents for Transpose's own segment 2, now confirmed in `teng2`/`Sqrt`.

No per-position delta model, however widened, can represent an insertion: it
would need to detect the shift and re-index every later byte, which none of
this project's tooling attempts. `teng2_cluster_patch_npoint.extrapolate`
still emits *a* prediction for `C=167` (reusing the widened window), and
correctly leaves the 29-byte insertion-shifted region wrong -- documented as
a known residual in the test, not hidden.

## A more precise statement of what "cluster" means

This also refines the prior doc's own characterization. `{165,166,167}` was
grouped as one cluster because the *total* differing-byte count across the
triple was small relative to the ~3000-byte blob. But `165->166` is a clean
single-template step (every field a genuine value delta) while `166->167`
crosses a real form boundary. A small aggregate diff count across a cluster
is necessary but not sufficient evidence that *every* step within it is safe
to extrapolate through -- the boundary can be internal to an already-
identified cluster, not just at its edges.

## What this changes

- `scripts/axera/teng2_cluster_patch_npoint.py` supersedes
  `teng2_cluster_patch.py` for the carry-only failure mode, with no extra
  build cost -- it should be preferred whenever both references are
  available, since it can only do better or equal.
- It does **not** close the harder failure mode (length-changing
  re-encoding), which remains exactly the wall `docs/axera-teng2-*.md` and
  `docs/axera-transpose-mcode.md` already documented elsewhere. A fix for
  that would need real instruction-boundary/shift detection, not a wider
  window or more reference points -- more points cannot help represent an
  insertion any better than two can, since an insertion is not a value at a
  fixed position at all.

## Reproduction

New fixtures under `scripts/axera/fixtures/teng2_cluster_refine/`
(`sqrt_c165/166/167.axmodel.gz`, gzipped compiled `.axmodel`s, built the same
way as `teng2_sqrt_blocks/`'s). `tests/test_axera_teng2_cluster_patch_npoint.py`
checks all three claims above against committed fixtures, no Docker/device
required.
