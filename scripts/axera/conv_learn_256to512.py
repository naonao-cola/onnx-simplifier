"""`emitter.py`'s bit-permutation Conv weight learner, at ResNet18's stage-3-
to-stage-4 downsample pair -- the widest-channel Conv shapes in the training
step (`Cout=512`), and the hardest attempted in this project's Conv-learn
series (`docs/axera-conv-weight-learn-{stem,downsample,wide}.md`).

Real shapes (from `/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`'s
actual `Conv` nodes, batch=16):

* **1x1 downsample**: `x[16,256,14,14]`, `w[512,256,1,1]`, stride 2, pad 0, bias.
* **3x3 strided**: `x[16,256,14,14]`, `w[512,256,3,3]`, stride 2, pad 1, bias.

**Weight-code region: byte-exact for both, at k=65.** `emitter.learn()`
converges to zero collisions for both shapes at k=65 reference builds --
`Cout*Cin*K*K*8` mapped bits exactly (1,048,576 for 1x1, 9,437,184 for 3x3),
matching every other shape this project has checked. This is the widest
channel count (512) validated so far.

**Scaffold region: does NOT close for either shape, in two different ways.**

* **3x3**: a second-pass `learn()` (`requant_block_biased`'s output as the
  "code", the ambiguous table bytes as the "table") resolves 73.4% at k=65 --
  in the same range `docs/axera-conv-weight-learn-128-and-widegap.md` found
  for 128/128 (80.4%) and a fresh 256/256 run (79.9%), both also at k~65,
  consistent with "more data helps, not fully converged yet." **But unlike
  128/128's residual, this one is NOT numerically harmless**: a device check
  found the emitted holdout's error is not a uniform offset (128/128's
  residual was, and corrected to noise-floor with one additive term) but a
  scattered, non-uniform error -- most of the 512 output channels differ by a
  small ~0.03-0.07 (broadly consistent with a small systematic bias), but 17
  channels differ by up to 11.8 against a ~0.07 control baseline. Those 17
  channels show no obvious periodicity in their indices. **This shape's
  emitted output is confirmed wrong on real hardware, not just
  uncharacterized.**
* **1x1**: the second-pass learn does NOT improve with more data the way
  every other shape's has -- 71.0% resolved at k=40, 70.2% at k=65, with
  persistent, non-decreasing collisions (1136 vs 1119) rather than the
  smooth k-vs-collisions convergence every other shape (including this
  file's own 3x3 case) shows. That is a different failure signature: not
  "needs more samples," but a sign the second-pass hypothesis (the scaffold
  is a bit-permutation of `requant_block_biased`'s own byte layout) may not
  literally hold at this shape, at least not with the channel ordering used
  here. **Worse: the resulting emitted holdout was rejected by the AX8850
  runtime outright** -- `axcl_run_model` returned `0x8030070C` (a hardware
  fault, not a wrong-output run). `scripts/axera/mcode.py`'s `check()` found
  no structural violation in the emitted mcode, so the fault is most likely
  in the *values* left in the ~30% of scaffold bytes this pipeline could not
  explain, not in the instruction stream. Per this project's own device-
  safety rule, a control run and a health run (the unmodified reference
  build) were both confirmed clean immediately before and after this fault --
  the device itself was not wedged, only this one emitted model faulted it.

**So neither shape has a working full-pipeline emitter yet.** The weight-code
half of the technique is now validated at every real ResNet18 Conv shape
attempted (11 for 11, across every fork in this series). The scaffold half
remains genuinely unresolved at the widest channel counts, and this file adds
a new datum future work needs: a scaffold gap is not always numerically
harmless (3x3) and can in one case make the emitted artifact actively unsafe
to run (1x1) rather than merely inaccurate.

Usage (reproduction, given a fresh set of same-shape reference builds under
`work_root/<prefix>_<i:03d>/out/compiled.axmodel` plus `<prefix>_<i>/w.npy`
and `/b.npy`, matching `docs/axera-conv-weight-learn-*.md`'s established
build-harness convention)::

    origin, const, ambiguous = learn_weight_codes(work_root, "c3x3", k=65)
    amb_bytes, origin2, const2, ambiguous2 = learn_scaffold(
        work_root, "c3x3", k=65, origin=origin, ambiguous=ambiguous)
"""

from __future__ import annotations

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fixtures", "conv_learn_256to512")


def _load_build(work_root: str, name: str):
    import emitter

    wd = os.path.join(work_root, name)
    w = np.load(wd + "/w.npy")
    table = emitter.table_of(wd + "/out/compiled.axmodel")
    return w, table


def learn_weight_codes(work_root: str, prefix: str, k: int):
    """First-pass `emitter.learn()` over `k` same-shape builds `<prefix>_000..k-1`.

    Returns `(origin, const, ambiguous)` -- see `emitter.learn`'s docstring.
    Zero `emitter.collisions(codes, origin)` is what "byte-exact" means here;
    callers should check it, not assume it.
    """
    import emitter

    ws, tables = [], []
    for i in range(k):
        w, table = _load_build(work_root, f"{prefix}_{i:03d}")
        ws.append(emitter.codes_of(w))
        tables.append(table)
    origin, const, ambiguous = emitter.learn(ws, tables)
    return origin, const, ambiguous, ws, tables


def learn_scaffold(work_root: str, prefix: str, k: int, ambiguous, quant_scales_fn):
    """Second-pass `emitter.learn()`: treat `requant_block_biased`'s bytes as the
    "code" and the still-ambiguous table bytes from the first pass as the "table".

    `quant_scales_fn(work_dir) -> (x_scale, x_zero, y_scale, y_zero)` supplies each
    build's own quantisation (from its `out/quant/quant_axmodel.json`, e.g. via
    `conv_weight_learn.build_scales`). Returns `(amb_bytes, origin2, const2, ambiguous2)`.
    """
    import conv_weight_learn as cwl
    import emitter

    amb_bytes = np.array(sorted({int(b) // 8 for b in ambiguous}), dtype=np.int64)
    reqs, tabs = [], []
    for i in range(k):
        name = f"{prefix}_{i:03d}"
        wd = os.path.join(work_root, name)
        w = np.load(wd + "/w.npy")
        b = np.load(wd + "/b.npy")
        table = emitter.table_of(wd + "/out/compiled.axmodel")
        x_s, x_z, y_s, y_z = quant_scales_fn(wd)
        codes = emitter.codes_of(w)
        w_scale = emitter.weight_scales(w)
        reqs.append(cwl.requant_block_biased(codes, x_s, x_z, y_s, y_z, w_scale, b))
        tabs.append(table[amb_bytes])
    origin2, const2, ambiguous2 = emitter.learn(reqs, tabs)
    return amb_bytes, origin2, const2, ambiguous2


def emit_holdout(
    reference_table, origin, amb_bytes, origin2, holdout_codes, holdout_req
):
    """Apply both learned maps to a reference table for a held-out weight/bias set.

    Weight-code bits (`origin`) are always written correctly (validated byte-exact
    for both real shapes in this module). Scaffold bytes (`amb_bytes`/`origin2`) are
    only ~70-73% resolved here -- the rest keep the reference's own values, which
    is NOT confirmed safe (see the module docstring: wrong on real hardware for
    both shapes tried).
    """
    import emitter

    table = emitter.emit_table(reference_table, origin, holdout_codes).copy()
    amb_ref_sub = table[amb_bytes].copy()
    amb_new_sub = emitter.emit_table(amb_ref_sub, origin2, holdout_req)
    table[amb_bytes] = amb_new_sub
    return table


def load_maps(prefix: str):
    """The committed first- and second-pass maps for `prefix` in `"c1x1", "c3x3"`."""
    z1 = np.load(os.path.join(_FIXTURES, f"{prefix}_maps.npz"))
    z2 = np.load(os.path.join(_FIXTURES, f"{prefix}_scaffold_map.npz"))
    return {
        "origin": z1["origin"],
        "const": z1["const"],
        "ambiguous": z1["ambiguous"],
        "amb_bytes": z2["amb_bytes"],
        "origin2": z2["origin2"],
        "const2": z2["const2"],
        "ambiguous2": z2["ambiguous2"],
    }
