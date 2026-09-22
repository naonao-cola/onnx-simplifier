"""Applying `emitter.py`'s bit-permutation weight learning at real, wide-channel
ResNet18 Conv shapes -- the first time it has been run beyond toy 1-8 channel
8x8 fixtures (see `docs/axera-conv-weight-learn-wide.md` for the full account,
including a real device measurement of what is and is not closed).

**Validated, byte-exact**: the weight-code bit permutation itself. `learn()`
converges to zero collisions at k=47 reference builds for both
`Conv(x[16,256,14,14], w[256,256,3,3])` and `Conv(x[16,512,7,7],
w[512,512,3,3])`, exactly as it does at the toy 32-channel shape this
project's README first measured it at -- `k` needed did not grow with the
~192x/~768x larger code-bit count. A held-out weight set's mapped table bytes
matched a fresh native Pulsar2 build 100% outside the requantisation region.

**Not validated, and load-bearing**: the requantisation/scaffold region. At
this shape it is not one contiguous run the way the toy shape's is (a real
consequence of the "wide convolutions are split into slices" layout this
project's README documents) -- it is 512 scattered runs. Applying `learn()` a
second time, treating `requant_block()`'s computed bytes as the "code",
recovers about 51% of that region as a permutation of the same formula; the
rest is unexplained. And unlike the toy shape, `learn_mcode()`'s automatic
scale/zero-offset search fails outright here: only about half the reference
builds share one byte layout for those fields (a real, undecoded compiler
form-split at this shape), so the found offsets (`5527/5535/5543/5551` for a
4x-repeated `y_scale` literal, `5514` for the `y_zero` byt, both **specific
to the "form-A" reference build committed here**, not shape-general) only
apply within that one sub-population.

A held-out weight set emitted with the found scale/zero patched but the
requant residual left at the reference's own values measured **mean abs error
0.63 against a peak magnitude of ~6.2 on the AX8850** (device-to-device noise
alone is 0.015) -- visibly, substantially wrong. This module is therefore a
**characterization aid, not a working emitter**: `emit_weight_table` exists so
the byte-exact and the not-yet-explained portions can be inspected and
extended, not so it can be dropped into a pipeline expecting correct output.
"""

from __future__ import annotations

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fixtures", "conv_learn_wide")

# Found on the one committed "form-A" reference build (`conv256_reference.axmodel.gz`);
# see the module docstring -- only ~half of a 46-build sample shared this layout.
MCODE_SCALE_OFFSETS_FORM_A = (5527, 5535, 5543, 5551)
MCODE_ZERO_OFFSET_FORM_A = 5514


def load_maps(path=None):
    """The learned bit-permutation maps for `Conv(x[16,256,14,14], w[256,256,3,3])`.

    Returns a dict with `origin`/`const` (the weight-code bit permutation, `emitter.CONST`
    for unmapped bits), `ambiguous_bytes` (the table byte offsets `origin` could not
    explain -- the scattered requantisation/scaffold region), and `origin2`/`const2`
    (a *second* permutation, mapping `requant_block()`'s bytes into `ambiguous_bytes`;
    still leaves `ambiguous2` genuinely unexplained).
    """
    path = path or os.path.join(_FIXTURES, "conv256_maps.npz")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def emit_weight_table(reference_table, w, x_scale, x_zero, y_scale, y_zero, maps):
    """Best-effort weight table for new weights `w`, from a reference table.

    `x_scale`/`x_zero`/`y_scale`/`y_zero` are the new weights' own quantisation
    (from a real compile, or a chosen target range) -- `requant_block` needs
    them and this module has no way to guess them.

    Writes the weight-code bits through `maps["origin"]` (byte-exact, confirmed
    against a held-out build -- see the module docstring) and the
    `requant_block()`-explained subset of the scattered scaffold region through
    `maps["origin2"]`. Bytes in neither map keep the reference's own value,
    which is why this is **not** a correct emission on its own: about 49% of
    the scaffold region is still the reference's, not the new weights', and a
    held-out device check measured substantial resulting error (see the module
    docstring). Returned so the byte-exact and not-yet-explained portions can
    be inspected and extended.
    """
    import emitter

    origin = maps["origin"].astype(np.int64)
    codes = emitter.codes_of(w)
    table = emitter.emit_table(reference_table, origin, codes)

    ambiguous_bytes = maps["ambiguous_bytes"]
    req = emitter.requant_block(
        codes, x_scale, x_zero, y_scale, y_zero, emitter.weight_scales(w)
    )
    origin2 = maps["origin2"].astype(np.int64)
    amb_ref = table[ambiguous_bytes].copy()
    amb_new = emitter.emit_table(amb_ref, origin2, req)
    table = table.copy()
    table[ambiguous_bytes] = amb_new
    return table


def patch_mcode_scale_zero_form_a(mcode, y_scale, y_zero):
    """Patch a "form-A" reference mcode's output scale/zero literals in place.

    Only valid for a reference build sharing `conv256_reference.axmodel.gz`'s
    byte layout at `MCODE_SCALE_OFFSETS_FORM_A`/`MCODE_ZERO_OFFSET_FORM_A` --
    about half of a 46-build sample did not (see the module docstring). Callers
    must confirm their own reference matches before trusting this.
    """
    import numpy as _np

    out = bytearray(_np.asarray(mcode, dtype=_np.uint8).tobytes())
    scale_bytes = _np.float32(y_scale).tobytes()
    for off in MCODE_SCALE_OFFSETS_FORM_A:
        out[off : off + 4] = scale_bytes
    out[MCODE_ZERO_OFFSET_FORM_A] = round(float(y_zero)) & 0xFF
    return _np.frombuffer(bytes(out), dtype=_np.uint8)
