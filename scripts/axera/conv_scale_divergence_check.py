"""Item 1 of `docs/axera-conv-gather-loose-ends.md`: is `patch_scales.py` the
cause of `Conv(x[16,256,14,14], w[512,256,1,1], stride 2)`'s device fault
(`docs/axera-conv-256to512-tiled-fix.md`, PR #1790), or is the divergence
already present before the patch runs?

Answer, from three committed fixtures and no Docker/device: **already
present**. `patch_scales.py` writes 16 real bytes; the reference and a
genuine native rebuild of the held-out weights already differ in 1,548
bytes, unpatched. Patching moves the byte-diff count against the native
rebuild by exactly zero. See the doc's "Item 1" section for the full
reasoning.
"""

from __future__ import annotations

import gzip
import os

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_TILED_FIX_FIXTURES = os.path.join(_HERE, "fixtures", "conv_256to512_tiled_fix")
_LOOSE_ENDS_FIXTURES = os.path.join(_HERE, "fixtures", "conv_gather_loose_ends")


def _mcode_bytes(gz_path: str) -> bytes:
    with gzip.open(gz_path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


def divergence_report() -> dict:
    """`{comparison: differing_byte_count}` for the three pairings in the doc."""
    reference = _mcode_bytes(
        os.path.join(_TILED_FIX_FIXTURES, "c1x1_reference.axmodel.gz")
    )
    patched = _mcode_bytes(
        os.path.join(_TILED_FIX_FIXTURES, "c1x1_emitted_scaled.axmodel.gz")
    )
    native = _mcode_bytes(
        os.path.join(_LOOSE_ENDS_FIXTURES, "c1x1_holdout_native.axmodel.gz")
    )
    if not (len(reference) == len(patched) == len(native)):
        raise ValueError(
            f"mcode lengths differ: reference={len(reference)}, "
            f"patched={len(patched)}, native={len(native)}"
        )

    def diff(a: bytes, b: bytes) -> int:
        return sum(x != y for x, y in zip(a, b))

    return {
        "reference_vs_native": diff(reference, native),
        "patched_vs_native": diff(patched, native),
        "reference_vs_patched": diff(reference, patched),
    }


if __name__ == "__main__":
    for k, v in divergence_report().items():
        print(f"{k}: {v}")
