"""Regression coverage for the ResNet18 256->512 Conv scaffold fix.

Reproduces `emit_holdout_table` for the two real shapes and checks the result
against a committed oracle (a scale-patched, held-out emission already verified
on the AX8850 -- see `docs/axera-conv-256to512-tiled-fix.md`). No Docker/device
needed.
"""

import gzip
import os
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import conv_256to512_tiled_fix as fix  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_256to512_tiled_fix")
_MAPS = os.path.join(_AXERA_DIR, "fixtures", "conv_learn_256to512")

# The committed holdout weight/bias sets' real quantisation scales, read once from
# `out/quant/quant_axmodel.json` during the original build campaign
# (docs/axera-conv-weight-learn-256to512.md, PR #1777).
_REAL_SCALES = {
    "c1x1": dict(
        x_s=0.007058814167976379, x_z=127.0, y_s=0.016209039837121964, y_z=132.0
    ),
    "c3x3": dict(
        x_s=0.007058821152895689, x_z=127.0, y_s=0.04864101856946945, y_z=123.0
    ),
}


def _table_of_gz(path):
    with gzip.open(path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    for init in model.graph.initializer:
        if init.name == "npu_params":
            return np.frombuffer(bytes(init.raw_data), dtype=np.uint8)
    raise KeyError("npu_params")


@pytest.mark.parametrize("prefix", ["c1x1", "c3x3"])
def test_emit_holdout_table_matches_committed_oracle(prefix):
    origin = np.load(os.path.join(_MAPS, f"{prefix}_maps.npz"))["origin"]
    ref_table = _table_of_gz(os.path.join(_FIXTURES, f"{prefix}_reference.axmodel.gz"))
    w = np.load(os.path.join(_FIXTURES, f"{prefix}_holdout_w.npy"))
    b = np.load(os.path.join(_FIXTURES, f"{prefix}_holdout_b.npy"))
    oracle_table = _table_of_gz(
        os.path.join(_FIXTURES, f"{prefix}_emitted_scaled.axmodel.gz")
    )

    s = _REAL_SCALES[prefix]
    table = fix.emit_holdout_table(
        prefix, ref_table, origin, w, b, s["x_s"], s["x_z"], s["y_s"], s["y_z"]
    )

    assert table.shape == oracle_table.shape
    diff = np.flatnonzero(table != oracle_table)
    assert len(diff) == 0, (
        f"{prefix}: {len(diff)} table bytes differ from the committed oracle"
    )


@pytest.mark.parametrize("prefix", ["c1x1", "c3x3"])
def test_tile_params_cover_the_whole_cout(prefix):
    p = fix.TILE_PARAMS[prefix]
    assert fix.COUT % p["tw"] == 0


def test_c3x3_scaffold_has_a_duplicate_copy_but_c1x1_does_not():
    assert fix.TILE_PARAMS["c3x3"]["dup_offset"] == 37120
    assert fix.TILE_PARAMS["c1x1"]["dup_offset"] is None
