"""Regression coverage for docs/axera-conv-gather-loose-ends.md's items 1 and 2.

No Docker or device is required: item 1 diffs three committed compiled
fixtures; item 2 checks the index-layout invariant on committed compiled and
retargeted fixtures, plus the pure-numpy ground truth used to interpret the
device numbers recorded in the doc.
"""

import gzip
import os
import struct
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from conv_scale_divergence_check import divergence_report  # noqa: E402
from gather_aggregate_addsum_check import (  # noqa: E402
    GROUP,
    IDX_ADV,
    IDX_REF,
    N_IDX,
    X_SHAPE,
    groundtruth,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_gather_loose_ends")


def test_item1_patch_scales_moves_zero_bytes_closer_to_native():
    report = divergence_report()
    assert report["reference_vs_native"] == 1548
    assert report["patched_vs_native"] == 1548
    assert report["reference_vs_patched"] == 16


def _npu_params(gz_name):
    with gzip.open(os.path.join(_FIXTURES, gz_name), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


@pytest.mark.parametrize(
    "gz_name,expected_idx",
    [
        ("addsum_a_reference_narrow.axmodel.gz", IDX_REF),
        ("addsum_b_native_adversarial.axmodel.gz", IDX_ADV),
        ("addsum_c_wide_reference.axmodel.gz", IDX_REF),
        ("addsum_a_retargeted_to_adv.axmodel.gz", IDX_ADV),
        ("addsum_c_retargeted_to_adv.axmodel.gz", IDX_ADV),
    ],
)
def test_item2_leading_words_are_the_build_own_indices(gz_name, expected_idx):
    table = _npu_params(gz_name)
    words = struct.unpack(f"<{N_IDX}I", table[: N_IDX * 4])
    assert list(words) == expected_idx


def test_item2_retargeting_preserves_everything_after_the_index_words():
    # Retargeting a's own compiled table to idx_adv changes only the leading
    # N_IDX words; everything after (scales, mask constant, mcode-adjacent
    # table tail) is untouched, the same invariant memory_emit.py's
    # standalone Gather emitter documents.
    original = _npu_params("addsum_a_reference_narrow.axmodel.gz")
    retargeted = _npu_params("addsum_a_retargeted_to_adv.axmodel.gz")
    assert len(original) == len(retargeted)
    assert original[N_IDX * 4 :] == retargeted[N_IDX * 4 :]


def test_item2_groundtruth_shape_and_index_bands():
    rng = np.random.RandomState(0)
    x = np.zeros(X_SHAPE, dtype=np.float32)
    x[..., :39] = rng.uniform(-0.3, 0.3, x[..., :39].shape)
    x[..., 39:] = rng.uniform(-0.9, 0.9, x[..., 39:].shape)
    y_ref = groundtruth(x, IDX_REF)
    y_adv = groundtruth(x, IDX_ADV)
    assert y_ref.shape == y_adv.shape == (16, 1, 512, GROUP)
    # idx_ref draws only from the narrow [-0.3,0.3] band (summed 4-fold, minus
    # the mask's zeroed positions), idx_adv only from the wide [-0.9,0.9] band --
    # the adversarial aggregate should have a visibly larger typical magnitude.
    assert np.abs(y_adv).mean() > np.abs(y_ref).mean()


def test_item2_all_idx_within_declared_bands():
    assert all(v < 39 for v in IDX_REF)
    assert all(39 <= v < 49 for v in IDX_ADV)
