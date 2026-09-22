"""Regression coverage for the wide-channel Conv weight-learning characterization.

No Docker/device needed: checks the committed learned maps against a committed
held-out oracle build (`conv256_holdout_oracle.axmodel.gz`, a genuine fresh
Pulsar2 compile of weights not in the 47-build learning set).
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

import conv_learn_wide  # noqa: E402
import emitter  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_learn_wide")


def _load_gz_model(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _table_of_model(model, name="npu_params"):
    for init in model.graph.initializer:
        if init.name == name:
            return np.frombuffer(bytes(init.raw_data), dtype=np.uint8)
    raise KeyError(name)


@pytest.fixture(scope="module")
def maps():
    return conv_learn_wide.load_maps()


@pytest.fixture(scope="module")
def holdout():
    z = np.load(os.path.join(_FIXTURES, "conv256_holdout_weights.npz"))
    return z["weight"], z["bias"]


def test_maps_cover_every_weight_code_bit_with_no_collisions(maps):
    # Cout*Cin*K*K*8 = 256*256*3*3*8
    expected_mapped = 256 * 256 * 3 * 3 * 8
    mapped = int((maps["origin"] != emitter.CONST).sum())
    assert mapped == expected_mapped


def test_weight_code_bits_are_byte_exact_against_a_held_out_native_build(maps, holdout):
    """The load-bearing claim: outside the scattered requantisation region, an
    emitted table matches a genuine fresh Pulsar2 compile of unseen weights,
    everywhere."""
    w, _bias = holdout
    reference = _load_gz_model("conv256_reference.axmodel.gz")
    oracle = _load_gz_model("conv256_holdout_oracle.axmodel.gz")
    ref_table = _table_of_model(reference)
    native_table = _table_of_model(oracle)

    codes = emitter.codes_of(w)
    emitted = emitter.emit_table(ref_table, maps["origin"].astype(np.int64), codes)

    ambiguous_bytes = set(maps["ambiguous_bytes"].tolist())
    mismatched = np.flatnonzero(emitted != native_table)
    assert len(mismatched) > 0  # the scaffold region is real, not a fluke of this build
    assert set(mismatched.tolist()) <= ambiguous_bytes


def test_requant_second_pass_explains_most_but_not_all_of_the_scaffold_region(maps):
    total = len(maps["ambiguous_bytes"]) * 8
    unexplained = len(maps["ambiguous2"])
    mapped2 = int((maps["origin2"] != emitter.CONST).sum())
    # not a hard invariant of the format -- a regression pin on what this learning
    # run actually found (about 51%), so a future re-run's drift is visible
    assert 0.45 < mapped2 / total < 0.6
    assert unexplained > total * 0.3  # a real, sizeable, still-unexplained remainder


def test_patch_mcode_scale_zero_form_a_only_touches_the_named_offsets():
    mcode = (np.arange(6000, dtype=np.int64) % 256).astype(np.uint8)
    patched = conv_learn_wide.patch_mcode_scale_zero_form_a(mcode, 0.05, 130.0)
    changed = set(np.flatnonzero(patched != mcode).tolist())
    touched = set()
    for off in conv_learn_wide.MCODE_SCALE_OFFSETS_FORM_A:
        touched.update(range(off, off + 4))
    touched.add(conv_learn_wide.MCODE_ZERO_OFFSET_FORM_A)
    assert changed <= touched
