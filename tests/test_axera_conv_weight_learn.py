"""Regression coverage for the real-shape Conv weight-learning campaign.

Fixtures: a compiled reference build and a held-out compiled build of the
same real ResNet18 stage-1 shape (`Conv(cin=64, cout=64, k=3x3, stride=1,
pad=1, batch=16, spatial=56)`, with a real float bias), plus the
bit-permutation map learned from 48 such builds. See
`docs/axera-conv-weight-learn-stem.md` for the full campaign and the device
verification (not reproducible here without Docker/a card).
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

import conv_weight_learn as cwl  # noqa: E402
import emitter  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_weight_learn")
_BLOCK_AT = 36864
_BLOCK_LEN = 512
_CHANNELS = 64


def _load_gz_model(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _table(model):
    return next(
        np.frombuffer(bytes(i.raw_data), np.uint8)
        for i in model.graph.initializer
        if i.name == "npu_params"
    )


def _mcode(model):
    name = next(i.name for i in model.graph.initializer if i.name.endswith("_neu"))
    return next(
        np.frombuffer(bytes(i.raw_data), np.uint8)
        for i in model.graph.initializer
        if i.name == name
    )


def _load_npy_gz(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return np.load(f)


@pytest.fixture(scope="module")
def fixtures():
    ref_model = _load_gz_model("reference.axmodel.gz")
    holdout_model = _load_gz_model("holdout_native.axmodel.gz")
    return {
        "ref_table": _table(ref_model),
        "ref_mcode": _mcode(ref_model),
        "holdout_table": _table(holdout_model),
        "holdout_mcode": _mcode(holdout_model),
        "ref_w": _load_npy_gz("reference_w.npy.gz"),
        "ref_b": _load_npy_gz("reference_b.npy.gz"),
        "holdout_w": _load_npy_gz("holdout_w.npy.gz"),
        "holdout_b": _load_npy_gz("holdout_b.npy.gz"),
        "map": np.load(os.path.join(_FIXTURES, "stage1_map.npz")),
    }


def test_reference_table_matches_its_own_weights_through_the_learned_map(fixtures):
    """Emitting the reference's own weights through its own learned map must
    reproduce the reference table's weight-code region exactly (identity
    check on the map itself, independent of any held-out claim)."""
    origin = fixtures["map"]["origin"]
    codes = emitter.codes_of(fixtures["ref_w"])
    table = emitter.emit_table(fixtures["ref_table"], origin, codes)
    # the code region is the first Cout*Cin*K bytes; the block after it is
    # scaffolding this call does not touch.
    code_bytes = fixtures["ref_w"].size
    assert np.array_equal(table[:code_bytes], fixtures["ref_table"][:code_bytes])


def test_held_out_weights_emit_the_exact_code_region_pulsar2_built(fixtures):
    """The point of the campaign: a weight set none of the 48 learning builds
    saw still emits Pulsar2's own weight-code bytes exactly, through the
    learned map alone."""
    origin = fixtures["map"]["origin"]
    codes = emitter.codes_of(fixtures["holdout_w"])
    table = emitter.emit_table(fixtures["ref_table"], origin, codes)
    code_bytes = fixtures["holdout_w"].size
    assert np.array_equal(table[:code_bytes], fixtures["holdout_table"][:code_bytes])


def test_the_uncorrected_requant_block_is_off_by_the_bias_term(fixtures):
    """`emitter.requant_block` alone (no bias correction) must NOT match a
    real biased build -- pinning down the exact defect `conv_weight_learn`
    fixes, so a future emitter.py fix does not silently go unnoticed here."""
    import emitter

    codes = emitter.codes_of(fixtures["holdout_w"])
    w_scale = emitter.weight_scales(fixtures["holdout_w"])
    # scales taken from the committed build directly would need the raw
    # quant_axmodel.json, which is not committed (only the compiled model
    # is); reuse the values recorded in docs/axera-conv-weight-learn-stem.md
    # for the holdout build instead, so this test needs no extra fixture.
    x_scale, x_zero = 0.007058821618556976, 127.0
    y_scale, y_zero = 0.03239550068974495, 125.0
    uncorrected = emitter.requant_block(
        codes, x_scale, x_zero, y_scale, y_zero, w_scale
    )
    stored = fixtures["holdout_table"][_BLOCK_AT : _BLOCK_AT + _BLOCK_LEN]
    # the uncorrected formula measurably disagrees somewhere in the block
    assert not np.array_equal(uncorrected, stored)


def test_the_bias_corrected_requant_block_matches_to_float32_rounding(fixtures):
    x_scale, x_zero = 0.007058821618556976, 127.0
    y_scale, y_zero = 0.03239550068974495, 125.0
    codes = emitter.codes_of(fixtures["holdout_w"])
    w_scale = emitter.weight_scales(fixtures["holdout_w"])
    block = cwl.requant_block_biased(
        codes, x_scale, x_zero, y_scale, y_zero, w_scale, fixtures["holdout_b"]
    )
    stored = fixtures["holdout_table"][_BLOCK_AT : _BLOCK_AT + _BLOCK_LEN]
    got = block.view(np.float32)
    want = stored.view(np.float32)
    # matches to float32-rounding-scale error, not bit-exactly -- see the doc
    # for why (a different internal order of operations from Pulsar2's own).
    assert np.abs(got - want).max() < 2e-4


def test_tensor_scales_reads_the_expected_keys():
    """`tensor_scales` on a real captured `quant_axmodel.json` shape (a
    single-node graph's one tensor_configs entry) returns x/w/b/y."""
    import json
    import tempfile

    doc = {
        "tensor_configs": {
            "y": {
                "x": {"hash": 1},
                "w": {"hash": 2},
                "b": {"hash": 3},
                "y": {"hash": 4},
            }
        },
        "values": {
            "1": {"scale": [0.01], "zero_point": [127.0]},
            "2": {"scale": [0.001, 0.002], "zero_point": [0.0, 0.0]},
            "3": {"scale": [1e-5, 2e-5], "zero_point": [0.0, 0.0]},
            "4": {"scale": [0.03], "zero_point": [120.0]},
        },
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "quant_axmodel.json")
        json.dump(doc, open(path, "w"))
        out = cwl.tensor_scales(path)
    assert set(out) == {"x", "w", "b", "y"}
    assert out["x"]["scale"] == [0.01]
    assert out["y"]["zero_point"] == [120.0]
