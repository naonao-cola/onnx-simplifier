"""Regression coverage for the real-shape Conv weight-learning emitter.

No Docker/device needed: these check the *emitted* table/mcode against
committed reference fixtures. `docs/axera-conv-weight-learn-downsample.md`
has the real Pulsar2-build and AX8850-device evidence this is standing in
for -- these tests only confirm the emission pipeline is reproducible.
"""

import gzip
import json
import os
import sys

import numpy as np
import onnx

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import emitter  # noqa: E402
from conv_bias_requant import emit_conv_table  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_learn_downsample")


def _load_gz_model(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _table_of(model, name="npu_params"):
    return next(
        np.frombuffer(bytes(i.raw_data), dtype=np.uint8)
        for i in model.graph.initializer
        if i.name == name
    )


def _mcode_of(model):
    name = next(i.name for i in model.graph.initializer if i.name.endswith("_neu"))
    return (
        np.frombuffer(
            bytes(next(i for i in model.graph.initializer if i.name == name).raw_data),
            dtype=np.uint8,
        ),
        name,
    )


def _quant_scale_zero(gz_json_name, tensor):
    with gzip.open(os.path.join(_FIXTURES, gz_json_name), "rt") as f:
        doc = json.load(f)
    tc = list(doc["tensor_configs"].values())[0][tensor]
    v = doc["values"][str(tc["hash"])]
    return float(v["scale"][0]), float(v["zero_point"][0])


def test_shape1_1x1_downsample_weight_code_and_block_match_native():
    """The full pipeline: k=32-build-learned weight-code map + the bias-corrected
    block formula reproduces the real compiled table for held-out weights, to
    within the float32 rounding `requant_block`'s own docstring documents."""
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "s1_map.npz")
    )
    assert tuple(shape) == (128, 64, 1, 1)

    reference = _load_gz_model("s1_reference.axmodel.gz")
    ref_table = _table_of(reference)
    native = _load_gz_model("s1_holdout_native.axmodel.gz")
    native_table = _table_of(native)

    weights = np.load(os.path.join(_FIXTURES, "s1_holdout_weights.npz"))
    w, b = weights["w"], weights["b"]
    x_scale, x_zero = _quant_scale_zero("s1_reference_quant.json.gz", "x")
    y_scale, y_zero = _quant_scale_zero("s1_holdout_quant.json.gz", "y")

    emitted = emit_conv_table(
        ref_table,
        origin,
        w,
        b,
        x_scale,
        x_zero,
        y_scale,
        y_zero,
        block_at=9216,
        block_len=2 * 4 * 128,
    )
    assert emitted.shape == native_table.shape

    # Outside the per-channel bias/scale block: byte-exact (the weight-code
    # bit-permutation portion, k=32, zero collisions).
    outside = np.ones(len(emitted), dtype=bool)
    outside[9216 : 9216 + 2 * 4 * 128] = False
    assert np.array_equal(emitted[outside], native_table[outside])

    # Inside the block: `m` (per-channel scale) and the bias half are both
    # within float32 rounding, not bit-identical -- this pipeline takes
    # `x_scale` from the reference build (the realistic case: a real emission
    # has no separately-recalibrated holdout to read it from), which differs
    # from the holdout's own by about 1e-8 in the last decimal digit and
    # propagates to a few ULP here. Confirmed numerically harmless on device
    # (`docs/axera-conv-weight-learn-downsample.md`'s device section).
    m_emitted = emitted[9216 + 512 : 9216 + 1024].view("<f4")
    m_native = native_table[9216 + 512 : 9216 + 1024].view("<f4")
    assert (
        np.abs(m_emitted.astype(np.float64) - m_native.astype(np.float64)).max() < 1e-6
    )
    bias_emitted = emitted[9216 : 9216 + 512].view("<f4").astype(np.float64)
    bias_native = native_table[9216 : 9216 + 512].view("<f4").astype(np.float64)
    assert np.abs(bias_emitted - bias_native).max() < 1e-3


def test_shape1_zero_point_byte_is_at_the_documented_offset():
    """Pins the one mcode byte this pipeline needs a caller to patch by hand
    (`docs/axera-conv-weight-learn-downsample.md`'s "the real bug" section):
    `emitter.py`'s own `learn_mcode`+`emit_mcode` refuses this holdout's zero
    point as "unpatchable" (a real, separately reported over-caution), so this
    only confirms the byte position and that a plain overwrite is correct --
    it does not claim the built-in API path works unmodified."""
    reference = _load_gz_model("s1_reference.axmodel.gz")
    ref_mc, _ = _mcode_of(reference)
    native = _load_gz_model("s1_holdout_native.axmodel.gz")
    native_mc, _ = _mcode_of(native)

    x_scale, x_zero = _quant_scale_zero("s1_reference_quant.json.gz", "y")
    old_zero = int(round(x_zero)) & 0xFF
    _, new_zero = _quant_scale_zero("s1_holdout_quant.json.gz", "y")
    new_zero = int(round(new_zero)) & 0xFF

    assert ref_mc[4010] == old_zero
    assert native_mc[4010] == new_zero
    assert old_zero != new_zero


def test_shape2_3x3_downsample_weight_codes_match_native():
    """The 3x3 downsample's harder case: only the part this project fully
    validated -- the k=38-build weight-code map (0 collisions). The bias/scale
    auxiliary block is a documented, open gap (not a single contiguous span
    the way 1x1's is) and is not exercised here; see
    `test_shape2_scale_array_matches_formula` for the one piece of it that
    *is* confirmed."""
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "s2_map.npz")
    )
    assert tuple(shape) == (128, 64, 3, 3)

    reference = _load_gz_model("s2_reference.axmodel.gz")
    ref_table = _table_of(reference)
    native = _load_gz_model("s2_holdout_native.axmodel.gz")
    native_table = _table_of(native)

    weights = np.load(os.path.join(_FIXTURES, "s2_holdout_weights.npz"))
    w = weights["w"]
    codes = emitter.codes_of(w)
    emitted = emitter.emit_table(ref_table, origin, codes)

    # Every table byte the k=38 map actually names a code-bit origin for
    # must match exactly -- the weight-code bit-permutation portion alone.
    origin_arr = np.asarray(origin)
    mapped_byte = np.zeros(len(ref_table), dtype=bool)
    mapped_byte[np.unique(np.flatnonzero(origin_arr != -1) // 8)] = True
    assert np.array_equal(emitted[mapped_byte], native_table[mapped_byte])


def test_shape2_first_channel_groups_scale_value_present_verbatim():
    """A narrower, honestly-scoped version of what looked at first like a
    clean 512-byte `m` array (`docs/axera-conv-weight-learn-downsample.md`'s
    "correction" section): channel 0's `m = x_scale*w_scale/y_scale` value is
    present verbatim at a fixed offset, but the full 128-channel array is
    NOT one contiguous span -- it is scattered across (at least) four
    per-32-channel regions in an order this project did not decode further.
    This only pins the one part that is solid: the formula's value, for the
    first channel of the first group, really is in the table."""
    weights = np.load(os.path.join(_FIXTURES, "s2_holdout_weights.npz"))
    w = weights["w"]
    native = _load_gz_model("s2_holdout_native.axmodel.gz")
    native_table = _table_of(native)
    x_scale, _ = _quant_scale_zero("s2_holdout_quant.json.gz", "x")
    y_scale, _ = _quant_scale_zero("s2_holdout_quant.json.gz", "y")
    w_scale = emitter.weight_scales(w)
    m_computed = (x_scale * w_scale / y_scale).astype(np.float32)
    assert np.array_equal(native_table[18560:18564].view("<f4"), m_computed[:1])
