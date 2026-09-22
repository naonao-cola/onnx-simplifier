"""Regression coverage for the ResNet18 stem Conv weight-learning emitter.

No Docker/device needed: these check the *emitted* table/mcode against
committed reference fixtures. `docs/axera-conv-weight-learn-stem-full.md` has
the real Pulsar2-build (36+3 builds) and AX8850-device evidence this stands
in for -- these tests only confirm the emission pipeline is reproducible.
"""

import gzip
import json
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

import emitter  # noqa: E402
from conv_weight_learn_stem import (  # noqa: E402
    BLOCK_AT,
    BLOCK_LEN,
    SCALE_OFFSETS,
    UNPATCHABLE_ZERO_POINTS,
    ZERO_OFFSET,
    emit_stem_conv,
    is_patchable,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_learn_stem")


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
    data = next(bytes(i.raw_data) for i in model.graph.initializer if i.name == name)
    return data, name


def _quant_scale_zero(gz_json_name, tensor):
    with gzip.open(os.path.join(_FIXTURES, gz_json_name), "rt") as f:
        doc = json.load(f)
    for cfg in doc["tensor_configs"].values():
        if tensor in cfg:
            v = doc["values"][str(cfg[tensor]["hash"])]
            return float(v["scale"][0]), float(v["zero_point"][0])
    raise KeyError(tensor)


def test_map_matches_the_real_shape_with_zero_collisions():
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "stem_map.npz")
    )
    assert tuple(shape) == (64, 3, 7, 7)
    weight_bits = 64 * 3 * 7 * 7 * 8
    assert int((origin != emitter.CONST).sum()) == weight_bits


def test_holdout_weight_code_region_matches_native_exactly():
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "stem_map.npz")
    )
    reference = _load_gz_model("reference.axmodel.gz")
    ref_table = _table_of(reference)
    native = _load_gz_model("holdout_native.axmodel.gz")
    native_table = _table_of(native)

    weights = np.load(os.path.join(_FIXTURES, "holdout_weights.npz"))
    w = weights["w"]
    codes = emitter.codes_of(w)
    emitted_weights_only = emitter.emit_table(ref_table, origin, codes)

    amb_bytes = set((ambiguous // 8).tolist())
    diff = [
        i
        for i in range(len(ref_table))
        if emitted_weights_only[i] != native_table[i] and i not in amb_bytes
    ]
    assert diff == []


def test_full_emission_matches_native_table_within_float_tolerance():
    origin, const, ambiguous, shape = emitter.load_map(
        os.path.join(_FIXTURES, "stem_map.npz")
    )
    reference = _load_gz_model("reference.axmodel.gz")
    ref_table = _table_of(reference)
    native = _load_gz_model("holdout_native.axmodel.gz")
    native_table = _table_of(native)
    ref_mcode, mcode_name = _mcode_of(reference)

    weights = np.load(os.path.join(_FIXTURES, "holdout_weights.npz"))
    w, b = weights["w"], weights["b"]
    x_scale, x_zero = _quant_scale_zero("reference_quant.json.gz", "x")
    y_scale, y_zero = _quant_scale_zero("holdout_quant.json.gz", "y")

    assert is_patchable(y_zero)
    new_table, new_mcode = emit_stem_conv(
        ref_table, ref_mcode, origin, w, b, x_scale, x_zero, y_scale, y_zero
    )

    diff_bytes = [i for i in range(len(new_table)) if new_table[i] != native_table[i]]
    # Every mismatch is inside the 512-byte requantisation block, and each is a
    # tiny float32-rounding residual, not a wrong formula -- see the doc.
    assert all(BLOCK_AT <= i < BLOCK_AT + BLOCK_LEN for i in diff_bytes)
    block_a = np.frombuffer(
        bytes(new_table[BLOCK_AT : BLOCK_AT + BLOCK_LEN]), dtype=np.uint8
    ).view(np.float32)
    block_b = np.frombuffer(
        bytes(native_table[BLOCK_AT : BLOCK_AT + BLOCK_LEN]), dtype=np.uint8
    ).view(np.float32)
    assert np.max(np.abs(block_a - block_b)) < 1e-3


def test_full_emission_mcode_matches_native_outside_free_bytes():
    origin, _const, _ambiguous, _shape = emitter.load_map(
        os.path.join(_FIXTURES, "stem_map.npz")
    )
    reference = _load_gz_model("reference.axmodel.gz")
    ref_table = _table_of(reference)
    ref_mcode, _name = _mcode_of(reference)
    native = _load_gz_model("holdout_native.axmodel.gz")
    native_mcode, _name2 = _mcode_of(native)

    weights = np.load(os.path.join(_FIXTURES, "holdout_weights.npz"))
    w, b = weights["w"], weights["b"]
    x_scale, x_zero = _quant_scale_zero("reference_quant.json.gz", "x")
    y_scale, y_zero = _quant_scale_zero("holdout_quant.json.gz", "y")

    _new_table, new_mcode = emit_stem_conv(
        ref_table, ref_mcode, origin, w, b, x_scale, x_zero, y_scale, y_zero
    )
    assert len(new_mcode) == len(native_mcode)
    diff = [i for i in range(len(new_mcode)) if new_mcode[i] != native_mcode[i]]
    # The known "free"/scheduling bytes this shape's campaign found -- a
    # per-build value that does not affect device output (see the doc).
    assert len(diff) <= 12


def test_scale_and_zero_offsets_hold_the_reference_values():
    reference = _load_gz_model("reference.axmodel.gz")
    ref_mcode, _name = _mcode_of(reference)
    x_scale, x_zero = _quant_scale_zero("reference_quant.json.gz", "x")
    ref_y_scale, ref_y_zero = _quant_scale_zero("reference_quant.json.gz", "y")
    import struct

    for off in SCALE_OFFSETS:
        (value,) = struct.unpack("<f", ref_mcode[off : off + 4])
        assert abs(value - ref_y_scale) < 1e-9
    assert ref_mcode[ZERO_OFFSET] == int(round(ref_y_zero)) & 0xFF


@pytest.mark.parametrize("z", UNPATCHABLE_ZERO_POINTS)
def test_unpatchable_zero_points_are_rejected(z):
    assert not is_patchable(z)


def test_emit_stem_conv_raises_on_unpatchable_zero_point():
    origin, _const, _ambiguous, _shape = emitter.load_map(
        os.path.join(_FIXTURES, "stem_map.npz")
    )
    reference = _load_gz_model("reference.axmodel.gz")
    ref_table = _table_of(reference)
    ref_mcode, _name = _mcode_of(reference)
    weights = np.load(os.path.join(_FIXTURES, "holdout_weights.npz"))
    w, b = weights["w"], weights["b"]

    with pytest.raises(ValueError, match="unpatchable"):
        emit_stem_conv(ref_table, ref_mcode, origin, w, b, 0.02, 128.0, 0.06, 127.0)
