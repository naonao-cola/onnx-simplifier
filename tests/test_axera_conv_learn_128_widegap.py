"""Regression coverage for the 128/128 Conv weight-learn campaign and the
conv256 scaffold-resolution follow-up.

Fixtures: a reference build and a held-out build for `Conv(128,128,3,3)`
(the ResNet18 stage-2 shape), the 48-build learned map and its second-pass
scaffold map; and the same for a fresh 65-build `Conv(256,256,3,3)` campaign.
See `docs/axera-conv-weight-learn-128-and-widegap.md` for the full campaign,
the device verification, and the zero-point gap (not reproducible here
without Docker/a card).
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

import conv_learn_128_widegap as clw  # noqa: E402
import emitter  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "conv_learn_128_widegap")


def _load_gz_model(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _table(model):
    return next(
        np.frombuffer(bytes(i.raw_data), dtype=np.uint8)
        for i in model.graph.initializer
        if i.name == "npu_params"
    )


@pytest.mark.parametrize(
    "ref_name,hold_name,weights_name,map_name",
    [
        (
            "conv128_reference.axmodel.gz",
            "conv128_holdout_native.axmodel.gz",
            "conv128_holdout_weights.npz",
            "conv128_map.npz",
        ),
        (
            "conv256_fresh_reference.axmodel.gz",
            "conv256_fresh_holdout_native.axmodel.gz",
            "conv256_fresh_holdout_weights.npz",
            "conv256_fresh_map.npz",
        ),
    ],
)
def test_weight_code_region_matches_native_holdout(
    ref_name, hold_name, weights_name, map_name
):
    ref_model = _load_gz_model(ref_name)
    hold_model = _load_gz_model(hold_name)
    ref_table = _table(ref_model)
    native_table = _table(hold_model)

    m = np.load(os.path.join(_FIXTURES, map_name))
    origin = m["origin"]
    wz = np.load(os.path.join(_FIXTURES, weights_name))
    w = wz["w"]

    codes = emitter.codes_of(w)
    emitted = emitter.emit_table(ref_table, origin, codes)

    # Only bits the map actually names a code-bit source for are checked here --
    # the scaffold region (everything else) is a separate, documented, partial result.
    named = origin != emitter.CONST
    bit_emitted = np.unpackbits(emitted.reshape(1, -1), axis=1, bitorder="little")[0]
    bit_native = np.unpackbits(native_table.reshape(1, -1), axis=1, bitorder="little")[
        0
    ]
    assert np.array_equal(bit_emitted[named], bit_native[named])


def test_conv128_scaffold_map_has_expected_shape_and_resolution():
    m1 = np.load(os.path.join(_FIXTURES, "conv128_map.npz"))
    m2 = np.load(os.path.join(_FIXTURES, "conv128_scaffold_map.npz"))
    outer = m2["outer_ambiguous"]
    assert len(outer) == len(m1["ambiguous"])
    resolved = np.sum(m2["origin"] != emitter.CONST)
    frac = resolved / len(outer)
    # 80.4% at the time this was measured; a regression test pins the order of
    # magnitude, not the exact float (a rebuild with different seeds could shift it
    # slightly without indicating a real regression).
    assert 0.75 < frac < 0.85


def test_conv256_fresh_scaffold_resolution_improved_over_the_original_51_percent():
    m1 = np.load(os.path.join(_FIXTURES, "conv256_fresh_map.npz"))
    m2 = np.load(os.path.join(_FIXTURES, "conv256_fresh_scaffold_map.npz"))
    outer = m2["outer_ambiguous"]
    assert len(outer) == len(m1["ambiguous"])
    resolved = np.sum(m2["origin"] != emitter.CONST)
    frac = resolved / len(outer)
    # docs/axera-conv-weight-learn-wide.md's original 47-48-build campaign resolved 51%;
    # this fresh 65-build campaign should do meaningfully better, confirming more builds
    # help (not a plateau) rather than just reproducing the same number.
    assert frac > 0.65


def test_conv128_and_conv256_fresh_weight_code_bit_count_matches_shape():
    for map_name, cout, cin, k in [
        ("conv128_map.npz", 128, 128, 3),
        ("conv256_fresh_map.npz", 256, 256, 3),
    ]:
        m = np.load(os.path.join(_FIXTURES, map_name))
        mapped = int(np.sum(m["origin"] != emitter.CONST))
        assert mapped == cout * cin * k * k * 8


def test_emit_conv_128_reproduces_native_weight_codes_for_conv128():
    ref_model = _load_gz_model("conv128_reference.axmodel.gz")
    hold_model = _load_gz_model("conv128_holdout_native.axmodel.gz")
    ref_table = _table(ref_model)
    native_table = _table(hold_model)

    m1 = np.load(os.path.join(_FIXTURES, "conv128_map.npz"))
    m2 = np.load(os.path.join(_FIXTURES, "conv128_scaffold_map.npz"))
    wz = np.load(os.path.join(_FIXTURES, "conv128_holdout_weights.npz"))

    # x/y scale+zero from the real campaign (recorded in the doc; not re-derivable
    # from the gzipped .axmodel alone offline).
    x_scale, x_zero = 0.007058822084218264, 128
    y_scale, y_zero = 0.03161545842885971, 127

    emitted = clw.emit_conv_128(
        ref_table,
        m1["origin"],
        m2["origin"],
        m2["outer_ambiguous"],
        wz["w"],
        x_scale,
        x_zero,
        y_scale,
        y_zero,
        wz["b"],
    )

    diff_frac = np.mean(emitted != native_table)
    # ~0.13% differed in the real campaign (the unresolved ~20% of scaffold bits,
    # confirmed on-device to be dominated by the separate, unpatched zero-point --
    # see the doc); this pins that it stays small, not that it reaches zero.
    assert diff_frac < 0.01
