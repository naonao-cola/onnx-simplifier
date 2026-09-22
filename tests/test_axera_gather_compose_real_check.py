"""Regression coverage for the real-scale composed-Gather index layout check.

See `docs/axera-gather-compose-real-scale.md`: these fixtures are compiled
`Reshape`-skipped `Gather(axis=3, 32768 indices) -> Mul(mask)` models built
from the ResNet18 stem's own real index vector and padding mask, one 32,768
element slice, not the toy `[1,1,4,16]` graph `docs/axera-compose.md` used.
"""

import gzip
import os
import struct
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from gather_compose_real_check import gather_indices, patch_indices  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "gather_compose_real")
_N = 32768


def _unzip(tmp_path, name):
    path = tmp_path / name.removesuffix(".gz")
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        path.write_bytes(f.read())
    return str(path)


def _npu_params_words(path):
    model = onnx.load(path, load_external_data=False)
    table = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )
    return struct.unpack(f"<{len(table) // 4}I", table)


def test_reference_and_shuffled_share_layout_but_differ_only_in_indices(tmp_path):
    ref = _unzip(tmp_path, "mid_reference.axmodel.gz")
    native = _unzip(tmp_path, "mid_native_shuffled.axmodel.gz")
    ref_words = _npu_params_words(ref)
    native_words = _npu_params_words(native)
    assert len(ref_words) == len(native_words)
    ref_idx = ref_words[:_N]
    native_idx = native_words[:_N]
    # A real, non-degenerate index vector (not e.g. all zero, not identity).
    assert len(set(ref_idx)) > 1000
    assert ref_idx != native_idx
    assert sorted(ref_idx) == sorted(native_idx), (
        "the two builds' Gathers select the same multiset of source elements "
        "(a permutation), only in a different order -- by construction of this fixture pair"
    )


def test_patch_indices_reproduces_a_native_build_index_order(tmp_path):
    ref = _unzip(tmp_path, "mid_reference.axmodel.gz")
    native = _unzip(tmp_path, "mid_native_shuffled.axmodel.gz")
    native_idx = gather_indices(native, _N)

    out = str(tmp_path / "emitted.axmodel")
    patch_indices(ref, out, native_idx)

    assert gather_indices(out, _N) == native_idx
    # everything after the index words (scales, mask) is untouched, i.e. still the reference's
    ref_words = _npu_params_words(ref)
    out_words = _npu_params_words(out)
    assert ref_words[_N:] == out_words[_N:]


def test_patch_indices_rejects_an_oversized_index_vector(tmp_path):
    ref = _unzip(tmp_path, "mid_reference.axmodel.gz")
    with pytest.raises(ValueError):
        patch_indices(ref, str(tmp_path / "bad.axmodel"), list(range(10_000_000)))


def test_fullrange_fixture_has_the_same_index_layout_as_the_narrow_reference(tmp_path):
    ref = _unzip(tmp_path, "mid_reference.axmodel.gz")
    wide = _unzip(tmp_path, "mid_reference_fullrange.axmodel.gz")
    # Same indices (the fullrange fixture differs only in calibration data range).
    assert gather_indices(ref, _N) == gather_indices(wide, _N)
