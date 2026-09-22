"""Regression coverage for the (best-effort, not always exact) teng2 cluster
extrapolation tool. See scripts/axera/teng2_cluster_patch.py's docstring and
docs/axera-teng2-sqrt-blocks.md for what this does and does not achieve."""

import gzip
import os
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from teng2_cluster_patch import cluster_positions, extrapolate  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "teng2_sqrt_blocks")


def _unzip(tmp_path, name):
    path = tmp_path / f"{name}.axmodel"
    with gzip.open(os.path.join(_FIXTURES, f"{name}.axmodel.gz"), "rb") as source:
        path.write_bytes(source.read())
    return str(path)


def _mcode(path):
    model = onnx.load(path, load_external_data=False)
    return bytes(
        next(i.raw_data for i in model.graph.initializer if i.name.endswith("_neu"))
    )


def _params(path):
    model = onnx.load(path, load_external_data=False)
    return bytes(
        next(i.raw_data for i in model.graph.initializer if i.name == "npu_params")
    )


def test_cluster_positions_finds_the_known_slopes(tmp_path):
    ref_a = _unzip(tmp_path, "sqrt_c157")
    ref_b = _unzip(tmp_path, "sqrt_c158")
    fields = cluster_positions(ref_a, ref_b)
    deltas = sorted({delta for _, _, delta in fields})
    # The recurring per-step deltas this cluster and every other measured one shares.
    assert deltas == [1, 14, 28, 49]


def test_cluster_positions_rejects_mismatched_shapes(tmp_path):
    ref_a = _unzip(tmp_path, "sqrt_c157")
    ref_c = _unzip(tmp_path, "sqrt_c159")  # two channels apart, not one
    with pytest.raises(ValueError, match="one channel apart"):
        cluster_positions(ref_a, ref_c)


def test_extrapolate_rejects_a_target_that_is_not_the_next_step(tmp_path):
    ref_a = _unzip(tmp_path, "sqrt_c157")
    ref_b = _unzip(tmp_path, "sqrt_c158")
    with pytest.raises(ValueError, match="next/previous step"):
        extrapolate(ref_a, ref_b, 200, str(tmp_path / "bad.axmodel"))


def test_extrapolate_is_exact_for_the_157_158_159_cluster(tmp_path):
    # This is the one held-out triple (of three checked while building this tool)
    # with no carry-boundary byte between the reference pair and the target --
    # see docs/axera-teng2-sqrt-blocks.md for the other two, which are NOT exact.
    ref_a = _unzip(tmp_path, "sqrt_c157")
    ref_b = _unzip(tmp_path, "sqrt_c158")
    real_c159 = _unzip(tmp_path, "sqrt_c159")
    out = str(tmp_path / "pred_159.axmodel")

    extrapolate(ref_a, ref_b, 159, out)

    pred_mcode, real_mcode = bytearray(_mcode(out)), bytearray(_mcode(real_c159))
    # 301-325 is ordinary compiler noise between any two independent builds of the same
    # shape (see docs/axera-memory-op-generator.md); zero it on both sides before comparing.
    pred_mcode[301:326] = bytes(25)
    real_mcode[301:326] = bytes(25)
    assert bytes(pred_mcode) == bytes(real_mcode)
    assert _params(out) == _params(real_c159)
    pred_shape = onnx.load(out, load_external_data=False)
    assert [
        d.dim_value for d in pred_shape.graph.input[0].type.tensor_type.shape.dim
    ] == [
        1,
        159,
        56,
        56,
    ]


def test_extrapolate_can_miss_a_carry_byte_known_limitation(tmp_path):
    # Documents, rather than hides, the tool's real failure mode: a field whose
    # low byte is identical between the two references (no visible slope) but
    # carries into a new value at the target is invisible to a 2-reference diff.
    ref_a = _unzip(tmp_path, "sqrt_c153")
    ref_b = _unzip(tmp_path, "sqrt_c154")
    real_c155 = _unzip(tmp_path, "sqrt_c155")
    out = str(tmp_path / "pred_155.axmodel")

    extrapolate(ref_a, ref_b, 155, out)

    pred_mcode = _mcode(out)
    real_mcode = _mcode(real_c155)
    assert len(pred_mcode) == len(real_mcode)
    # 301-325 is ordinary compiler noise present between ANY two independent builds
    # of the same shape (see e.g. docs/axera-memory-op-generator.md); it is not part
    # of what this tool is trying to reproduce, so it is excluded here.
    mismatches = [
        i
        for i in range(len(pred_mcode))
        if pred_mcode[i] != real_mcode[i] and not (301 <= i < 326)
    ]
    # A real, small, known residual -- not zero. If a future fix to the field-merging
    # logic drives this to zero, tighten this test rather than leaving it loose.
    assert 0 < len(mismatches) <= 4
    # npu_params has no carry-sensitive fields in this cluster and stays exact.
    assert _params(out) == _params(real_c155)
