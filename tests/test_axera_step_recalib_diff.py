"""Regression coverage for the compiled-step recalibration classifier.

Fixtures are four Pulsar2 7.0-lite builds of one real toy training step
(``toy-npu2``: 118 nodes, two MatMuls, Softmax/Log distillation loss, Adam),
identical except for their calibration data -- see
``docs/axera-step-recalibrate.md``.
"""

import os
import struct
import sys

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
from step_recalib_diff import (  # noqa: E402
    NOISE_TAIL,
    classify,
    explain_literals,
    load_scales,
    load_step,
    noise_bytes,
)

_FIX = os.path.join(_AXERA_DIR, "fixtures", "step_recalib")


def _model(name):
    return os.path.join(_FIX, f"{name}.axmodel.gz")


def _scales(name):
    return load_scales(os.path.join(_FIX, f"{name}.scales.json"))


def _classify(a, b):
    return classify(_model(a), _model(b), _scales(a), _scales(b))


def test_rebuild_noise_is_confined_to_segment0_tail():
    ma, _ = load_step(_model("toyf_A1"))
    mb, _ = load_step(_model("toyf_A2"))
    a, b = next(iter(ma.values())), next(iter(mb.values()))
    _, segs = mcode.segments(a)
    first, last = segs[0][0], segs[-1][0] + segs[-1][1]
    noise = noise_bytes(a)
    assert len(noise) == NOISE_TAIL and noise.stop == segs[0][0] + segs[0][1]
    in_segments = [i for i in range(first, last) if a[i] != b[i]]
    assert in_segments and all(i in noise for i in in_segments)


def test_same_calibration_rebuilds_are_equivalent():
    result = _classify("toyf_A1", "toyf_A2")
    assert result["verdict"] == "equivalent"
    assert result["changed_segments"] == []
    assert result["npu_params_runs"] == []


def test_adam_moment_recalibration_changes_only_the_teng_segment():
    result = _classify("toyf_A1", "toyf_Dmom100")
    assert [seg for _, seg in result["changed_segments"]] == [2]
    assert result["npu_params_runs"] == []
    teng = result["segments"][("subgraph_npu_0_b1_neu", 2)]
    assert teng["size"] == (17216, 17216)
    assert teng["records"] == (4506, 4501)
    # Every changed float literal is a function of the build's own scales...
    assert teng["explained"] == (27, 27)
    # ...but value-dependent re-encodings still change the record structure.
    assert len(teng["structural"]) == 5
    assert result["verdict"] == "not-patchable"


def test_input_range_recalibration_resizes_teng_and_touches_matmul_params():
    result = _classify("toyf_A1", "toyf_Bx2")
    assert [seg for _, seg in result["changed_segments"]] == [2]
    teng = result["segments"][("subgraph_npu_0_b1_neu", 2)]
    assert teng["size"] == (17216, 17344)
    assert teng["explained"] == (140, 181)
    assert len(result["npu_params_runs"]) == 7
    assert result["verdict"] == "not-patchable"


def test_explain_literals_matches_scale_functions_only():
    scales_a, scales_b = [0.5, 0.25], [0.5, 0.125]

    def f(x):
        return struct.pack("<f", x)

    literals = [
        (0, f(1 / 0.25), f(1 / 0.125)),  # reciprocal of a scale
        (1, f(0.5 / 0.25), f(0.5 / 0.125)),  # ratio of two scales
        (2, f(3.0), f(7.0)),  # neither
    ]
    assert explain_literals(literals, scales_a, scales_b) == (2, 3)
