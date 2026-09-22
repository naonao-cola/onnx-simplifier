"""Regression coverage for the evidence-scoped Relu/Sqrt emitter.

Oracles are real Pulsar2 7.0-lite (AX650, MinMax, Numpy calibration with the
range's endpoints pinned) builds at shapes and ranges the emitter was not derived
from; the emitted model must equal them outside the known 301-325 noise window.
"""

import gzip
import json
import os
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from elementwise_emit import (  # noqa: E402
    _NOISE_END,
    _NOISE_START,
    elementwise_mcode,
    emit_elementwise_axmodel,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")


def _load(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _mcode(model):
    return bytes(
        next(i for i in model.graph.initializer if i.name.endswith("_neu")).raw_data
    )


def _init(model, name):
    return next(i for i in model.graph.initializer if i.name == name)


def _outside_noise(a, b):
    assert len(a) == len(b)
    return [
        i for i in range(len(a)) if a[i] != b[i] and not _NOISE_START <= i < _NOISE_END
    ]


_ORACLES = [
    ("relu", 58, -2.0, 2.0, "relu_1x58_pm2p0_oracle.axmodel.gz"),
    ("relu", 59, -4.4, 4.4, "relu_1x59_pm4p4_oracle.axmodel.gz"),
    ("relu", 55, -0.05, 0.05, "relu_1x55_pm0p05_oracle.axmodel.gz"),
    ("sqrt", 50, 0.05, 2.0, "sqrt_1x50_0p05_2p0_oracle.axmodel.gz"),
    ("sqrt", 60, 0.2, 5.5, "sqrt_1x60_0p2_5p5_oracle.axmodel.gz"),
    ("sqrt", 57, 0.15, 20.0, "sqrt_1x57_0p15_20p0_oracle.axmodel.gz"),
    ("sqrt", 49, 0.01, 9.9, "sqrt_1x49_0p01_9p9_oracle.axmodel.gz"),
]


@pytest.mark.parametrize("op,n,lo,hi,oracle", _ORACLES, ids=[o[4] for o in _ORACLES])
def test_emitted_model_matches_compiler_built_oracle(tmp_path, op, n, lo, hi, oracle):
    built = _load(oracle)
    out = tmp_path / "e.axmodel"

    emit_elementwise_axmodel(op, str(out), n=n, lo=lo, hi=hi)

    emitted = onnx.load(str(out), load_external_data=False)
    assert _outside_noise(_mcode(emitted), _mcode(built)) == []
    assert _init(emitted, "npu_params").raw_data == _init(built, "npu_params").raw_data
    assert _init(emitted, "npu_dyn_params").dims == [0]
    assert [d.dim_value for d in emitted.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        n,
    ]
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == [
        1,
        n,
    ]
    info = next(a for a in emitted.graph.node[0].attribute if a.name == "outputs_info")
    assert json.loads(info.s) == {"y": ["FP32", [1, n]]}


@pytest.mark.parametrize(
    "op,lo,hi,fixture",
    [
        ("relu", -0.9, 0.9, "relu_1x49_pm0p9.axmodel.gz"),
        ("sqrt", 0.1, 0.9, "sqrt_1x49_0p1_0p9.axmodel.gz"),
    ],
)
def test_template_range_and_shape_reproduce_the_template(op, lo, hi, fixture):
    template = _mcode(_load(fixture))
    assert _outside_noise(elementwise_mcode(op, 49, lo, hi), template) == []


@pytest.mark.parametrize("n", [48, 56, 64, 8, 40, 65, 100, 0, -49])
def test_rejects_n_outside_the_measured_family(n):
    with pytest.raises(ValueError, match="measured family"):
        elementwise_mcode("relu", n, -0.9, 0.9)


@pytest.mark.parametrize("n", [49.0, True, "50"])
def test_rejects_non_integer_n(n):
    with pytest.raises(ValueError, match="integer"):
        elementwise_mcode("sqrt", n, 0.1, 0.9)


@pytest.mark.parametrize(
    "op,lo,hi",
    [
        ("relu", -0.5, 1.5),  # asymmetric: ~365 bytes reflow
        ("relu", 0.0, 1.0),
        ("relu", -1.0, 0.25),
        ("relu", -0.77, 0.77),  # symmetric but reflows in the measured builds
        ("relu", -0.333, 0.333),
        ("sqrt", 0.0, 1.0),  # hi == 1.0 is another program form
        ("sqrt", 0.1, 1.0),
        ("sqrt", 0.7, 0.8),  # lo/hi > 0.5: output scale 1 ulp off
        ("sqrt", -0.1, 0.9),
    ],
)
def test_refuses_ranges_outside_the_measured_set(op, lo, hi):
    with pytest.raises(ValueError, match="unmeasured"):
        elementwise_mcode(op, 50, lo, hi)


def test_rejects_unknown_op():
    with pytest.raises(ValueError, match="unsupported op"):
        elementwise_mcode("mul", 50, -0.9, 0.9)


def test_allow_unmeasured_only_skips_the_range_check():
    elementwise_mcode("relu", 51, -0.77, 0.77, allow_unmeasured=True)
    with pytest.raises(ValueError, match="measured family"):
        elementwise_mcode("relu", 64, -0.77, 0.77, allow_unmeasured=True)


def test_known_reflow_case_is_really_a_mismatch():
    # Guards the refusal list: the compiler-built +/-0.77 model differs from the
    # template-derived prediction in hundreds of bytes, so the refusal is earned.
    built = _mcode(_load("relu_1x51_pm0p77_reflow.axmodel.gz"))
    predicted = elementwise_mcode("relu", 51, -0.77, 0.77, allow_unmeasured=True)
    assert len(_outside_noise(predicted, built)) > 100


def test_sqrt_at_hi_one_is_a_different_program_length():
    built = _mcode(_load("sqrt_1x63_0_1p0_otherform.axmodel.gz"))
    assert len(built) != len(elementwise_mcode("sqrt", 63, 0.0, 0.9))
