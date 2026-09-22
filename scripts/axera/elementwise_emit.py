"""Evidence-scoped emitters for standalone AX650 ``Relu`` and ``Sqrt`` models.

Pulsar2 7.0-lite compiles a float32 ``Relu`` or ``Sqrt`` on ``x[1, n]`` to one
``neu mode`` node whose MCode differs across shapes and calibration ranges in a
small, explicit set of places (see ``docs/axera-elementwise-coverage.md``):

* five size immediates (low bytes of ``4n-1`` twice, ``n-1``, ``4n`` twice),
  which are the only n-dependent bytes for ``49 <= n <= 63``, ``n % 8 != 0``;
* float32 scale words that depend on the calibration range: ``1/x_scale``
  (four copies) and ``x_scale`` (four copies) for Relu, plus the output scale
  ``r_scale = sqrt(hi)/255`` (four copies) for Sqrt.

Everything else in the stream is identical across those shapes and ranges once
the known 301-325 compiler-noise window is ignored, so a compiled model at one
shape and range (the committed fixtures) can be retargeted to another shape in
that family and to another calibration range inside the measured set. Ranges
that are known to leave that set (asymmetric Relu ranges, two symmetric ones that
reflow for an unexplained reason, Sqrt at ``hi == 1.0`` or ``lo/hi > 0.5``) and
every other ``n`` are refused rather than guessed. A range nobody built may
still reflow: the check can only refuse what was measured.

Measured, not general: standalone ``x[1, n]``, float32, one calibration style
(range endpoints present in the data). See the coverage document for the audit of
what the emitters can and cannot reach, and for the byte-exact and device results.
"""

from __future__ import annotations

import gzip
import json
import os
import struct

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOISE_START = 301
_NOISE_END = 326

# Retargetable region, measured on n in 49..63 (n % 8 != 0): the streams are
# byte-identical there apart from the size immediates and the scale families.
_N_MIN = 49
_N_MAX = 63
_TEMPLATE_N = 49

_SPECS = {
    "relu": {
        "fixture": "relu_1x49_pm0p9.axmodel.gz",
        "template_range": (-0.9, 0.9),
        # size immediates: (offset, kind)
        "size_offsets": (878, 1000, 1412, 1776, 1936),
        # (first offset, stride, copies) per float family
        "families": {"recip_x": (1069, 8, 4), "x_scale": (1368, 7, 4)},
    },
    "sqrt": {
        "fixture": "sqrt_1x49_0p1_0p9.axmodel.gz",
        "template_range": (0.1, 0.9),
        "size_offsets": (878, 1000, 1393, 1776, 1936),
        "families": {
            "recip_x": (1068, 8, 4),
            "x_scale": (1180, 7, 4),
            "r_scale": (1224, 8, 4),
        },
    },
}


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _round_f32(value: float) -> float:
    return struct.unpack("<f", _f32(value))[0]


def _x_scale(lo: float, hi: float) -> float:
    """Input scale: MinMax over the range widened to include 0, in float32 steps."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi <= lo:
        raise ValueError("calibration range must be non-empty")
    return _round_f32(_round_f32(hi - lo) / _round_f32(255.0))


def _family_words(op: str, lo: float, hi: float) -> dict[str, bytes]:
    """Float32 words that move with the calibration range (fit on 8 builds each)."""
    scale = _x_scale(lo, hi)
    words = {"x_scale": _f32(scale), "recip_x": _f32(_round_f32(1.0 / scale))}
    if op == "sqrt":
        root = _round_f32(max(hi, 0.0) ** 0.5)
        words["r_scale"] = _f32(_round_f32(root / _round_f32(255.0)))
    return words


# Symmetric Relu half-ranges whose Pulsar2 build reflows ~365 bytes even though
# the range is symmetric (each rebuilt three times: deterministic). Neighbours
# such as 0.76 and 0.78 do not reflow, so the trigger is unexplained.
_RELU_REFLOW_HALF_RANGES = (0.77, 0.333)


def _measured_range_problem(op: str, lo: float, hi: float) -> str | None:
    """Why a calibration range is outside what was measured, or None if it is inside.

    Relu: 20 of 22 symmetric ranges built (half-ranges 0.05 to 12) matched the
    template outside the named fields; asymmetric ranges reflow (~365 bytes).
    Sqrt: 15 builds matched with 0 <= lo and lo/hi <= 0.5, hi != 1.0; at
    ``[0.7, 0.8]`` the output scale is 1 ulp off ``f32(sqrt(hi))/255``, and at
    ``hi == 1.0`` (where ``r_scale == x_scale``) the program has another form.
    """
    if op == "relu":
        if not lo < 0 < hi or lo != -hi:
            return "Relu range must be symmetric about 0 (asymmetric ranges reflow ~365 bytes)"
        if hi in _RELU_REFLOW_HALF_RANGES:
            return (
                f"half-range {hi} reflows in the measured builds (trigger unexplained)"
            )
        return None
    if lo < 0 or hi <= 0:
        return "Sqrt calibration range must satisfy 0 <= lo < hi"
    if hi == 1.0:
        return "hi == 1.0 gives a different program form (length 1920), not decoded"
    if lo / hi > 0.5:
        return "lo/hi > 0.5 was off by one ulp in the output scale, not decoded"
    return None


def _mcode_of(model: onnx.ModelProto) -> onnx.TensorProto:
    matches = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *_neu initializer, found {len(matches)}"
        )
    return matches[0]


def _load_template(op: str) -> onnx.ModelProto:
    with gzip.open(os.path.join(_HERE, "fixtures", _SPECS[op]["fixture"]), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _check_supported(n: object) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n must be an integer")
    if not _N_MIN <= n <= _N_MAX or n % 8 == 0:
        raise ValueError(
            f"n={n} is outside the measured family: {_N_MIN} <= n <= {_N_MAX}, n % 8 != 0"
        )
    return n


def elementwise_mcode(
    op: str, n: int, lo: float, hi: float, *, allow_unmeasured: bool = False
) -> bytes:
    """MCode for ``op`` (``relu`` or ``sqrt``) on ``x[1, n]`` calibrated on ``[lo, hi]``.

    ``n`` must be in ``[49, 63]`` and not a multiple of 8. The calibration range
    must be inside what was measured (see ``_measured_range_problem``);
    ``allow_unmeasured=True`` skips only that range check, for experiments, and the
    result is then a guess. The output keeps the template's bytes inside the
    301-325 noise window.
    """
    if op not in _SPECS:
        raise ValueError(f"unsupported op {op!r}; measured: {sorted(_SPECS)}")
    _check_supported(n)
    spec = _SPECS[op]
    template = _load_template(op)
    t_lo, t_hi = spec["template_range"]
    problem = _measured_range_problem(op, lo, hi)
    if problem and not allow_unmeasured:
        raise ValueError(f"range [{lo}, {hi}] for {op} is unmeasured: {problem}")
    out = bytearray(_mcode_of(template).raw_data)

    old = _family_words(op, t_lo, t_hi)
    new = _family_words(op, lo, hi)
    for name, (first, stride, copies) in spec["families"].items():
        for k in range(copies):
            at = first + k * stride
            if bytes(out[at : at + 4]) != old[name]:
                raise ValueError(
                    f"{name} copy {k} at {at} is {bytes(out[at : at + 4]).hex()}, "
                    f"expected the template value {old[name].hex()}"
                )
            out[at : at + 4] = new[name]

    a, b, c, d, e = spec["size_offsets"]
    last = (4 * n - 1) & 255
    for at, value in (
        (a, last),
        (b, (n - 1) & 255),
        (c, last),
        (d, (4 * n) & 255),
        (e, (4 * n) & 255),
    ):
        out[at] = value
    return bytes(out)


def emit_elementwise_axmodel(
    op: str,
    output_path: str,
    *,
    n: int,
    lo: float,
    hi: float,
    allow_unmeasured: bool = False,
) -> str:
    """Write a compiled-model file for ``op`` on ``x[1, n]`` (see ``elementwise_mcode``)."""
    mcode_bytes = elementwise_mcode(op, n, lo, hi, allow_unmeasured=allow_unmeasured)
    model = _load_template(op)
    _mcode_of(model).raw_data = mcode_bytes
    for value_info in [
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ]:
        if value_info.name in ("x", "y"):
            dims = value_info.type.tensor_type.shape.dim
            if len(dims) == 2:
                dims[1].dim_value = n
    node = model.graph.node[0]
    for attr in node.attribute:
        if attr.name == "outputs_info":
            attr.s = json.dumps({"y": ["FP32", [1, n]]}).encode()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
