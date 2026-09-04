#!/usr/bin/env python3
"""A local, Docker-free approximation of Pulsar2's INT8 PTQ, via onnxruntime.

Pulsar2's own quantizer is proprietary: inspecting the real
`output/*/quant/quant_axmodel.onnx` that `pulsar2 build` writes (from the
real `resnet18d_Opset18` conversion -- see `pulsar2_ops.py`'s docstring)
shows it rewrites the graph into its own custom ops in the plain default
domain (`AxQuantizedConv`, `AxQuantizedAdd`, `AxQuantizeLinear`, ...) -- not
standard ONNX `QuantizeLinear`/`DequantizeLinear`, so it cannot be executed
by onnxruntime directly and there is no way to reproduce it bit-for-bit here.

What *is* confirmed from that same real file, read off its
`AxQuantizedConv` node attributes directly:

* activations: **U8** (uint8), a single scale/zero-point per tensor (e.g.
  `input_scales=[0.0187]`, `input_zeropoints=[114]`) -- i.e. per-tensor,
  **asymmetric**.
* weights: **S8** (int8), one scale *per output channel* (e.g.
  `weight_scales` has length 32 for a 32-out-channel conv) -- i.e.
  **per-channel, symmetric** (no zero-point attribute on the weight side).
* `quant_method = 0`, matching the `"calibration_method": "MinMax"` in the
  build config (see `pulsar2_ops.py`'s docstring for the real build log).

This module reproduces exactly that numeric convention --
`onnxruntime.quantization.quantize_static` with `CalibrationMethod.MinMax`,
`activation_type=QUInt8`, `weight_type=QInt8`, `per_channel=True` -- and
outputs a standard QDQ ONNX model onnxruntime can actually run. Treat the
result as **compatible in calibration methodology and numeric precision**,
not a reproduction of Pulsar2's internal IR or exact rounding/fusion
behavior. Always confirm on real hardware before trusting it for a
deployment decision -- see `pulsar2_simulator.py` for how this is used
against real-device output.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Iterable, List, Optional

import numpy as np
import onnx

PULSAR2_QUANTIZER_AVAILABLE = False
_UNAVAILABLE_REASON: Optional[str] = None

try:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    PULSAR2_QUANTIZER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
    CalibrationDataReader = object  # type: ignore[assignment,misc]


def unavailable_reason() -> Optional[str]:
    """Why the quantizer is unusable on this host, or None if it is usable."""
    return _UNAVAILABLE_REASON


if PULSAR2_QUANTIZER_AVAILABLE:

    class _ListDataReader(CalibrationDataReader):
        def __init__(self, feeds_list: List[Dict[str, np.ndarray]]):
            self._iter = iter(feeds_list)

        def get_next(self):
            return next(self._iter, None)


def quantize_like_pulsar2(
    model: onnx.ModelProto,
    calibration_feeds: Iterable[Dict[str, np.ndarray]],
    *,
    per_channel: bool = True,
) -> onnx.ModelProto:
    """Return an INT8 QDQ ONNX model calibrated the way Pulsar2 does (MinMax).

    ``calibration_feeds`` should be several representative input dicts (8-32,
    matching typical PTQ calibration set sizes -- the real build used 16).
    Raises if the quantizer is unavailable; check `PULSAR2_QUANTIZER_AVAILABLE`
    first.
    """
    if not PULSAR2_QUANTIZER_AVAILABLE:
        raise RuntimeError(f"pulsar2_quantizer unavailable: {_UNAVAILABLE_REASON}")

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.onnx")
        out_path = os.path.join(td, "out.onnx")
        onnx.save(model, in_path)
        quantize_static(
            in_path,
            out_path,
            calibration_data_reader=_ListDataReader(list(calibration_feeds)),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=per_channel,
        )
        return onnx.load(out_path)
