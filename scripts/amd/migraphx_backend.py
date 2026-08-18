#!/usr/bin/env python3
"""Thin wrapper around the ONNX Runtime MIGraphX execution provider.

AMD's [MIGraphX](https://github.com/ROCm/AMDMIGraphX) graph compiler ships as
the `MIGraphXExecutionProvider` in the separate
[`onnxruntime-migraphx`](https://pypi.org/project/onnxruntime-migraphx/) wheel
(it replaces, rather than supplements, plain `onnxruntime`, the same relationship
`onnxruntime-openvino` has). Unlike Core ML or OpenVINO's CPU device target,
MIGraphX has **no CPU fallback or host emulator**: it compiles and executes
directly on a ROCm-capable AMD GPU. There is no way to exercise it on an
ordinary CPU-only CI runner, so this harness needs real AMD GPU + ROCm
hardware -- see `scripts/amd/README.md` for how to run it.

Because the provider's shared library is bundled into the wheel regardless of
whether a ROCm device is actually present, `MIGraphXExecutionProvider`
showing up in ``onnxruntime.get_available_providers()`` is not by itself proof
the EP is usable (unlike the Core ML / OpenVINO checks, which stop there).
This module additionally tries to build a session for a trivial graph on the
EP, so ``MIGRAPHX_AVAILABLE`` only ends up True when a ROCm device actually
answers.

This module degrades gracefully: whenever the EP can't be loaded or no ROCm
device responds, ``MIGRAPHX_AVAILABLE`` is False and ``unavailable_reason()``
explains why, so callers can skip rather than error.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from common.ep_numerics import compare, random_feeds  # noqa: E402,F401

_EP_NAME = "MIGraphXExecutionProvider"
# fp16 halves compile time and matches how most deployed MIGraphX models run;
# override to "0" to probe/compile in fp32 instead.
_USE_FP16 = os.environ.get("MIGRAPHX_FP16", "1") not in ("0", "", "false", "False")

MIGRAPHX_AVAILABLE = False
_UNAVAILABLE_REASON: str | None = None


def _probe_model() -> onnx.ModelProto:
    """A single-node Identity graph, just enough to force a real device probe."""
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "migraphx_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


try:
    import onnxruntime as ort

    if _EP_NAME not in ort.get_available_providers():
        raise RuntimeError(
            "MIGraphXExecutionProvider not available "
            "(install onnxruntime-migraphx, not plain onnxruntime)"
        )
    # Presence in get_available_providers() only means the wheel bundles the
    # provider's shared library, not that a ROCm device answered -- unlike
    # Core ML/OpenVINO, MIGraphX has no CPU path to fall back to, so an actual
    # session build is the only real availability signal.
    _probe_so = ort.SessionOptions()
    _probe_so.log_severity_level = 3  # ERROR
    ort.InferenceSession(
        _probe_model().SerializeToString(),
        sess_options=_probe_so,
        providers=[_EP_NAME],
    )
    MIGRAPHX_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> str | None:
    """Why the MIGraphX EP is unusable on this host, or None if it is usable."""
    return _UNAVAILABLE_REASON


def _session_options(strict: bool):
    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR
    if strict:
        # Fail session creation if any node cannot be placed on MIGraphX,
        # instead of silently falling back to ORT's CPU provider. Used to
        # measure coverage.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return so


def _migraphx_provider_options() -> Dict[str, str]:
    return {"migraphx_fp16_enable": "1"} if _USE_FP16 else {}


def run_with_migraphx(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    strict: bool = False,
) -> List[np.ndarray]:
    """Run ``model`` on the MIGraphX EP, returning the outputs.

    With ``strict=True`` the session is built with CPU fallback disabled, so
    creation raises unless the *entire* graph maps onto MIGraphX.
    """
    so = _session_options(strict)
    providers = [(_EP_NAME, _migraphx_provider_options())]
    if not strict:
        providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=providers
    )
    return sess.run(None, feeds)


def run_with_cpu(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run ``model`` on ONNX Runtime's plain CPU provider (the numerical reference)."""
    so = _session_options(strict=False)
    sess = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    return sess.run(None, feeds)


def coverage(model: onnx.ModelProto) -> str:
    """ "full" if the whole graph maps onto MIGraphX, else "partial".

    Determined by trying to build a session with CPU fallback disabled:
    success means every node was accepted by the MIGraphX EP.
    """
    try:
        run_with_migraphx(model, random_feeds(model), strict=True)
        return "full"
    except Exception:
        return "partial"
