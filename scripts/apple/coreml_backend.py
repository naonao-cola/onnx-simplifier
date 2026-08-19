#!/usr/bin/env python3
"""Thin wrapper around the ONNX Runtime CoreML execution provider.

Apple's Core ML EP ships in the standard ``onnxruntime`` PyPI wheel, but only
the macOS build actually links it in -- ``CoreMLExecutionProvider`` is absent
from ``get_available_providers()`` on Linux/Windows. On an ordinary macOS CI
runner (no extra pip package, no special hardware beyond the Mac itself) this
gives two signals:

* **compatibility** -- does the (simplified) graph compile onto Core ML at
  all, and how much of it maps onto the accelerator vs. falling back to ONNX
  Runtime's CPU provider;
* **numerics** -- does the Core ML result match the ONNX Runtime CPU
  reference, i.e. did onnxsim preserve semantics along the Core ML path.

Core ML itself picks the actual compute unit (CPU/GPU/Neural Engine) at
runtime; ``MLComputeUnits=ALL`` (the default here) lets it use whichever it
judges best, same as a real deployment would.

This module degrades gracefully: on a host where the Core ML EP is not
present (not macOS, or an onnxruntime build without it) ``COREML_AVAILABLE``
is False and ``unavailable_reason()`` explains why, so callers can skip
rather than error.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import onnx

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only keep scripts/ on sys.path for the duration of this import: scripts/
# also holds directories like rfdetr/ with no __init__.py, which Python 3
# treats as importable namespace packages. Leaving scripts/ on sys.path for
# the rest of the process would make `import rfdetr` "succeed" as that empty
# namespace package instead of skipping via pytest.importorskip, and shadow
# the real one everywhere else it's checked for.
_inserted = _SCRIPTS_DIR not in sys.path
if _inserted:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from common.ep_numerics import compare, random_feeds  # noqa: E402,F401
finally:
    if _inserted:
        sys.path.remove(_SCRIPTS_DIR)

_EP_NAME = "CoreMLExecutionProvider"
# ALL lets Core ML place ops on CPU/GPU/Neural Engine as it judges best, the
# same choice a real deployment would leave it to make.
_COMPUTE_UNITS = os.environ.get("COREML_COMPUTE_UNITS", "ALL")

COREML_AVAILABLE = False
_UNAVAILABLE_REASON: str | None = None

try:
    import onnxruntime as ort

    if _EP_NAME not in ort.get_available_providers():
        raise RuntimeError(
            "CoreMLExecutionProvider not built into this onnxruntime "
            "(expected on non-macOS, or a CPU-only wheel)"
        )
    COREML_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> str | None:
    """Why the Core ML EP is unusable on this host, or None if it is usable."""
    return _UNAVAILABLE_REASON


def _session_options(strict: bool):
    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR
    if strict:
        # Fail session creation if any node cannot be placed on Core ML,
        # instead of silently falling back to ORT's CPU provider. Used to
        # measure coverage.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return so


def _coreml_provider_options() -> Dict[str, str]:
    return {"MLComputeUnits": _COMPUTE_UNITS}


def run_with_coreml(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    strict: bool = False,
) -> List[np.ndarray]:
    """Run ``model`` on the Core ML EP, returning the outputs.

    With ``strict=True`` the session is built with CPU fallback disabled, so
    creation raises unless the *entire* graph maps onto Core ML.
    """
    so = _session_options(strict)
    providers = [(_EP_NAME, _coreml_provider_options())]
    if not strict:
        providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=providers
    )
    return sess.run(None, feeds)


def run_with_cpu(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run ``model`` on ONNX Runtime's CPU provider (the numerical reference)."""
    so = _session_options(strict=False)
    sess = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    return sess.run(None, feeds)


def coverage(model: onnx.ModelProto) -> str:
    """ "full" if the whole graph maps onto Core ML, else "partial".

    Determined by trying to build a session with CPU fallback disabled:
    success means every node was accepted by the Core ML EP.
    """
    try:
        run_with_coreml(model, random_feeds(model), strict=True)
        return "full"
    except Exception:
        return "partial"
