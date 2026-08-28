#!/usr/bin/env python3
"""Thin wrapper around the ONNX Runtime OpenVINO execution provider.

Intel's OpenVINO EP does **not** ship in the standard ``onnxruntime`` PyPI
wheel; it comes from the separate
[`onnxruntime-openvino`](https://pypi.org/project/onnxruntime-openvino/)
package (which bundles its own OpenVINO runtime and replaces plain
``onnxruntime`` -- the two must not be installed side by side). Its default
``CPU`` device target runs on any x86-64 host, no Intel-specific hardware or
driver needed, so this runs on a stock CI runner with nothing but
``pip install onnxruntime-openvino``.

This gives two signals:

* **compatibility** -- does the (simplified) graph compile onto OpenVINO at
  all, and how much of it maps onto the backend vs. falling back to ONNX
  Runtime's CPU provider;
* **numerics** -- does the OpenVINO result match the plain ONNX Runtime CPU
  reference, i.e. did onnxsim preserve semantics along the OpenVINO path.

This module degrades gracefully: on a host where the OpenVINO EP is not
present (``onnxruntime-openvino`` not installed) ``OPENVINO_AVAILABLE`` is
False and ``unavailable_reason()`` explains why, so callers can skip rather
than error.
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

_EP_NAME = "OpenVINOExecutionProvider"
# CPU needs no discrete Intel hardware/driver, so it's the device target that
# runs unconditionally on any CI runner. Override for a GPU/NPU-equipped host.
_DEVICE_TYPE = os.environ.get("OPENVINO_DEVICE_TYPE", "CPU")

OPENVINO_AVAILABLE = False
_UNAVAILABLE_REASON: str | None = None

try:
    import onnxruntime as ort

    if _EP_NAME not in ort.get_available_providers():
        raise RuntimeError(
            "OpenVINOExecutionProvider not available "
            "(install onnxruntime-openvino, not plain onnxruntime)"
        )
    OPENVINO_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> str | None:
    """Why the OpenVINO EP is unusable on this host, or None if it is usable."""
    return _UNAVAILABLE_REASON


def _session_options(strict: bool):
    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR
    if strict:
        # Fail session creation if any node cannot be placed on OpenVINO,
        # instead of silently falling back to ORT's CPU provider. Used to
        # measure coverage.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return so


def _openvino_provider_options() -> Dict[str, str]:
    return {"device_type": _DEVICE_TYPE}


def run_with_openvino(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    strict: bool = False,
) -> List[np.ndarray]:
    """Run ``model`` on the OpenVINO EP, returning the outputs.

    With ``strict=True`` the session is built with CPU fallback disabled, so
    creation raises unless the *entire* graph maps onto OpenVINO.
    """
    so = _session_options(strict)
    providers = [(_EP_NAME, _openvino_provider_options())]
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
    """ "full" if the whole graph maps onto OpenVINO, else "partial".

    Determined by trying to build a session with CPU fallback disabled:
    success means every node was accepted by the OpenVINO EP.
    """
    try:
        run_with_openvino(model, random_feeds(model), strict=True)
        return "full"
    except Exception:
        return "partial"
