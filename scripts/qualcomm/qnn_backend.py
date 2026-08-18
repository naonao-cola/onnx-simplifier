#!/usr/bin/env python3
"""Thin wrapper around the ONNX Runtime QNN execution provider.

The Qualcomm AI Runtime (QNN) execution provider ships in the ``onnxruntime-qnn``
PyPI wheel. On an ordinary x86-64 Linux host (no Snapdragon device) the bundled
**HTP** backend still does the full *offline graph preparation* — the same
compile that a Qualcomm converter would run — via ``libHtpPrepare.so``, and then
executes the compiled graph in the **x86 host emulator**. That gives us two
signals with nothing but ``pip install onnxruntime-qnn`` and a CPU runner:

* **compatibility** — does the (simplified) graph compile onto QNN at all, and
  how much of it maps to the accelerator vs. falling back to ONNX Runtime's CPU
  provider;
* **numerics** — do the emulated QNN outputs match the ONNX Runtime CPU
  reference, i.e. did onnxsim preserve semantics along the QNN path.

The wheel does **not** ship the QNN *CPU reference* backend (``libQnnCpu.so``);
that only comes with the full QAIRT SDK. Point ``QNN_BACKEND_PATH`` at a
``libQnnCpu.so`` from an SDK install to use the true reference backend instead
of HTP emulation; everything else here is unchanged.

This module degrades gracefully: on a host where the QNN EP cannot be loaded
(e.g. no ``onnxruntime-qnn``, or an unsupported platform) ``QNN_AVAILABLE`` is
False and ``unavailable_reason()`` explains why, so callers can skip rather than
error.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, List, Optional

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

# HTP graph-finalization optimization level. "3" is the most aggressive offline
# optimization; it makes the compile a little slower but exercises the same path
# a shipped model would take.
_HTP_OPT_MODE = os.environ.get("QNN_HTP_OPT_MODE", "3")

_EP_NAME = "QNNExecutionProvider"

QNN_AVAILABLE = False
_UNAVAILABLE_REASON: Optional[str] = None
_REGISTERED = False

try:
    import onnxruntime as ort
    import onnxruntime_qnn as _oqnn

    # Backend selection: default to the wheel-bundled HTP backend (offline
    # prepare + x86 emulation); allow an override to e.g. a QAIRT libQnnCpu.so.
    _BACKEND_PATH = os.environ.get("QNN_BACKEND_PATH") or _oqnn.get_qnn_htp_path()
    if not os.path.exists(_BACKEND_PATH):
        raise FileNotFoundError(f"QNN backend library not found: {_BACKEND_PATH}")
    QNN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> Optional[str]:
    """Why the QNN EP is unusable on this host, or None if it is usable."""
    return _UNAVAILABLE_REASON


def backend_path() -> str:
    """Absolute path to the QNN backend library in use (HTP by default)."""
    return _BACKEND_PATH


def ensure_registered() -> None:
    """Register the QNN plugin EP with ONNX Runtime (idempotent).

    ORT 1.24+ loads the QNN provider as a plugin: the library is registered by
    name, then selected per session via the OrtEpDevice API. Registering twice
    raises, so guard it.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    ort.register_execution_provider_library(_EP_NAME, _oqnn.get_library_path())
    _REGISTERED = True


def _qnn_devices() -> list:
    ensure_registered()
    devices = [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]
    if not devices:
        raise RuntimeError("QNN EP registered but no QNN OrtEpDevice was exposed")
    return devices


@contextlib.contextmanager
def silence_native_output():
    """Silence C-level stdout/stderr (the HTP compiler is very chatty).

    The QNN HTP backend prints its compile stages straight to the process's file
    descriptors, below Python's ``sys.stdout``. Redirect fds 1/2 to /dev/null
    for the duration so aggregated logs stay readable. Only wrap the ORT calls,
    never the code that emits real results.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (devnull, *saved):
            os.close(fd)


def _session_options(strict: bool):
    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR
    if strict:
        # Fail session creation if any node cannot be placed on QNN, instead of
        # silently falling back to ORT's CPU provider. Used to measure coverage.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return so


def run_with_qnn(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    strict: bool = False,
) -> List[np.ndarray]:
    """Compile ``model`` with the QNN EP and run it, returning the outputs.

    With ``strict=True`` the session is built with CPU fallback disabled, so
    creation raises unless the *entire* graph maps onto QNN.
    """
    devices = _qnn_devices()
    so = _session_options(strict)
    so.add_provider_for_devices(
        devices,
        {
            "backend_path": _BACKEND_PATH,
            "htp_graph_finalization_optimization_mode": _HTP_OPT_MODE,
        },
    )
    with silence_native_output():
        sess = ort.InferenceSession(model.SerializeToString(), sess_options=so)
        return sess.run(None, feeds)


def run_with_cpu(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run ``model`` on ONNX Runtime's CPU provider (the numerical reference)."""
    so = _session_options(strict=False)
    with silence_native_output():
        sess = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        return sess.run(None, feeds)


def coverage(model: onnx.ModelProto) -> str:
    """ "full" if the whole graph maps onto QNN, else "partial".

    Determined by trying to build a session with CPU fallback disabled: success
    means every node was accepted by the QNN EP.
    """
    try:
        devices = _qnn_devices()
        so = _session_options(strict=True)
        so.add_provider_for_devices(
            devices,
            {
                "backend_path": _BACKEND_PATH,
                "htp_graph_finalization_optimization_mode": _HTP_OPT_MODE,
            },
        )
        with silence_native_output():
            ort.InferenceSession(model.SerializeToString(), sess_options=so)
        return "full"
    except Exception:
        return "partial"
