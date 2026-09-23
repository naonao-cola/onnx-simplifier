#!/usr/bin/env python3
"""Thin wrapper around tinygrad's Hexagon (DSP) and Metal devices.

Feeds onnxsim's (simplified) graphs into ``tinygrad.nn.onnx.OnnxRunner`` with
its tensors placed on a specific tinygrad device -- ``DSP`` (Qualcomm Hexagon,
via tinygrad's ``runtime/ops_dsp.py``) or ``METAL`` (Apple's GPU, via
``runtime/ops_metal.py``) -- giving two signals, the same two the Core ML
(``scripts/apple``) and QNN (``scripts/qualcomm``) harnesses get from their
own backends:

* **compatibility** -- does the (simplified) graph import and run through
  ``OnnxRunner`` on that device at all;
* **numerics** -- does the on-device result match the reference (tinygrad's
  own default device), i.e. did onnxsim preserve semantics along that path.

The check is framed as *original vs. simplified on the same device*, so a
backend limitation cancels out and only an onnxsim-introduced change can fail
the run -- exactly like the Core ML / QNN harnesses.

Two environment notes:

* ``DSP`` needs ``MOCKDSP=1`` (run the generated Hexagon code under
  ``qemu-hexagon`` on the host instead of talking to FastRPC hardware). This
  module applies it with ``setdefault`` -- an explicit real-device setup is
  never overridden -- plus a Hexagon-capable clang and ``qemu-hexagon`` (see
  ``README.md``). Without those, ``HEXAGON_AVAILABLE`` is False.
* ``METAL`` needs macOS (it is absent from tinygrad's device list on
  Linux/Windows). Without it, ``METAL_AVAILABLE`` is False.

This module degrades gracefully: when a device is unusable ``*_AVAILABLE`` is
False and ``unavailable_reason()`` explains why, so callers skip rather than
error.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

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

# Mock-DSP execution (qemu) unless the caller explicitly configured a
# real-device setup already.
os.environ.setdefault("MOCKDSP", "1")

HEXAGON_DEVICE = os.environ.get("TINYGRAD_HEXAGON_DEVICE", "DSP")
METAL_DEVICE = "METAL"

_TINYGRAD_IMPORT_ERROR: Optional[str] = None
try:
    from tinygrad import Device, Tensor  # noqa: E402
    from tinygrad.nn.onnx import OnnxRunner  # noqa: E402
except Exception as exc:  # pragma: no cover - depends on the host
    _TINYGRAD_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    Device = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    OnnxRunner = None  # type: ignore[assignment]


def _canary_model() -> onnx.ModelProto:
    """A trivial Add graph used to probe whether a device really executes."""
    from onnx import helper
    from onnx import TensorProto as TP

    node = helper.make_node("Add", ["x", "y"], ["z"])
    graph = helper.make_graph(
        [node],
        "canary",
        [
            helper.make_tensor_value_info("x", TP.FLOAT, [4]),
            helper.make_tensor_value_info("y", TP.FLOAT, [4]),
        ],
        [helper.make_tensor_value_info("z", TP.FLOAT, [4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def run_with_device(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """Import ``model`` into tinygrad's ``OnnxRunner`` on ``device`` and run it.

    ``device=None`` runs on tinygrad's default device (the numerical
    reference). Any other ``device`` is applied with ``OnnxRunner.to()`` for
    the graph's constants *and* to the feeds: ``OnnxRunner`` parses plain
    numpy feeds onto the default device, which fails mixed-device graphs
    (``all buffers must be on the same device``), so feeds are wrapped as
    ``Tensor(..., device=device)`` first.
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        path = f.name
    try:
        runner = OnnxRunner(path)
        if device is not None:
            runner = runner.to(device)
            tiny_feeds = {
                k: (Tensor(v, device=device) if not isinstance(v, Tensor) else v)
                for k, v in feeds.items()
            }
        else:
            tiny_feeds = dict(feeds)
        out = runner(tiny_feeds)
    finally:
        os.unlink(path)
    return [out[o.name].numpy() for o in model.graph.output]


def run_with_hexagon(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run ``model`` on tinygrad's Hexagon (DSP) device."""
    return run_with_device(model, feeds, HEXAGON_DEVICE)


def run_with_metal(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run ``model`` on tinygrad's Metal device."""
    return run_with_device(model, feeds, METAL_DEVICE)


def run_reference(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run ``model`` on tinygrad's default device (the numerical reference)."""
    return run_with_device(model, feeds, None)


@functools.lru_cache(maxsize=None)
def probe_device(device: str) -> Tuple[bool, str]:
    """Run the canary on ``device``; ``(True, "")`` or ``(False, reason)``.

    Opening the device is not enough: under ``MOCKDSP=1`` the DSP device
    object constructs fine on any host, but the first kernel compile needs a
    Hexagon-capable clang and the run needs ``qemu-hexagon``. The canary
    exercises the whole path, so a success here means ``run_with_device``
    will work too. Cached per process -- one probe kernel per device.
    """
    if _TINYGRAD_IMPORT_ERROR is not None:
        return False, f"tinygrad is not installed ({_TINYGRAD_IMPORT_ERROR})"
    try:
        available = list(Device.get_available_devices())
    except Exception as exc:
        return False, f"could not list tinygrad devices: {exc}"
    if device not in available:
        return False, f"{device} not in tinygrad devices {available}"
    if device == HEXAGON_DEVICE and os.environ.get("MOCKDSP") == "1":
        if shutil.which("qemu-hexagon-static") is None and shutil.which(
            "qemu-hexagon"
        ) is None:
            return False, "MOCKDSP=1 needs qemu-hexagon(-static) on PATH"
    try:
        x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
        y = np.array([0.5, -1.0, 2.0, 8.0], np.float32)
        out = run_with_device(_canary_model(), {"x": x, "y": y}, device)[0]
    except Exception as exc:
        return False, f"canary run on {device} failed: {type(exc).__name__}: {exc}"
    if not np.allclose(out, x + y, rtol=1e-5, atol=1e-6):
        return False, f"canary on {device} computed the wrong result"
    return True, ""


HEXAGON_AVAILABLE, _HEXAGON_REASON = (
    probe_device(HEXAGON_DEVICE) if _TINYGRAD_IMPORT_ERROR is None else (False, "")
)
if _TINYGRAD_IMPORT_ERROR is not None:
    _HEXAGON_REASON = f"tinygrad is not installed ({_TINYGRAD_IMPORT_ERROR})"

METAL_AVAILABLE, _METAL_REASON = (
    probe_device(METAL_DEVICE) if _TINYGRAD_IMPORT_ERROR is None else (False, "")
)
if _TINYGRAD_IMPORT_ERROR is not None:
    _METAL_REASON = f"tinygrad is not installed ({_TINYGRAD_IMPORT_ERROR})"


def unavailable_reason(device: str = HEXAGON_DEVICE) -> Optional[str]:
    """Why ``device`` is unusable here, or None if it is usable."""
    ok, reason = probe_device(device)
    return None if ok else reason


def devices() -> Dict[str, bool]:
    """``{"DSP": ..., "METAL": ...}`` availability snapshot (no probing cost)."""
    return {HEXAGON_DEVICE: HEXAGON_AVAILABLE, METAL_DEVICE: METAL_AVAILABLE}
