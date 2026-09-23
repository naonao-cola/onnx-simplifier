"""tinygrad Hexagon (DSP) / Metal device compatibility test.

Verifies that onnxsim's output still imports and runs through tinygrad's own
``OnnxRunner`` on the ``DSP`` (Qualcomm Hexagon, via ``MOCKDSP=1`` + qemu) and
``METAL`` (Apple GPU) devices, and that simplification does not change the
on-device result -- the tinygrad-device analogue of ``test_coreml_compat.py``
/ ``test_qnn_compat.py``.

Each device is skipped independently when it is unavailable (no
qemu-hexagon/clang for the mock DSP path, non-macOS for Metal, or no tinygrad
installed), so this is safe to keep in ``tests/``. The heavier sweep lives in
``scripts/hexagon``.
"""

import os
import sys

import numpy as np
import pytest

# The harness lives under scripts/hexagon; reuse it rather than duplicate.
_HEXAGON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "hexagon"
)
if _HEXAGON_DIR not in sys.path:
    sys.path.insert(0, _HEXAGON_DIR)
_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import tinygrad_hexagon_backend as backend  # noqa: E402

# fresh(), not a bare `import models`/`from worker import check`: every
# scripts/<vendor>/ directory has its own models.py (and most have their own
# worker.py too), all imported by the same bare name -- see
# scripts/axera/_local_import.py's docstring for why a plain import here can
# silently resolve to a *different* vendor's module in the full test suite.
from _local_import import fresh  # noqa: E402

models = fresh("models", _HEXAGON_DIR)
_worker = fresh("worker", _HEXAGON_DIR)

_DEVICES = (backend.HEXAGON_DEVICE, backend.METAL_DEVICE)


def _skip_reason(device: str) -> str | None:
    ok, reason = backend.probe_device(device)
    return None if ok else reason


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", models.names())
def test_tinygrad_device_compat_suite(device, name):
    """Each suite model: simplification must not break or change the device result."""
    reason = _skip_reason(device)
    if reason is not None:
        pytest.skip(f"tinygrad {device} unavailable: {reason}")
    result = _worker.check(name, None, device)
    # ``unsupported`` (the backend can't handle the graph at all) is acceptable;
    # a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


@pytest.mark.parametrize("device", _DEVICES)
def test_tinygrad_device_matches_reference(device):
    """A directly-built graph: device(simplified) matches the default-device reference."""
    reason = _skip_reason(device)
    if reason is not None:
        pytest.skip(f"tinygrad {device} unavailable: {reason}")
    from onnxsim import simplify

    model = models.conv_bn_relu()
    simp, _ = simplify(model)
    feeds = backend.random_feeds(model, seed=0)

    reference = backend.run_with_device(model, feeds, None)
    candidate = backend.run_with_device(simp, feeds, device)

    close, max_diff = backend.compare(reference, candidate)
    assert close, f"{device} output diverged from reference: max_abs_diff={max_diff}"


@pytest.mark.parametrize("device", _DEVICES)
def test_simplify_is_bit_exact_on_tinygrad_device(device):
    """Removing a redundant Transpose must not change the device result.

    Tight tolerance, not bitwise identity: a fused GPU backend (Metal) may
    fuse the original and simplified graphs' kernels differently, so
    float32 rounding can differ in the last ulp even though the arithmetic
    is the same. The point stands -- onnxsim removed a node and the device
    result is unchanged up to float noise.
    """
    reason = _skip_reason(device)
    if reason is not None:
        pytest.skip(f"tinygrad {device} unavailable: {reason}")
    from onnxsim import simplify

    model = models.redundant_transpose()
    simp, _ = simplify(model)
    feeds = backend.random_feeds(model, seed=1)

    dev_orig = backend.run_with_device(model, feeds, device)
    dev_simp = backend.run_with_device(simp, feeds, device)

    for a, b in zip(dev_orig, dev_simp):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
