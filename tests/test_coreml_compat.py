"""Apple Core ML execution-provider compatibility test.

Verifies that onnxsim's output still compiles and runs on Apple's Core ML
execution provider, and that simplification does not change the on-device
result. Uses the ``CoreMLExecutionProvider`` built into the standard
``onnxruntime`` pip wheel's macOS build.

The whole module is skipped when the Core ML EP is unavailable (any host
other than macOS, or a CPU-only onnxruntime build), so it is safe to keep in
``tests/``. The heavier, curated model sweep lives in ``scripts/apple`` and
runs on a schedule; this file is the fast, in-tree smoke test.
"""

import os
import sys

import pytest

# The Core ML harness lives under scripts/apple; reuse it rather than duplicate.
_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)
_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import coreml_backend as coreml  # noqa: E402

# fresh(), not a bare `import models`/`from worker import check`: every
# scripts/<vendor>/ directory has its own models.py (and most have their own
# worker.py too), all imported by the same bare name -- see
# scripts/axera/_local_import.py's docstring for why a plain import here can
# silently resolve to a *different* vendor's module in the full test suite.
from _local_import import fresh  # noqa: E402

models = fresh("models", _APPLE_DIR)
check = fresh("worker", _APPLE_DIR).check

pytestmark = pytest.mark.skipif(
    not coreml.COREML_AVAILABLE,
    reason=f"Core ML execution provider unavailable: {coreml.unavailable_reason()}",
)


@pytest.mark.parametrize("name", models.names())
def test_coreml_compat_suite(name):
    """Each suite model: simplification must not break or change the Core ML result."""
    result = check(name, None)
    # ``unsupported`` (the backend can't handle the graph at all) is acceptable;
    # a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


def test_coreml_matches_cpu_reference():
    """A directly-built graph: Core ML(simplified) matches the ORT CPU reference."""
    from onnxsim import simplify

    model = models.conv_bn_relu()
    simp, _ = simplify(model)
    feeds = coreml.random_feeds(model, seed=0)

    reference = coreml.run_with_cpu(model, feeds)
    candidate = coreml.run_with_coreml(simp, feeds)

    close, max_diff = coreml.compare(reference, candidate)
    assert close, f"Core ML output diverged from CPU reference: max_abs_diff={max_diff}"
