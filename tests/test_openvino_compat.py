"""Intel OpenVINO execution-provider compatibility test.

Verifies that onnxsim's output still compiles and runs on Intel's OpenVINO
execution provider, and that simplification does not change the on-device
result. Uses the pip-installable ``onnxruntime-openvino`` wheel with its
``CPU`` device target, so it runs on an ordinary x86-64 CI runner with no
Intel-specific hardware.

The whole module is skipped when ``onnxruntime-openvino`` is not installed
(the default test matrix does not pull it in), so it is safe to keep in
``tests/``. The heavier, curated model sweep lives in ``scripts/intel`` and
runs on a schedule; this file is the fast, in-tree smoke test.
"""

import os
import sys

import pytest

# The OpenVINO harness lives under scripts/intel; reuse it rather than duplicate.
_INTEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "intel"
)
if _INTEL_DIR not in sys.path:
    sys.path.insert(0, _INTEL_DIR)
_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import openvino_backend as openvino  # noqa: E402

# fresh(), not a bare `import models`/`from worker import check`: every
# scripts/<vendor>/ directory has its own models.py (and most have their own
# worker.py too), all imported by the same bare name -- see
# scripts/axera/_local_import.py's docstring for why a plain import here can
# silently resolve to a *different* vendor's module in the full test suite.
from _local_import import fresh  # noqa: E402

models = fresh("models", _INTEL_DIR)
check = fresh("worker", _INTEL_DIR).check

pytestmark = pytest.mark.skipif(
    not openvino.OPENVINO_AVAILABLE,
    reason=f"OpenVINO execution provider unavailable: {openvino.unavailable_reason()}",
)


@pytest.mark.parametrize("name", models.names())
def test_openvino_compat_suite(name):
    """Each suite model: simplification must not break or change the OpenVINO result."""
    result = check(name, None)
    # ``unsupported`` (the backend can't handle the graph at all) is acceptable;
    # a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


def test_openvino_matches_cpu_reference():
    """A directly-built graph: OpenVINO(simplified) matches the ORT CPU reference."""
    from onnxsim import simplify

    model = models.conv_bn_relu()
    simp, _ = simplify(model)
    feeds = openvino.random_feeds(model, seed=0)

    reference = openvino.run_with_cpu(model, feeds)
    candidate = openvino.run_with_openvino(simp, feeds)

    close, max_diff = openvino.compare(reference, candidate)
    assert close, (
        f"OpenVINO output diverged from CPU reference: max_abs_diff={max_diff}"
    )
