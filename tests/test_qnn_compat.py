"""Qualcomm QNN execution-provider compatibility test.

Verifies that onnxsim's output still compiles and runs on the Qualcomm AI Runtime
(QNN) execution provider, and that simplification does not change the on-device
result. Uses the pip-installable ``onnxruntime-qnn`` wheel: on a plain x86-64
CPU host (no Snapdragon device) the bundled HTP backend does the full offline
graph compile and executes in the x86 emulator, so this runs on an ordinary CI
runner.

The whole module is skipped when ``onnxruntime-qnn`` is not installed (the
default test matrix does not pull it in), so it is safe to keep in ``tests/``.
The heavy, curated model sweep lives in ``scripts/qualcomm`` and runs on a
schedule; this file is the fast, in-tree smoke test.
"""

import os
import sys

import numpy as np
import pytest

# The QNN harness lives under scripts/qualcomm; reuse it rather than duplicate.
_QUALCOMM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "qualcomm"
)
if _QUALCOMM_DIR not in sys.path:
    sys.path.insert(0, _QUALCOMM_DIR)

import models  # noqa: E402
import qnn_backend as qnn  # noqa: E402
from worker import check  # noqa: E402

pytestmark = pytest.mark.skipif(
    not qnn.QNN_AVAILABLE,
    reason=f"QNN execution provider unavailable: {qnn.unavailable_reason()}",
)


@pytest.mark.parametrize("name", models.names())
def test_qnn_compat_suite(name):
    """Each suite model: simplification must not break or change the QNN result."""
    result = check(name, None)
    # ``unsupported`` (the backend can't handle the graph at all) is acceptable;
    # a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


def test_qnn_matches_cpu_reference():
    """A directly-built graph: QNN(simplified) matches the ORT CPU reference."""
    from onnxsim import simplify

    model = models.conv_bn_relu()
    simp, _ = simplify(model)
    feeds = qnn.random_feeds(model, seed=0)

    reference = qnn.run_with_cpu(model, feeds)
    candidate = qnn.run_with_qnn(simp, feeds)

    close, max_diff = qnn.compare(reference, candidate)
    assert close, f"QNN output diverged from CPU reference: max_abs_diff={max_diff}"


def test_simplify_is_bit_exact_on_qnn():
    """Simplification should not change what the QNN backend computes."""
    from onnxsim import simplify

    model = models.redundant_transpose()
    simp, _ = simplify(model)
    feeds = qnn.random_feeds(model, seed=1)

    qnn_orig = qnn.run_with_qnn(model, feeds)
    qnn_simp = qnn.run_with_qnn(simp, feeds)

    for a, b in zip(qnn_orig, qnn_simp):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
