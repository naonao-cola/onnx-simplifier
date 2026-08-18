"""AMD MIGraphX execution-provider compatibility test.

Verifies that onnxsim's output still compiles and runs on AMD's MIGraphX
execution provider, and that simplification does not change the on-device
result. Uses the pip-installable ``onnxruntime-migraphx`` wheel. Unlike the
Core ML / OpenVINO checks, MIGraphX has no CPU fallback or emulator -- it
needs a real ROCm-capable AMD GPU, so this module is skipped on every
ordinary CI runner and only runs where that hardware is present.

The whole module is skipped when the MIGraphX EP isn't usable (package
missing, or no ROCm device answers), so it is safe to keep in ``tests/``. The
heavier, curated model sweep lives in ``scripts/amd`` and is meant to run on
a self-hosted ROCm runner; this file is the fast, in-tree smoke test for that
same hardware.
"""

import os
import sys

import pytest

# The MIGraphX harness lives under scripts/amd; reuse it rather than duplicate.
_AMD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "amd"
)
if _AMD_DIR not in sys.path:
    sys.path.insert(0, _AMD_DIR)

import migraphx_backend as migraphx  # noqa: E402
import models  # noqa: E402
from worker import check  # noqa: E402

pytestmark = pytest.mark.skipif(
    not migraphx.MIGRAPHX_AVAILABLE,
    reason=f"MIGraphX execution provider unavailable: {migraphx.unavailable_reason()}",
)


@pytest.mark.parametrize("name", models.names())
def test_migraphx_compat_suite(name):
    """Each suite model: simplification must not break or change the MIGraphX result."""
    result = check(name, None)
    # ``unsupported`` (the backend can't handle the graph at all) is acceptable;
    # a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


def test_migraphx_matches_cpu_reference():
    """A directly-built graph: MIGraphX(simplified) matches the ORT CPU reference."""
    from onnxsim import simplify

    model = models.conv_bn_relu()
    simp, _ = simplify(model)
    feeds = migraphx.random_feeds(model, seed=0)

    reference = migraphx.run_with_cpu(model, feeds)
    candidate = migraphx.run_with_migraphx(simp, feeds)

    close, max_diff = migraphx.compare(reference, candidate)
    assert close, (
        f"MIGraphX output diverged from CPU reference: max_abs_diff={max_diff}"
    )
