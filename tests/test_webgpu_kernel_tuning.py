"""Tests for ``onnxsim.webgpu_kernel_tuning`` -- offline (no real GPU)
structural checks that :func:`generate_kernel_candidates` actually produces
several *different*, independently valid kernel candidates for the same
computation, all sharing identical bindings. Numeric correctness and actual
speed comparison need a real WebGPU device; see
``scripts/convertmodel/test/webgpu_kernel_tuning.test.mjs`` for that half
(Playwright/Chromium), matching the same division of labor
``tests/test_webgpu_tinygrad_codegen.py``'s own module docstring describes
for :mod:`onnxsim.webgpu_tinygrad_codegen`.
"""

import numpy as np
import pytest

pytest.importorskip("tinygrad")

from onnxsim.webgpu_kernel_tuning import generate_kernel_candidates  # noqa: E402


def _conv2d_named_tensors():
    """A real Conv2d with a rich tuning search space -- verified directly
    (see ``onnxsim/webgpu_kernel_tuning.py``'s own docstring) that a small
    ``Conv2d`` like this one already gives ``get_kernel_actions`` dozens of
    genuinely distinct candidates, unlike the minimal 3x3x3 Conv3D fixture
    used elsewhere in this repo (whose tiny spatial extent leaves little
    room to tile/upcast/unroll).
    """
    from tinygrad import Tensor

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((1, 4, 16, 16)).astype(np.float32), device="WEBGPU")
    w = Tensor(rng.standard_normal((4, 4, 3, 3)).astype(np.float32), device="WEBGPU")
    y = x.conv2d(w, padding=1)
    return {"x": x, "w": w, "y": y}


def test_produces_multiple_distinct_candidates():
    named = _conv2d_named_tensors()
    results = generate_kernel_candidates(named, "y", max_candidates=8)

    assert len(results) == 1, "a single Conv2d schedules to exactly one kernel call"
    candidates, intermediates = results[0]

    assert len(candidates.steps) > 1
    assert len(candidates.steps) == len(candidates.applied_opts)
    # tinygrad's own get_kernel_actions(include_0=True) always includes the
    # untuned baseline first.
    assert candidates.applied_opts[0] == "[]"

    wgsl_texts = [step.wgsl for step in candidates.steps]
    assert len(set(wgsl_texts)) == len(wgsl_texts), (
        "every candidate should render distinct WGSL"
    )


def test_every_candidate_shares_identical_bindings():
    """A kernel's tuning options change its loop/tiling structure, never
    which buffers it reads or writes -- see the module's own docstring for
    why this is expected, not incidental.
    """
    named = _conv2d_named_tensors()
    candidates, intermediates = generate_kernel_candidates(
        named, "y", max_candidates=8
    )[0]

    def binding_shape(step):
        return tuple((b.tensor, b.intermediate, b.access) for b in step.bindings)

    shapes = {binding_shape(step) for step in candidates.steps}
    assert len(shapes) == 1, (
        f"expected identical bindings across all candidates, got {shapes}"
    )

    (only_shape,) = shapes
    named_tensor_bindings = {name for name, _, _ in only_shape if name is not None}
    assert named_tensor_bindings == {"x", "w", "y"}


def test_spec_for_builds_a_single_step_spec():
    named = _conv2d_named_tensors()
    candidates, intermediates = generate_kernel_candidates(
        named, "y", max_candidates=4
    )[0]

    spec = candidates.spec_for(0, intermediates)
    assert spec.steps == (candidates.steps[0],)
    assert spec.intermediates == intermediates


def test_max_candidates_caps_how_many_get_rendered():
    named = _conv2d_named_tensors()
    candidates, _ = generate_kernel_candidates(named, "y", max_candidates=3)[0]
    assert len(candidates.steps) == 3


def test_raises_when_nothing_is_scheduled():
    from tinygrad import Tensor

    x = Tensor([1.0, 2.0, 3.0], device="WEBGPU")
    # an output that's just its own input schedules to zero compute
    # kernels (verified directly: schedule_linear() finds no Ops.CALL at
    # all here) -- nothing to tune.
    with pytest.raises(RuntimeError):
        generate_kernel_candidates({"x": x, "y": x}, "y")
