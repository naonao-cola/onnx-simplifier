"""Generates several *alternative* WebGPU kernels for the same computation
-- differently tiled/upcast/localized variants of the exact same
:mod:`onnxsim.webgpu_tinygrad_codegen` output -- so a caller can dispatch
every candidate on a **real** WebGPU device (in a real browser) and time
each with :mod:`scripts/convertmodel/webgpu_kernel_dispatcher.mjs`'s own
``profile: true`` option, keeping whichever is actually fastest on that
device. Answers "could we tune a UOp-generated kernel by handing execution
to a real browser" -- see
``scripts/convertmodel/test/webgpu_kernel_tuning.test.mjs`` for the
real-GPU half of this story.

``tinygrad`` is an **optional** dependency, matching the precedent set by
:mod:`onnxsim.webgpu_tinygrad_codegen` (see that module's own docstring):
nothing here runs, or is even imported, unless :func:`generate_kernel_candidates`
is actually called.

## Why this can't just be tinygrad's own BEAM search

tinygrad already has an autotuner: ``BEAM=N`` makes
``tinygrad.codegen.to_program`` call ``tinygrad.codegen.opt.search.beam_search``,
which tries several ``Opt``-tuned variants of a kernel and picks whichever
is fastest -- **by actually compiling and running each one itself**, via
``dev = Device[s.ren.target.device]`` (``tinygrad/codegen/opt/search.py``).
That line is exactly the wall this whole ``webgpu_tinygrad_codegen`` family
of modules exists to route around: it needs a real, natively-loaded
``Device["WEBGPU"]`` (the actual ``dawn``/``wgpu-native`` shared library),
which frequently isn't available wherever kernels are *generated* (a CI
runner, a server, this repo's own dev sandboxes) even though it's exactly
what an end user's real browser always has.

This module reuses only the **device-free half** of tinygrad's own
autotuner -- ``tinygrad.codegen.opt.postrange.Scheduler`` and
``tinygrad.codegen.opt.search.get_kernel_actions``, which enumerate
candidate ``Opt`` combinations and apply them via plain Python schedule
manipulation (``Scheduler.copy()`` + ``.apply_opt()``, no compilation, no
device) -- verified directly: calling ``get_kernel_actions`` on a real
``Conv2d`` kernel's own ``Scheduler`` returns dozens of candidates, each
rendering to genuinely different WGSL via the ordinary offline
``to_program`` path this whole module family already relies on. What
tinygrad's own ``beam_search`` does *next* (compile, run, time, pick) is
this module's caller's job instead, against a real WebGPU device reached
from JS -- see the test file above for exactly that loop.

## Why the same bindings apply to every candidate

A kernel's tuning options (``OptOps.UPCAST``/``LOCAL``/``GROUP``/...) change
its loop/tiling structure -- how many work-items iterate, how much gets
kept in registers/local memory -- never *which* buffers it reads or writes.
The owning ``Ops.CALL``'s own ``src[1:]`` (the buffer list
:func:`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program` already
extracts once, from the *schedule*, upstream of any per-kernel
optimization) is therefore identical for every candidate of one call, so
:func:`generate_kernel_candidates` computes that binding/intermediate
bookkeeping exactly once and reuses
:func:`onnxsim.webgpu_tinygrad_codegen._render_kernel_step` (factored out
of that module for exactly this reuse) to render each candidate's own
WGSL/dispatch against the same bindings.

## Scope

Like :mod:`onnxsim.webgpu_tinygrad_codegen` itself, this only ever produces
*candidates* -- it does not itself dispatch, time, or judge correctness of
any of them (no real device is ever touched from Python here). A model with
more than one scheduled kernel call gets candidates enumerated
independently per call (see :class:`KernelCandidates`); this module does
not attempt to jointly tune across calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from onnxsim.webgpu_kernel_metadata import WebgpuKernelSpec, WebgpuKernelStep
from onnxsim.webgpu_tinygrad_codegen import _render_kernel_step

if TYPE_CHECKING:
    from tinygrad import Tensor

__all__ = ["KernelCandidates", "generate_kernel_candidates"]


def _require_tinygrad():
    try:
        import tinygrad  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "onnxsim.webgpu_kernel_tuning needs the optional 'tinygrad' "
            "package: pip install tinygrad"
        ) from e


@dataclass(frozen=True)
class KernelCandidates:
    """The alternative :class:`~onnxsim.webgpu_kernel_metadata.WebgpuKernelStep`\\ s
    tinygrad's own tuning-option search space produces for **one** scheduled
    kernel call, all sharing the exact same bindings/intermediates (see this
    module's own docstring for why that's safe) -- only ``wgsl``,
    ``entry_point``, and ``dispatch`` vary between them.

    :param steps: one entry per candidate, in the same (arbitrary but
            deterministic, given a fixed tinygrad version) order
            ``tinygrad.codegen.opt.search.get_kernel_actions`` itself
            produces them -- index ``0`` is always the *untuned* baseline
            (``include_0=True``), matching what
            :func:`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program`
            alone would have produced.
    :param applied_opts: each candidate's own ``Scheduler.applied_opts``,
            ``repr()``-ed for logging/debugging (e.g. picking a specific
            candidate back out after a browser reports timings) -- not
            meant to be parsed back into real ``Opt`` objects.
    """

    steps: Tuple[WebgpuKernelStep, ...]
    applied_opts: Tuple[str, ...]

    def spec_for(self, index: int, intermediates: Dict[str, int]) -> WebgpuKernelSpec:
        """A single-step :class:`WebgpuKernelSpec` for candidate ``index``
        -- the shape ``webgpu_kernel_dispatcher.mjs``'s own
        ``dispatchWebgpuProgram`` (and therefore a browser-side tuning
        loop) actually consumes. ``intermediates`` is
        :func:`generate_kernel_candidates`'s own per-call return value.
        """
        return WebgpuKernelSpec(steps=(self.steps[index],), intermediates=intermediates)


def generate_kernel_candidates(
    named_tensors: Dict[str, "Tensor"],
    output_name: str,
    max_candidates: int = 32,
) -> List[Tuple[KernelCandidates, Dict[str, int]]]:
    """The tuning-candidate counterpart to
    :func:`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program`: same
    inputs, same scheduling/buffer-identification, but for each real
    compute kernel tinygrad schedules, returns every candidate
    ``tinygrad.codegen.opt.search.get_kernel_actions`` finds (capped at
    ``max_candidates``) instead of only tinygrad's own default (untuned)
    rendering.

    :param named_tensors: same contract as ``_lower_tensor_program``'s own
            parameter of the same name -- every entry other than
            ``output_name`` must be a leaf ``Tensor`` (built directly via
            ``Tensor(data, device="WEBGPU")``, never derived), so its
            buffer is identifiable by object identity.
    :param output_name: key into ``named_tensors`` for the computation's
            own output.
    :param max_candidates: upper bound on how many of
            ``get_kernel_actions``'s own candidates get rendered per
            scheduled kernel call -- rendering every candidate
            ``to_program`` produces (dozens, for a real ``Conv2d``) is
            wasted work once a caller only has budget to actually dispatch
            a handful of them on a real device.
    :returns: one ``(candidates, intermediates)`` pair per scheduled kernel
            call (almost always exactly one, for the single-node kernels
            this module family targets) -- ``intermediates`` is that call's
            own scratch-buffer byte-length map, needed alongside whichever
            candidate step a caller picks to build a full
            :class:`~onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec` via
            :meth:`KernelCandidates.spec_for`.
    :raises RuntimeError: tinygrad scheduled no compute kernel at all (e.g.
            the whole graph constant-folded away) -- same condition
            ``_lower_tensor_program`` itself raises on.
    """
    _require_tinygrad()

    from tinygrad.codegen.opt.postrange import Scheduler
    from tinygrad.codegen.opt.search import get_kernel_actions
    from tinygrad.helpers import Target
    from tinygrad.renderer.wgsl import WGSLRenderer
    from tinygrad.uop.ops import Ops

    output = named_tensors[output_name]
    linear = output.schedule_linear()
    kernel_calls = [
        u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK
    ]
    if not kernel_calls:
        raise RuntimeError(
            "tinygrad scheduled no compute kernel for this graph -- it may "
            "have constant-folded away entirely"
        )

    # Same buffer-identification approach as _lower_tensor_program's own
    # (private, unexported) helper -- duplicated rather than imported since
    # it's a two-line closure over this function's own locals, not worth a
    # third module-level indirection.
    def _base_buffer(t):
        return next((u for u in t.uop.toposort() if u.op is Ops.BUFFER), None)

    uop_to_name = {}
    for name, t in named_tensors.items():
        buf = _base_buffer(t)
        if buf is None:
            raise RuntimeError(
                f"tensor {name!r} has no underlying BUFFER after scheduling"
            )
        uop_to_name[buf] = name
    renderer = WGSLRenderer(Target())

    results: List[Tuple[KernelCandidates, Dict[str, int]]] = []
    for call in kernel_calls:
        ast = call.src[0]
        buffer_uops = list(call.src[1:])

        intermediate_names: Dict[object, str] = {}
        intermediate_bytes: Dict[str, int] = {}
        next_intermediate = [0]

        def _name_for(buf_uop):
            name = uop_to_name.get(buf_uop)
            if name is not None:
                return "tensor", name
            name = intermediate_names.get(buf_uop)
            if name is None:
                name = f"_intermediate_{next_intermediate[0]}"
                next_intermediate[0] += 1
                intermediate_names[buf_uop] = name
                intermediate_bytes[name] = buf_uop.size()[0] * buf_uop.dtype.itemsize
            return "intermediate", name

        scheduler = Scheduler(ast, renderer)
        actions = get_kernel_actions(scheduler, include_0=True)

        steps = []
        applied_opts = []
        for _, candidate in list(actions.items())[:max_candidates]:
            candidate_ast = candidate.get_optimized_ast()
            steps.append(
                _render_kernel_step(candidate_ast, buffer_uops, _name_for, renderer)
            )
            applied_opts.append(repr(candidate.applied_opts))

        results.append(
            (
                KernelCandidates(steps=tuple(steps), applied_opts=tuple(applied_opts)),
                dict(intermediate_bytes),
            )
        )

    return results
