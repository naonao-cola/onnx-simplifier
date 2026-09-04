import dataclasses
from typing import Dict, List, Tuple

import onnx
from rich import print
from rich.table import Table
from rich.text import Text

from onnxsim.model_info import human_readable_size

__all__ = [
    "MemoryPlan",
    "plan_activation_memory",
    "print_memory_plan",
]


@dataclasses.dataclass(frozen=True)
class MemoryPlan:
    """A static byte-offset allocation plan for a model's activation tensors,
    computed by :func:`plan_activation_memory`.

    Every *planned* tensor (a graph input, node output, or graph output with a
    concretely known size) gets a ``(offset, size)`` pair in
    :attr:`tensor_offsets`, all within one shared arena of
    :attr:`arena_bytes`: two tensors only ever get overlapping offset ranges
    when their liveness intervals do not overlap, so a deployment target can
    allocate a single :attr:`arena_bytes`-sized buffer and hand each tensor
    its own slice, reused over the tensor's lifetime, instead of one
    permanent buffer per activation (:attr:`naive_bytes` -- the baseline
    :attr:`compression_ratio` is measured against).

    Weights (initializers) are never part of the plan: they stay resident for
    the whole graph, outside this arena. A tensor whose size cannot be
    resolved to a concrete byte count -- unknown shape/dtype, or a dynamic
    dimension -- is excluded from both :attr:`tensor_offsets` and
    :attr:`naive_bytes` and listed in :attr:`unplanned` instead of guessed, so
    a plan with a non-empty :attr:`unplanned` is a partial lower bound, not a
    complete plan.

    Only the top-level graph is planned -- tensors inside control-flow
    (``If``/``Loop``/``Scan``) subgraph bodies are not visited at all.
    """

    arena_bytes: int
    naive_bytes: int
    tensor_offsets: Dict[str, Tuple[int, int]]
    unplanned: List[str]

    @property
    def compression_ratio(self) -> float:
        """Fraction of :attr:`naive_bytes` the plan saves:
        ``1 - arena_bytes / naive_bytes``. ``0.0`` when there is nothing
        plannable (``naive_bytes == 0``).
        """
        if self.naive_bytes == 0:
            return 0.0
        return 1.0 - self.arena_bytes / self.naive_bytes


def plan_activation_memory(
    model: onnx.ModelProto, run_shape_inference: bool = True
) -> MemoryPlan:
    """Compute a :class:`MemoryPlan` for ``model``'s top-level graph.

    Delegates the allocation itself to the C++ core
    (``onnxsim/memory_planning.h``): a greedy best-fit placement over each
    tensor's liveness interval (from production to last use, the same
    liveness convention :class:`onnxsim.model_info.ModelInfo`'s
    ``memory_footprint`` uses), largest tensor first. ``model`` is not
    modified. Pass ``run_shape_inference=False`` when ``model`` already
    carries populated shapes (e.g. inferred with data propagation) to skip
    the extra pass.
    """
    from onnxsim import onnxsim_cpp2py_export as _C

    offsets, arena_bytes, naive_bytes, unplanned = _C._memory_plan(
        model.SerializeToString(), run_shape_inference
    )
    return MemoryPlan(
        arena_bytes=arena_bytes,
        naive_bytes=naive_bytes,
        tensor_offsets=dict(offsets),
        unplanned=list(unplanned),
    )


def print_memory_plan(plan: MemoryPlan, limit: int = 50) -> None:
    """Pretty-print ``plan`` as a table of tensor offsets (ordered by offset,
    then name), followed by the arena/naive totals and compression ratio, and
    -- when non-empty -- the list of tensors :func:`plan_activation_memory`
    could not place. Capped at ``limit`` rows (with a "... and N more" line)
    so a large model's plan stays readable.
    """
    table = Table(title="Activation Memory Plan")
    table.add_column("Tensor")
    table.add_column("Offset")
    table.add_column("Size")

    ordered = sorted(plan.tensor_offsets.items(), key=lambda kv: (kv[1][0], kv[0]))
    for name, (offset, size) in ordered[:limit]:
        table.add_row(name, human_readable_size(offset), human_readable_size(size))
    print(table)
    if len(ordered) > limit:
        print(Text(f"... and {len(ordered) - limit} more", style="dim"))

    print(
        f"Arena: {human_readable_size(plan.arena_bytes)} "
        f"(naive: {human_readable_size(plan.naive_bytes)}, "
        f"{plan.compression_ratio * 100:.1f}% smaller)"
    )
    if plan.unplanned:
        shown = ", ".join(plan.unplanned[:10])
        more = ", ..." if len(plan.unplanned) > 10 else ""
        print(
            Text(
                f"{len(plan.unplanned)} tensor(s) could not be planned "
                f"(unknown shape/dtype or a dynamic dimension): {shown}{more}",
                style="yellow",
            )
        )
