"""Plot the "node reduction per loop" curve from an onnxsim profiling trace.

``simplify(..., profile=...)`` (see ``onnx_simplifier.py``) writes a Chrome
Trace Event Format JSON. Alongside its span flame graph, that trace carries a
series of "NodeCount" counter events (``ph: "C"``) -- one right after every
round of each simplification fixed-point loop, tagged in ``args.loop`` with
which loop produced it:

* ``"Initial"``     -- the node count before any round has run (one sample).
* ``"Optimize"``    -- each round of the inner (shape inference + onnx-optimizer
  passes) fixed point (``OptAndShapeOnGraph`` in onnxsim.cpp).
* ``"FoldConstant"`` -- each round of the outer (optimize + constant-fold)
  fixed point (``Pipeline`` in onnxsim.cpp).
* ``"Rewrite"``     -- each round of a user-supplied ``custom_rewriter``, when
  one was passed to ``simplify()``.

This module turns that raw data into a plot: one subplot per loop, node count
against round index within that loop, so how quickly (and whether) each loop
converges to a fixed point is visible at a glance -- a flat tail means the
loop stopped changing the graph before ``ONNXSIM_FIXED_POINT_ITERS`` (default
50) rounds were exhausted; a curve still descending at the last round means it
hit that cap without converging.
"""

from typing import Dict, List, Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe; the caller decides how to view the PNG.
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is an optional dependency; see plot_node_reduction.
    plt = None

import json

__all__ = ["load_node_counts", "plot_node_reduction"]

# Preserve the order these loops naturally run in during a simplify() call,
# for a stable, readable subplot order. Any other tag (e.g. a future loop)
# is appended after these, in first-seen order.
_LOOP_ORDER = ["Initial", "Optimize", "FoldConstant", "Rewrite"]


def load_node_counts(trace_path: str) -> Dict[str, List[int]]:
    """Read the "NodeCount" counter events out of a ``simplify(..., profile=...)`` trace.

    :param trace_path: path to the JSON trace.
    :returns: a dict mapping loop name (see the module docstring) to the list
            of node counts recorded for it, one per round, in the order the
            rounds ran (the trace's events are written in start-timestamp
            order by ``Profiler::Finish()``).
    """
    with open(trace_path) as f:
        trace = json.load(f)
    node_counts: Dict[str, List[int]] = {}
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "C" or e.get("name") != "NodeCount":
            continue
        args = e.get("args", {})
        count = args.get("node_count")
        if count is None:
            continue
        loop = args.get("loop", "?")
        node_counts.setdefault(loop, []).append(int(count))
    return node_counts


def plot_node_reduction(trace_path: str, out_path: Optional[str] = None) -> str:
    """Plot node count per round, one subplot per simplification fixed-point loop.

    Loops that never ran (e.g. ``"Rewrite"`` when no ``custom_rewriter`` was
    passed to ``simplify()``, or ``"FoldConstant"`` when ``skip_constant_folding``
    was set) are simply absent from the trace and skipped here.

    :param trace_path: path to a JSON trace written by ``simplify(..., profile=...)``.
    :param out_path: where to save the figure (a PNG, regardless of the
            extension given). When ``None``, derived from ``trace_path`` by
            appending ``_node_reduction.png``.
    :returns: the path the figure was saved to.
    :raises RuntimeError: if matplotlib is not installed, or the trace has no
            "NodeCount" events (i.e. it was not written by a profiled
            ``simplify()`` call -- ``ONNXSIM_PROFILE``/``profile=`` must be set).
    """
    if plt is None:
        raise RuntimeError(
            "plot_node_reduction() needs matplotlib; install it with "
            "`pip install onnxsim[plot]` (or `pip install matplotlib`)."
        )
    node_counts = load_node_counts(trace_path)
    if not node_counts:
        raise RuntimeError(
            f"{trace_path!r} has no \"NodeCount\" events -- was it written by "
            "a profiled simplify(..., profile=...) call?"
        )
    if out_path is None:
        out_path = f"{trace_path}_node_reduction.png"

    loops = [loop for loop in _LOOP_ORDER if loop in node_counts]
    loops += [loop for loop in node_counts if loop not in loops]

    fig, axes = plt.subplots(
        len(loops), 1, figsize=(7.0, 2.6 * len(loops)), squeeze=False
    )
    for ax, loop in zip(axes[:, 0], loops):
        counts = node_counts[loop]
        rounds = list(range(len(counts)))
        ax.plot(rounds, counts, marker="o", markersize=4, linewidth=1.5)
        converged = len(counts) >= 2 and counts[-1] == counts[-2]
        status = "" if len(counts) < 2 else " (converged)" if converged else " (hit round cap)"
        ax.set_title(f"{loop}: {len(counts)} round(s){status}")
        ax.set_xlabel("round")
        ax.set_ylabel("node count")
        ax.grid(True, alpha=0.3)
        if counts:
            ax.annotate(
                str(counts[-1]),
                xy=(rounds[-1], counts[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    fig.suptitle("onnxsim: node count per round, per fixed-point loop")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
