#!/usr/bin/env python3
"""Static TinyEngine coverage check -- no compiler, no device, no PyPI package.

Like ``scripts/edgeai/tidl_backend.py``, this wraps no real compiler: there
is no ``pip install``-able TinyEngine, and the real code generator ingests
TFLite, not ONNX (see ``tinyengine_ops.py``'s docstring). ``TINYENGINE_AVAILABLE``
is always True and this always runs on a plain CPU host with only ``onnx``
installed -- it exists for interface symmetry with the sibling vendor
backends, not because availability can vary.

The one thing genuinely different from every other backend in this repo:
there is no such thing as "partial coverage, rest on a fallback" for
TinyEngine. One unrecognized op fails the *entire* compile (see
``tinyengine_ops.py``'s docstring on ``_handleOperator``'s terminal
``NotImplementedError``) -- so ``coverage()`` returning ``"partial"`` here
means "will not compile at all", not "will run some ops on CPU and some on
an accelerator" the way it does for TIDL/QNN/OpenVINO.
"""

from __future__ import annotations

from typing import List, Set

import onnx
from tinyengine_ops import (
    RESHAPE_SKIP_CAVEAT,
    BlockingOp,
    blocking_op_types,
    blocking_ops,
    has_dynamic_shape,
    skip_op_types,
)

TINYENGINE_AVAILABLE = True


def unavailable_reason() -> None:
    """Always None: this check has no external dependency to be missing."""
    return None


def coverage(model: onnx.ModelProto) -> str:
    """ "full" if no known TinyEngine blocker was found, else "partial".

    "partial" here means "this heuristic believes real conversion + codegen
    would fail outright", not "some ops fall back to a slower path" --
    TinyEngine has no such fallback (see this module's docstring).
    """
    return "partial" if (blocking_ops(model) or dynamic_shape_risks(model)) else "full"


def blockers(model: onnx.ModelProto) -> List[BlockingOp]:
    return blocking_ops(model)


def new_blocking_op_types(orig: onnx.ModelProto, simp: onnx.ModelProto) -> Set[str]:
    """Blocking op types present after simplification but not before.

    An empty result does not mean simplification made the graph fully
    TinyEngine-compatible -- only that it did not *introduce* a new
    known-blocking op type relative to the original graph. onnxsim's own
    folds (BN into Conv, constant shape chains) often *remove* blockers
    this heuristic already flagged before simplification -- see
    ``README.md``'s worked examples.
    """
    return blocking_op_types(simp) - blocking_op_types(orig)


def dynamic_shape_risks(model: onnx.ModelProto) -> List[str]:
    """Human-readable reasons a fixed-size C code generator would reject `model`."""
    if has_dynamic_shape(model):
        return [
            "one or more graph inputs have a symbolic or unranked dimension; "
            "TinyEngine generates fixed-size C buffers and loop bounds ahead "
            "of time and needs every shape known statically"
        ]
    return []


def skip_op_risks(model: onnx.ModelProto) -> List[str]:
    """Human-readable caveats for ops TinyEngine silently drops rather than
    rejects or code-generates -- currently only `Reshape` carries one, see
    `tinyengine_ops.RESHAPE_SKIP_CAVEAT`."""
    return [RESHAPE_SKIP_CAVEAT] if "Reshape" in skip_op_types(model) else []
