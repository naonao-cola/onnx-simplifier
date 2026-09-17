"""A TVM-frontend-import-only estimate of whether an ONNX graph is even
loadable by Renesas RZ/V's DRP-AI TVM -- no compiler, no hardware, no
DRP-AI Translator involved. See `drp_ai_tvm_ops.py`'s module docstring for
what this data is (and, importantly, is not) sourced from.

Unlike `scripts/axelera/voyager_simulator.py`'s `partition()`, which
classifies each node independently (Voyager SDK's compiler falls back
per-node to host CPU for an undocumented op_type), TVM v0.8's ONNX frontend
does not partition at all: `GraphProto.from_onnx()` raises
`tvm.error.OpNotImplemented` for the *whole* import the moment any node's
op_type isn't in its convert map. So the meaningful question for a DRP-AI
TVM graph isn't "what fraction of nodes are eligible" -- it's binary: would
import succeed at all? `would_import_succeed()` and `coverage()` below
answer exactly that; `partition()` is kept for symmetry with the Axelera/
Axera simulators (and because knowing *how many* offending nodes/op_types
there are is still useful for a diagnostic message), but its per-node split
does not correspond to a CPU/accelerator split the way it does for Voyager.

**Nothing here says anything about DRP-AI hardware acceleration.** An
op_type this module calls "importable" only means TVM's frontend can turn
it into a Relay op; whether DRP-AI TVM's own BYOC pass then actually
accelerates that op on the DRP-AI engine, vs. running it on CPU, is decided
by the (non-public) DRP-AI Translator Manual and is entirely out of scope
here. Read `coverage() == "full"` as "onnxsim's output wouldn't immediately
break DRP-AI TVM's ONNX import step", not as "this graph runs well on
DRP-AI hardware".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import onnx
from drp_ai_tvm_ops import DRP_AI_TVM_IMPORTABLE_OPS


@dataclass
class Partition:
    importable_nodes: List[str]
    unimportable_nodes: List[str]
    unimportable_op_types: Dict[str, int] = field(default_factory=dict)

    @property
    def importable_node_fraction(self) -> float:
        total = len(self.importable_nodes) + len(self.unimportable_nodes)
        return len(self.importable_nodes) / total if total else 1.0


def partition(model: onnx.ModelProto) -> Partition:
    """Op-type-only classification against
    `drp_ai_tvm_ops.DRP_AI_TVM_IMPORTABLE_OPS`. See this module's docstring
    for why, unlike Voyager SDK, this split is *not* an NPU-vs-CPU-fallback
    partition -- TVM v0.8's ONNX frontend has no such per-node fallback; any
    node in `unimportable_nodes` means the whole model fails to import.
    """
    importable: List[str] = []
    unimportable: List[str] = []
    unimportable_types: Dict[str, int] = {}
    for node in model.graph.node:
        label = node.name or f"<{node.op_type}>"
        if node.op_type in DRP_AI_TVM_IMPORTABLE_OPS:
            importable.append(label)
        else:
            unimportable.append(label)
            unimportable_types[node.op_type] = (
                unimportable_types.get(node.op_type, 0) + 1
            )
    return Partition(importable, unimportable, unimportable_types)


def would_import_succeed(model: onnx.ModelProto) -> bool:
    """True iff every node's op_type is in TVM v0.8's ONNX convert map (or
    is "Constant") -- i.e. `tvm.relay.frontend.from_onnx()` would not raise
    `OpNotImplemented` for this graph, per `drp_ai_tvm_ops.py`'s scraped
    data. Says nothing about whether import would otherwise succeed (a
    converter can still raise for an unsupported attribute/shape on an
    op_type that *is* in the map -- see `drp_ai_tvm_ops.py`'s docstring),
    nor about DRP-AI hardware acceleration.
    """
    return not partition(model).unimportable_nodes


def coverage(model: onnx.ModelProto) -> str:
    """'full' (every node's op_type is in TVM v0.8's ONNX convert map),
    'none', or 'partial' -- by op-type membership alone. Only 'full' implies
    anything about whether DRP-AI TVM's import step would succeed (see
    `would_import_succeed()`); 'partial' and 'none' both mean import would
    raise `OpNotImplemented`, same as if no ops at all were importable --
    unlike Voyager SDK, there is no partial-CPU-fallback outcome here.
    """
    p = partition(model)
    if not p.unimportable_nodes:
        return "full"
    if not p.importable_nodes:
        return "none"
    return "partial"


def unsupported_op_error_message(model: onnx.ModelProto) -> str | None:
    """Reproduces the exact `tvm.error.OpNotImplemented` message TVM v0.8's
    `GraphProto.from_onnx()` would raise for this graph, or `None` if import
    would not raise (`would_import_succeed(model)` is True). Byte-for-byte
    matches the real source (`onnx.py`: `"The following operators are not
    supported for frontend ONNX: " + ", ".join(unsupported_ops)`), including
    that `unsupported_ops` is a `set` there too -- so, like the real TVM
    code, op ordering in the message is not deterministic across Python
    versions/runs for more than one unsupported op_type.
    """
    unimportable_types = set(partition(model).unimportable_op_types)
    if not unimportable_types:
        return None
    return "The following operators are not supported for frontend ONNX: " + ", ".join(
        unimportable_types
    )
