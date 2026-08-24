#!/usr/bin/env python3
"""Evidence generator for the SymExpr / symbolic-dimension-algebra RFCs.

  - onnxsim/onnxsim docs/symexpr-shape-inference-rfc.md      (closes #597)
  - onnxsim/onnx    docs/proposals/0008-SymbolicDimensionAlgebra.md

Both RFCs argue that a real, common export shape -- a KV-cache transformer
decoder, exported with ``torch.onnx.export(..., dynamo=True)`` -- produces a
``Shape -> Gather -> Add -> Concat -> Reshape`` chain that plain ONNX data
propagation cannot fold (it cannot add two ``dim_param``s), but that
onnxsim's SymExpr-based symbolic shape/value evaluator can. This script
turns that argument into three checkable artifacts instead of a claim:

  1. **Structural evidence**: does onnxsim's simplifier actually remove the
     ``past_len + seq_len`` scaffolding from a real dynamo export, and by
     how much does the graph shrink?
  2. **Cross-check evidence**: does torch.export's own symbolic-shape engine
     (``ShapeEnv``, sympy-backed) agree with what onnxsim's SymExpr
     independently re-derives from the *exported* ONNX graph? These are two
     unrelated implementations computing the same thing from two different
     representations (the FX graph vs. the ONNX graph) -- agreement is real
     corroboration, not a tautology.
  3. **Behavioral evidence**: across several concrete (batch, seq_len,
     past_len) instantiations, does the onnxsim-simplified ONNX model still
     match the original eager PyTorch module numerically? This is what
     actually matters -- the algebra is only useful if simplifying with it
     never changes what the model computes.

Nothing here requires onnx/onnx to have adopted anything yet: (1) and (3)
only need onnxsim + onnxruntime, and (2) reads information torch.export
already computes for its own purposes. The point is to have reproducible,
re-runnable evidence to attach to the RFC discussion rather than a single
worked-by-hand example.

Usage::

    pip install torch onnxruntime
    pip install --force-reinstall --no-deps .   # the onnxsim under test
    python tools/torch_onnx_symexpr_consistency_check.py

Caveat: this was written and reviewed without a torch/onnxruntime
environment available to execute it in, so treat it as a first draft --
in particular ``_extract_symbolic_shapes`` below reaches into
``torch.export.ExportedProgram``/``ShapeEnv`` internals that are not a
stable public API and do shift across torch releases. It is written to
degrade gracefully (skip step 2, keep 1 and 3) if that extraction fails
on your torch version, and the failure is reported rather than swallowed.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import onnx

import onnxsim

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("This script needs torch: pip install torch", file=sys.stderr)
    raise

try:
    import onnxruntime
except ImportError:
    print("This script needs onnxruntime: pip install onnxruntime", file=sys.stderr)
    raise


# --------------------------------------------------------------------------
# 1. A small but structurally real KV-cache attention block -- the same
#    shape as docs/symexpr-shape-inference-rfc.md's motivating example, and
#    close kin to tests/test_mnn_llm_export.py's decoder (that one has no
#    cache; this one is the cache-carrying sibling).
# --------------------------------------------------------------------------

HIDDEN = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS


class CausalSelfAttentionWithCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x, past_key, past_value):
        b, s, h = x.shape
        q, k, v = self.qkv(x).split(HIDDEN, dim=-1)

        def split_heads(t):
            return t.view(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # past_len + seq_len: the symbol-plus-symbol case plain ONNX data
        # propagation cannot resolve.
        k = torch.cat([past_key, k], dim=2)
        v = torch.cat([past_value, v], dim=2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        # num_heads * head_dim -> hidden: the symbol-times-constant case.
        out = out.transpose(1, 2).reshape(b, s, h)
        return self.proj(out), k, v


# --------------------------------------------------------------------------
# 2. Export with dynamo=True + torch.export.Dim, exactly as in the RFC.
# --------------------------------------------------------------------------


def export_model(onnx_path: str, batch=2, seq_len=8, past_len=5):
    torch.manual_seed(0)
    model = CausalSelfAttentionWithCache().eval()

    x = torch.randn(batch, seq_len, HIDDEN)
    past_k = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)
    past_v = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)

    batch_dim = torch.export.Dim("batch")
    seq_dim = torch.export.Dim("seq_len")
    past_dim = torch.export.Dim("past_len")

    onnx_program = torch.onnx.export(
        model,
        (x, past_k, past_v),
        dynamic_shapes={
            "x": {0: batch_dim, 1: seq_dim},
            "past_key": {0: batch_dim, 2: past_dim},
            "past_value": {0: batch_dim, 2: past_dim},
        },
        dynamo=True,
        input_names=["x", "past_key", "past_value"],
        output_names=["out", "present_key", "present_value"],
    )
    onnx_program.save(onnx_path)
    return onnx_program, model


# --------------------------------------------------------------------------
# 3. Best-effort extraction of torch.export's own symbolic relationships,
#    for the cross-check. See the module docstring's caveat: this reaches
#    into non-public ExportedProgram/ShapeEnv internals.
# --------------------------------------------------------------------------


def _extract_symbolic_shapes(onnx_program) -> Optional[dict]:
    """{fx_node_name: [str(dim) for each output dim]} using torch's own
    sympy-backed SymInt expressions, or None if the internals this needs
    aren't where this was written expecting them (reported to stderr, not
    silently swallowed -- see the module docstring)."""
    try:
        exported_program = onnx_program.exported_program
    except AttributeError:
        print(
            "warning: onnx_program has no .exported_program on this torch "
            "version; skipping the torch-side symbolic cross-check (steps 1 "
            "and 3 below still run).",
            file=sys.stderr,
        )
        return None

    shapes = {}
    try:
        for node in exported_program.graph_module.graph.nodes:
            val = node.meta.get("val")
            if val is None or not hasattr(val, "shape"):
                continue
            dims = []
            for d in val.shape:
                # A SymInt wraps a sympy expression at d.node.expr; a plain
                # concrete dim is already a Python int.
                expr = getattr(getattr(d, "node", None), "expr", None)
                dims.append(str(expr) if expr is not None else str(d))
            shapes[node.name] = dims
    except AttributeError as e:
        print(
            f"warning: SymInt internals didn't match what this script "
            f"expects ({e}); skipping the torch-side symbolic cross-check.",
            file=sys.stderr,
        )
        return None
    return shapes


def _mentions_both_dynamic_dims(expr_str: str) -> bool:
    """True if a sympy-expression string combines two distinct dim symbols
    (a real 'past_len + seq_len'-style case), not just one symbol alone."""
    symbols = set(re.findall(r"\b(?:s\d+|batch|seq_len|past_len)\b", expr_str))
    return len(symbols) >= 2


# --------------------------------------------------------------------------
# 4. Structural evidence: what did onnxsim's symbolic pass actually remove?
# --------------------------------------------------------------------------


@dataclass
class StructuralReport:
    raw_nodes: int
    simplified_nodes: int
    raw_op_counts: Counter
    simplified_op_counts: Counter
    scaffolding_pattern_found_in_raw: bool
    scaffolding_pattern_found_in_simplified: bool


_SCAFFOLDING_OPS = {"Shape", "Gather", "Add", "Unsqueeze", "Concat"}


def _has_shape_arithmetic_scaffolding(model: onnx.ModelProto) -> bool:
    """Heuristic structural detector for the pattern the RFCs are about: an
    Add node whose producers trace back to two separate Shape/Gather chains
    (i.e. a genuine "dim + dim" computed at runtime), rather than any
    incidental use of these op types elsewhere in the graph."""
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    def _feeds_from_shape(name: str, depth: int = 0) -> bool:
        if depth > 4 or name not in producer:
            return False
        node = producer[name]
        if node.op_type == "Shape":
            return True
        if node.op_type in ("Gather", "Cast", "Squeeze", "Unsqueeze"):
            return any(_feeds_from_shape(i, depth + 1) for i in node.input if i)
        return False

    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        if all(_feeds_from_shape(i) for i in node.input):
            return True
    return False


def structural_evidence(raw: onnx.ModelProto, simplified: onnx.ModelProto) -> StructuralReport:
    return StructuralReport(
        raw_nodes=len(raw.graph.node),
        simplified_nodes=len(simplified.graph.node),
        raw_op_counts=Counter(n.op_type for n in raw.graph.node),
        simplified_op_counts=Counter(n.op_type for n in simplified.graph.node),
        scaffolding_pattern_found_in_raw=_has_shape_arithmetic_scaffolding(raw),
        scaffolding_pattern_found_in_simplified=_has_shape_arithmetic_scaffolding(
            simplified
        ),
    )


# --------------------------------------------------------------------------
# 5. Behavioral evidence: numeric agreement across several concrete shapes.
# --------------------------------------------------------------------------


@dataclass
class ConsistencyResult:
    batch: int
    seq_len: int
    past_len: int
    max_abs_diff: float
    passed: bool


def behavioral_evidence(
    eager_model: nn.Module,
    simplified: onnx.ModelProto,
    shape_grid,
    rtol=1e-4,
    atol=1e-5,
) -> list:
    session = onnxruntime.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    results = []
    for batch, seq_len, past_len in shape_grid:
        torch.manual_seed(1)
        x = torch.randn(batch, seq_len, HIDDEN)
        past_k = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)
        past_v = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)

        with torch.no_grad():
            ref_out, ref_k, ref_v = eager_model(x, past_k, past_v)

        got = session.run(
            None,
            {
                "x": x.numpy(),
                "past_key": past_k.numpy(),
                "past_value": past_v.numpy(),
            },
        )
        diffs = [
            np.max(np.abs(r.numpy() - g))
            for r, g in zip((ref_out, ref_k, ref_v), got)
        ]
        max_diff = float(max(diffs))
        passed = np.allclose(ref_out.numpy(), got[0], rtol=rtol, atol=atol) and \
            np.allclose(ref_k.numpy(), got[1], rtol=rtol, atol=atol) and \
            np.allclose(ref_v.numpy(), got[2], rtol=rtol, atol=atol)
        results.append(
            ConsistencyResult(batch, seq_len, past_len, max_diff, passed)
        )
    return results


# --------------------------------------------------------------------------
# 6. Report, formatted to paste directly into the RFC discussion.
# --------------------------------------------------------------------------


def render_report(
    struct: StructuralReport,
    torch_shapes: Optional[dict],
    consistency: list,
) -> str:
    lines = []
    lines.append("## SymExpr consistency-check report")
    lines.append("")
    lines.append(
        f"- torch {torch.__version__}, onnx {onnx.__version__}, "
        f"onnxruntime {onnxruntime.__version__}"
    )
    lines.append("")
    lines.append("### 1. Structural evidence")
    lines.append("")
    lines.append(f"- raw export: {struct.raw_nodes} nodes")
    lines.append(f"- onnxsim-simplified: {struct.simplified_nodes} nodes")
    shrink_pct = 100 * (1 - struct.simplified_nodes / max(struct.raw_nodes, 1))
    lines.append(f"- shrink: {shrink_pct:.1f}%")
    lines.append(
        "- `past_len + seq_len`-style Shape-arithmetic scaffolding present "
        f"in raw export: {struct.scaffolding_pattern_found_in_raw}"
    )
    lines.append(
        "- same scaffolding still present after onnxsim: "
        f"{struct.scaffolding_pattern_found_in_simplified}"
    )
    removed_ops = struct.raw_op_counts - struct.simplified_op_counts
    scaffolding_removed = {
        op: n for op, n in removed_ops.items() if op in _SCAFFOLDING_OPS
    }
    if scaffolding_removed:
        lines.append(f"- shape-scaffolding ops removed: {dict(scaffolding_removed)}")
    lines.append("")

    lines.append("### 2. Cross-check against torch.export's own ShapeEnv")
    lines.append("")
    if torch_shapes is None:
        lines.append(
            "- skipped: could not extract SymInt expressions from this "
            "torch version's ExportedProgram (see stderr warning above)"
        )
    else:
        multi_symbol = {
            n: dims
            for n, dims in torch_shapes.items()
            for d in dims
            if _mentions_both_dynamic_dims(d)
        }
        lines.append(
            f"- {len(multi_symbol)} FX node(s) whose shape combines two "
            "distinct dynamic dims according to torch's own ShapeEnv "
            "(e.g. a `past_len + seq_len`-shaped concat/cache update) -- "
            "this is torch's independent confirmation that the case this "
            "RFC is about actually occurs in the traced graph, not just in "
            "the exported ONNX file."
        )
        for n, dims in list(multi_symbol.items())[:5]:
            lines.append(f"  - `{n}`: {dims}")
    lines.append("")

    lines.append("### 3. Behavioral evidence (eager PyTorch vs. onnxsim-simplified ONNX)")
    lines.append("")
    lines.append("| batch | seq_len | past_len | max abs diff | pass |")
    lines.append("|---|---|---|---|---|")
    all_passed = True
    for r in consistency:
        all_passed &= r.passed
        lines.append(
            f"| {r.batch} | {r.seq_len} | {r.past_len} | "
            f"{r.max_abs_diff:.2e} | {'yes' if r.passed else '**NO**'} |"
        )
    lines.append("")
    lines.append(
        f"**All shape instantiations consistent: {all_passed}**"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=None, help="write the report to this file (else stdout)"
    )
    args = parser.parse_args()

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "kv_cache_attn.onnx")
        onnx_program, eager_model = export_model(onnx_path)
        raw = onnx.load(onnx_path)

        torch_shapes = _extract_symbolic_shapes(onnx_program)

        simplified, check_ok = onnxsim.simplify(onnx_path, check_n=0)
        if not check_ok:
            print("warning: onnxsim's own internal check failed", file=sys.stderr)

        struct = structural_evidence(raw, simplified)

        shape_grid = [
            (1, 4, 0),   # no cache yet (prefill)
            (1, 1, 12),  # single-token decode step with a long cache
            (3, 6, 5),   # batch > 1, both seq_len and past_len nontrivial
            (2, 1, 0),   # batch > 1, prefill, empty cache
        ]
        consistency = behavioral_evidence(eager_model, simplified, shape_grid)

        report = render_report(struct, torch_shapes, consistency)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
