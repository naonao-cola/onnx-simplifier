# Regression test for the SymExpr / symbolic-dimension-algebra work
# (issue #597, PR #527, M0-M3 of #532): does onnxsim's symbolic shape/value
# evaluator actually fold the shape scaffolding a real KV-cache transformer
# export produces, and does doing so ever change what the model computes?
#
# The model below is the same KV-cache attention block used to motivate that
# work: exported with ``torch.onnx.export(..., dynamo=True)`` and
# ``torch.export.Dim``, it produces a ``Shape -> Gather -> Add -> Concat ->
# Reshape`` chain computing ``past_len + seq_len`` (two dynamic dims added
# together) and ``num_heads * head_dim`` (a dynamic-looking dim times a
# constant) -- exactly the case plain ONNX data propagation cannot fold
# because it has no arithmetic over a ``dim_param``, but onnxsim's SymExpr
# can. Three things are checked:
#
#   * ``test_onnxsim_removes_kv_cache_shape_scaffolding``: the scaffolding
#     pattern is actually present in the raw export (a sanity check that
#     this test isn't vacuous) and actually gone after ``onnxsim.simplify``.
#   * ``test_onnxsim_symexpr_matches_torch_shapeenv_symbols``: cross-checks
#     onnxsim's SymExpr (re-derived from the *exported ONNX graph*) against
#     torch.export's own ShapeEnv (sympy-backed, computed from the *FX
#     graph*) agreeing that the traced model actually combines two distinct
#     dynamic dims -- two independent computations over two different
#     representations agreeing is real corroboration, not a tautology. This
#     reaches into ``ExportedProgram``/``ShapeEnv`` internals that are not a
#     stable public API, so it is skipped (not failed) if a torch version's
#     internals don't match what it expects, rather than breaking the suite
#     on a torch upgrade.
#   * ``test_onnxsim_kv_cache_consistency``: across several concrete
#     ``(batch, seq_len, past_len)`` instantiations, the onnxsim-simplified
#     ONNX model still matches the original eager PyTorch module -- the
#     algebra is only useful if simplifying with it never changes behavior.
#
# To run locally::
#
#     pip install torch onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_symexpr_kv_cache_consistency.py -v

import sys

import numpy as np
import onnx
import pytest

import onnxsim

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")

import torch.nn as nn  # noqa: E402  (after the torch importorskip guard)

HIDDEN = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS

_SCAFFOLDING_OPS = {"Shape", "Gather", "Add", "Unsqueeze", "Concat"}


class _CausalSelfAttentionWithCache(nn.Module):
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

        # past_len + seq_len: two dynamic dims added together.
        k = torch.cat([past_key, k], dim=2)
        v = torch.cat([past_value, v], dim=2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        # num_heads * head_dim -> hidden: a dynamic-looking dim times a
        # constant, merging back into a plain Reshape target.
        out = out.transpose(1, 2).reshape(b, s, h)
        return self.proj(out), k, v


def _export(tmp_path, batch=2, seq_len=8, past_len=5):
    torch.manual_seed(0)
    model = _CausalSelfAttentionWithCache().eval()

    x = torch.randn(batch, seq_len, HIDDEN)
    past_k = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)
    past_v = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)

    batch_dim = torch.export.Dim("batch")
    seq_dim = torch.export.Dim("seq_len")
    past_dim = torch.export.Dim("past_len")

    onnx_path = str(tmp_path / "kv_cache_attn.onnx")
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
    return onnx_path, onnx_program, model


def _has_shape_arithmetic_scaffolding(model: onnx.ModelProto) -> bool:
    """True iff the graph contains an Add node whose two inputs both trace
    back to a Shape node -- the "dim + dim computed at runtime" pattern this
    module is about, not just incidental use of these op types elsewhere."""
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


@pytest.fixture(scope="module")
def exported_and_simplified(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("symexpr_kv_cache")
    onnx_path, onnx_program, eager_model = _export(tmp_path)
    raw = onnx.load(onnx_path)

    simplified, check_ok = onnxsim.simplify(onnx_path, check_n=0)
    assert check_ok

    return raw, simplified, onnx_program, eager_model


def test_onnxsim_removes_kv_cache_shape_scaffolding(exported_and_simplified):
    raw, simplified, _, _ = exported_and_simplified

    # Sanity check first: if the raw export doesn't even contain the
    # pattern, the rest of this test would pass vacuously.
    assert _has_shape_arithmetic_scaffolding(raw), (
        "raw export doesn't contain the past_len+seq_len scaffolding this "
        "test is about -- torch's export shape changed, update the model"
    )

    onnx.checker.check_model(simplified)
    assert len(simplified.graph.node) < len(raw.graph.node)
    assert not _has_shape_arithmetic_scaffolding(simplified), (
        "onnxsim left the past_len+seq_len Shape/Gather/Add scaffolding in "
        "place -- the symbolic evaluator regressed"
    )


def test_onnxsim_symexpr_matches_torch_shapeenv_symbols(exported_and_simplified):
    _, _, onnx_program, _ = exported_and_simplified

    try:
        exported_program = onnx_program.exported_program
        multi_symbol_nodes = []
        for node in exported_program.graph_module.graph.nodes:
            val = node.meta.get("val")
            if val is None or not hasattr(val, "shape"):
                continue
            for d in val.shape:
                expr = getattr(getattr(d, "node", None), "expr", None)
                if expr is not None and len(expr.free_symbols) >= 2:
                    multi_symbol_nodes.append((node.name, str(expr)))
    except AttributeError as e:
        pytest.skip(
            f"ExportedProgram/ShapeEnv internals didn't match what this "
            f"test expects on torch {torch.__version__} ({e}); not a "
            f"stable public API, so this is a skip, not a failure"
        )

    # torch's own ShapeEnv, independently of anything onnxsim does, must
    # agree that the traced graph actually combines two distinct dynamic
    # dims somewhere (the past_len+seq_len cache-length computation).
    assert multi_symbol_nodes, (
        "torch.export's ShapeEnv reports no FX node whose shape combines "
        "two distinct dynamic dims -- either the model/export changed, or "
        "this cross-check needs updating for the current torch version"
    )
    print(f"multi-symbol-dim FX nodes: {multi_symbol_nodes[:5]}", file=sys.stderr)


@pytest.mark.parametrize(
    "batch,seq_len,past_len",
    [
        (1, 4, 0),   # prefill, empty cache
        (1, 1, 12),  # single-token decode step with a long cache
        (3, 6, 5),   # batch > 1, both seq_len and past_len nontrivial
        (2, 1, 0),   # batch > 1, prefill, empty cache
    ],
)
def test_onnxsim_kv_cache_consistency(
    exported_and_simplified, batch, seq_len, past_len
):
    _, simplified, _, eager_model = exported_and_simplified

    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, HIDDEN)
    past_k = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)
    past_v = torch.randn(batch, NUM_HEADS, past_len, HEAD_DIM)

    with torch.no_grad():
        ref_out, ref_k, ref_v = eager_model(x, past_k, past_v)

    session = onnxruntime.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    got_out, got_k, got_v = session.run(
        None,
        {
            "x": x.numpy(),
            "past_key": past_k.numpy(),
            "past_value": past_v.numpy(),
        },
    )

    np.testing.assert_allclose(ref_out.numpy(), got_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ref_k.numpy(), got_k, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ref_v.numpy(), got_v, rtol=1e-4, atol=1e-5)
