"""Tests for `scripts/renesas/drp_ai_tvm_ops.py` / `drp_ai_tvm_simulator.py`
-- a TVM-frontend-import-only estimate of whether a graph is loadable by
Renesas RZ/V's DRP-AI TVM (see those modules' docstrings for the full scope
and, more importantly, what they explicitly do NOT claim: no real compiler,
hardware, or DRP-AI Translator was ever involved, and nothing here covers
R-Car's Hybrid Compiler at all -- see `scripts/renesas/README.md`).

Every model below is built with `onnx.parser` and checked against
`TVM_V08_ONNX_CONVERT_MAP_OPS`/`DRP_AI_TVM_IMPORTABLE_OPS`, transcribed
verbatim from `apache/tvm`'s `v0.8`-tagged `python/tvm/relay/frontend/
onnx.py` (the exact TVM version `renesas-rz/rzv_drp-ai_tvm` vendors as its
`tvm` git submodule at the time this was written) -- these are regression
tests for the scraper's own output and the simulator's partition logic, not
a claim that DRP-AI TVM still pins this exact TVM version indefinitely.
"""

import os
import sys

import onnx
from onnx import parser

_RENESAS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "renesas"
)
if _RENESAS_DIR not in sys.path:
    sys.path.insert(0, _RENESAS_DIR)

import drp_ai_tvm_ops as ops  # noqa: E402
import drp_ai_tvm_simulator as sim  # noqa: E402


def _model(body, opset=13, ir_version=7):
    model = parser.parse_model(f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """)
    onnx.checker.check_model(model)
    return model


# --- drp_ai_tvm_ops.py: the scraped data itself ---


def test_convert_map_has_known_common_ops():
    # Spot-check against apache/tvm's v0.8 onnx.py `_get_convert_map()`,
    # read directly at the time this test was written.
    assert "Relu" in ops.DRP_AI_TVM_IMPORTABLE_OPS
    assert "Conv" in ops.DRP_AI_TVM_IMPORTABLE_OPS
    assert "MatMul" in ops.DRP_AI_TVM_IMPORTABLE_OPS
    assert "BatchNormalization" in ops.DRP_AI_TVM_IMPORTABLE_OPS
    # An ordinary _get_convert_map() entry, not a special case (see
    # drp_ai_tvm_ops.py's docstring for how an earlier version of this
    # project got that wrong).
    assert "Constant" in ops.DRP_AI_TVM_IMPORTABLE_OPS


def test_convert_map_excludes_ops_not_in_tvm_v08():
    # ONNX ops introduced (or never mapped) after/outside TVM v0.8's own
    # onnx.py -- absence here is a real fact about that source, not a
    # placeholder. If DRP-AI TVM moves to a newer TVM pin, re-run
    # scrape_tvm_onnx_frontend.py and this assertion may need updating.
    assert "LayerNormalization" not in ops.DRP_AI_TVM_IMPORTABLE_OPS
    assert "GridSample" not in ops.DRP_AI_TVM_IMPORTABLE_OPS


def test_importable_ops_data_is_nonempty_and_deduplicated():
    assert len(ops.DRP_AI_TVM_IMPORTABLE_OPS) > 100
    assert len(ops.DRP_AI_TVM_IMPORTABLE_OPS) == len(set(ops.DRP_AI_TVM_IMPORTABLE_OPS))


# --- drp_ai_tvm_simulator.py: partition() / coverage() / would_import_succeed() ---


def test_all_importable_ops_graph_is_full_coverage():
    model = _model("""
        g (float[1,3,8,8] x) => (float[1,3,8,8] y)
        {
          r = Relu(x)
          y = Sigmoid(r)
        }
        """)
    p = sim.partition(model)
    assert p.unimportable_nodes == []
    assert len(p.importable_nodes) == 2
    assert sim.coverage(model) == "full"
    assert sim.would_import_succeed(model) is True
    assert sim.unsupported_op_error_message(model) is None


def test_unimportable_op_breaks_whole_model_not_just_that_node():
    # Unlike Voyager SDK's per-node CPU fallback, TVM v0.8's ONNX frontend
    # raises for the *entire* import -- one unsupported op_type is enough
    # for coverage() to report "partial", same practical outcome (import
    # fails) as if every node were unsupported.
    model = _model(
        """
        g (float[1,4] x, float[4] scale, float[4] bias) => (float[1,4] y)
        {
          r = Relu(x)
          y = LayerNormalization(r, scale, bias)
        }
        """,
        opset=17,
    )
    p = sim.partition(model)
    assert len(p.importable_nodes) == 1
    assert len(p.unimportable_nodes) == 1
    assert p.unimportable_op_types == {"LayerNormalization": 1}
    assert sim.coverage(model) == "partial"
    assert sim.would_import_succeed(model) is False


def test_coverage_none_when_only_op_is_unimportable():
    model = _model(
        """
        g (float[1,4] x, float[4] scale, float[4] bias) => (float[1,4] y)
        { y = LayerNormalization(x, scale, bias) }
        """,
        opset=17,
    )
    assert sim.coverage(model) == "none"
    assert sim.partition(model).importable_node_fraction == 0.0


def test_unsupported_op_error_message_matches_tvm_wording():
    model = _model(
        """
        g (float[1,4] x, float[4] scale, float[4] bias) => (float[1,4] y)
        { y = LayerNormalization(x, scale, bias) }
        """,
        opset=17,
    )
    msg = sim.unsupported_op_error_message(model)
    assert (
        msg
        == "The following operators are not supported for frontend ONNX: LayerNormalization"
    )


def test_partition_labels_nodes_by_name_or_op_type_fallback():
    # onnx.parser's `<name=...>` syntax sets a node *attribute*, not
    # NodeProto.name -- set it programmatically instead.
    model = _model("""
        g (float[1,4] x) => (float[1,4] y)
        { y = Relu(x) }
        """)
    model.graph.node[0].name = "my_relu"
    p = sim.partition(model)
    assert p.importable_nodes == ["my_relu"]

    unnamed = _model("""
        g (float[1,4] x) => (float[1,4] y)
        { y = Relu(x) }
        """)
    assert sim.partition(unnamed).importable_nodes == ["<Relu>"]
