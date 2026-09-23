"""No-device checks for the ResNet18 step's live-operand MatMul templates
(``scripts/axera/fixtures/matmul_step_templates``): every committed template
must recalibrate onto its held-out native build, and back, record for record
and ``npu_params`` byte for byte; the manifest must map each step node onto
its template positionally."""

import os
import sys

import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import matmul_record_emit as mre  # noqa: E402
import tinygrad_ax_backend as tb  # noqa: E402

MANIFEST = mre.step_manifest()
TEMPLATES = sorted(MANIFEST["templates"])


def _pair(meta, which):
    d = mre.STEP_TEMPLATE_DIR
    if which == "template":
        return os.path.join(d, meta["axmodel"]), os.path.join(d, meta["quant"])
    return os.path.join(d, meta["heldout"]), os.path.join(d, meta["heldout_quant"])


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_template_recalibrates_onto_heldout_build(name, direction):
    meta = MANIFEST["templates"][name]
    src, dst = (
        ("template", "heldout") if direction == "forward" else ("heldout", "template")
    )
    (sm, sq), (dm, dq) = _pair(meta, src), _pair(meta, dst)
    old, new = mre.load_scales(sq), mre.load_scales(dq)
    assert old != new
    out, report = mre.recalibrate(mre.load_model(sm), old, new)
    assert report["records"] == meta["records"]
    assert report["params_words"] == meta["params_words"]
    got = mre.compare(out, mre.load_model(dm))
    assert got["structure"]
    assert got["record_diffs"] == []
    assert got["params_diff_bytes"] == 0


def test_every_node_maps_onto_its_template():
    for node, entry in MANIFEST["nodes"].items():
        meta = MANIFEST["templates"][entry["template"]]
        assert len(entry["names"]) == len(meta["names"]), node
        assert len(set(entry["names"])) == len(entry["names"]), node


def test_step_node_scales_rekeys_and_refuses_gaps():
    node = MANIFEST["templates"]["dX_MatMul_54"]["step_node"]
    entry = mre.step_template(node)
    tq = mre.load_scales(entry["quant"])
    back = {v: k for k, v in entry["names"].items()}
    step_scales = {back[t]: v for t, v in tq.items() if t in back}
    assert mre.step_node_scales(entry, tq, step_scales) == tq
    first = next(iter(step_scales))
    del step_scales[first]
    with pytest.raises(mre.CalibrationError, match="no step scale"):
        mre.step_node_scales(entry, tq, step_scales)


def test_unknown_node_is_refused():
    with pytest.raises(ValueError, match="no validated live-operand template"):
        mre.step_template("MatMul_not_in_step")


def test_backend_plans_live_operand_nodes_from_the_manifest():
    for node in MANIFEST["nodes"]:
        status, detail = tb.plan_node({"op": "MatMul", "name": node, "attrs": {}})
        assert status == "conditional", (node, detail)
        assert MANIFEST["nodes"][node]["template"] in detail
    status, _ = tb.plan_node({"op": "MatMul", "name": "not_a_step_node", "attrs": {}})
    assert status == "refused"
