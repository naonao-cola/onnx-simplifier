"""Keeps ``docs/graph_grad-ops.md`` honest against ``onnxsim.graph_grad``.

That doc exists because the module it indexes has grown too large to read
top to bottom in one sitting (see the doc's own intro). A hand-maintained
index is exactly the kind of thing that silently drifts -- an op added to
``_RULES``/``_MULTI_OUTPUT_RULES``/``_PYTHON_ONLY_RULES`` and forgotten in
the doc, or the reverse -- so this test parses the "ONNX op" column back out
of the doc's tables and cross-checks it against
:func:`onnxsim.graph_grad.supported_ops`, minus
:data:`onnxsim.graph_grad._CUSTOM_RULES` (dynamic, test-time registrations
that are never this doc's concern). It does not, and cannot, check that the
formulas themselves stay accurate -- that is still on a reviewer -- only
that the op *coverage* the doc claims matches what the module actually
differentiates.
"""

from __future__ import annotations

import re
from pathlib import Path

from onnxsim import graph_grad

_DOC = Path(__file__).parent.parent / "docs" / "graph_grad-ops.md"


def _ops_in_doc() -> set:
    """Every ONNX op named in the doc's ``| `Op` | ...`` table rows.

    Matches the first backtick-quoted, capitalized identifier at the start
    of a table row -- deliberately narrow (rather than every backtick span
    in the file) so a formula that happens to backtick-quote an op name
    (` ``Split``'s adjoint`` in the ``Concat`` row, say) is not
    double-counted as its own row.
    """
    text = _DOC.read_text()
    return set(re.findall(r"^\|\s*`([A-Z]\w*)`\s*\|", text, re.MULTILINE))


def test_doc_covers_exactly_the_ops_graph_grad_supports():
    documented = _ops_in_doc()
    actual = graph_grad.supported_ops() - frozenset(graph_grad._CUSTOM_RULES)

    missing_from_doc = actual - documented
    assert not missing_from_doc, (
        f"{sorted(missing_from_doc)} differentiate in graph_grad.py but are not "
        f"listed in {_DOC.name} -- add a row (and its C++ counterpart, if it has "
        "one) before merging."
    )

    stale_in_doc = documented - actual
    assert not stale_in_doc, (
        f"{sorted(stale_in_doc)} are listed in {_DOC.name} but graph_grad no "
        "longer differentiates them -- remove the stale row."
    )
