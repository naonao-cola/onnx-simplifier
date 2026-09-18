"""A *real* backend for `scripts/renesas/`, wrapping Apache TVM v0.8's
actual `tvm.relay.frontend.from_onnx()` -- unlike `drp_ai_tvm_simulator.py`
(scraped-data-only, no TVM installed at all), this module calls the real
ONNX importer and reports genuinely observed behavior.

This is the exact TVM version `renesas-rz/rzv_drp-ai_tvm` vendors as its
`tvm` git submodule (`.gitmodules`: `branch = v0.8`) -- built from
`apache/tvm`'s own public source, Apache-2.0, no Renesas account needed.
**This still does not touch DRP-AI hardware acceleration or the DRP-AI
Translator** -- that binary (and the AI SDK) require a gated Renesas
account this repository does not have; see `scripts/renesas/README.md` for
exactly where that line is. What this module *can* do that
`drp_ai_tvm_simulator.py` alone can't: confirm the scraped
`TVM_V08_ONNX_CONVERT_MAP_OPS` data actually matches what the real,
compiled TVM v0.8 package does, rather than trusting the scrape.

TVM v0.8 needs a Python <=3.8 interpreter and a from-source C++ build (no
PyPI wheel goes back that far -- `apache-tvm` on PyPI starts at 0.25). See
`.github/workflows/renesas-integration.yml` for how CI builds it; nothing
in this module attempts to build TVM itself, only to use it if already
importable.
"""

from __future__ import annotations

from typing import Optional

import onnx

_INSTALL_HINT = (
    "The real TVM v0.8 ONNX frontend needs a from-source build of "
    "apache/tvm at the v0.8 tag/branch (Python <=3.8; see "
    "docs/install/from_source.rst in that checkout, and "
    "renesas-integration.yml's real-tvm-v08-frontend job for a working "
    "recipe) -- there is no PyPI wheel for this TVM version."
)


def has_tvm() -> bool:
    """Whether a real `tvm` package (any version) is importable here."""
    try:
        import tvm  # noqa: F401
    except ImportError:
        return False
    return True


def real_convert_map_ops() -> frozenset:
    """The actual set of ONNX op_types the installed TVM's
    `tvm.relay.frontend.onnx._get_convert_map()` returns, read live from the
    installed package -- not `drp_ai_tvm_ops.py`'s scraped snapshot. Used to
    cross-check that the snapshot hasn't drifted from what's actually
    installed.
    """
    try:
        from tvm.relay.frontend import onnx as tvm_onnx
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return frozenset(tvm_onnx._get_convert_map(13))


def try_import(model: onnx.ModelProto) -> tuple[bool, Optional[str]]:
    """Attempts the real `tvm.relay.frontend.from_onnx(model)`. Returns
    `(True, None)` on success, or `(False, str(exception))` -- most commonly
    a `tvm.error.OpNotImplemented` whose message this module's caller can
    compare against `drp_ai_tvm_simulator.unsupported_op_error_message()`'s
    (docs-reproduced) prediction for the same graph.
    """
    try:
        import tvm.relay
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    try:
        tvm.relay.frontend.from_onnx(model)
        return True, None
    except Exception as exc:  # noqa: BLE001 -- reporting *any* real failure verbatim is the point
        return False, str(exc)
