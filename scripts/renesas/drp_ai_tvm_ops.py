"""ONNX-frontend-import support data for Renesas RZ/V's DRP-AI TVM
(https://github.com/renesas-rz/rzv_drp-ai_tvm), scraped from the exact
version of Apache TVM's own ONNX importer that project vendors -- see
`scrape_tvm_onnx_frontend.py` for how, and this module's docstring for what
that data does and does not license.

## Why this exists, and why it's narrower than `scripts/axelera/voyager_ops.py`

Axelera's Voyager SDK publishes a real per-operator AIPU support reference
with formal constraint predicates (`voyager_ops.py`'s docstring). Neither
Renesas AI toolchain has an equivalent public document:

- **R-Car's Hybrid Compiler (HyCo)** -- the ONNX/TVM-based toolchain for
  R-Car SoCs (V3H/V3M/V3U/V4H/V4M, and Gen5's X5H) -- has a GitHub repo
  (renesas-rcar/renesas-rcar-HybridCompiler) that is documentation-only, with
  no op-support data published and no code to install; it explicitly directs
  questions to Renesas Technical Support. There is nothing to scrape here,
  and nothing in this directory says anything about R-Car/HyCo at all.
- **RZ/V's DRP-AI TVM** is genuinely open source (real code, no login), but
  the actual per-operator *hardware acceleration* constraint list -- which
  ops the DRP-AI accelerator itself can run, and under what attribute/shape
  restrictions -- lives in the DRP-AI Translator Manual, Section 4.1, which
  is not published in the GitHub repo or (as far as this research found)
  anywhere else publicly.

What DRP-AI TVM's GitHub repo *does* pin publicly and exactly is a specific
Apache TVM version as a git submodule (`.gitmodules`: `branch = v0.8`, as of
the checkout this was scraped against) -- and that TVM version's own ONNX
importer (`python/tvm/relay/frontend/onnx.py`) is real, public code with an
exact, checkable list of which ONNX `op_type`s it can convert into Relay IR
at all. `GraphProto.from_onnx()` raises `tvm.error.OpNotImplemented` for the
*entire* import if even one node's `op_type` isn't in that list (or isn't
"Constant", special-cased separately) -- it does not silently fall back
individual unsupported nodes to CPU. That hard, whole-graph gate is exactly
what `TVM_V08_ONNX_CONVERT_MAP_OPS` reproduces.

## What this module can and cannot tell you

- **Can**: whether `tvm.relay.frontend.from_onnx()` would even accept a
  given graph without raising `OpNotImplemented`, at the specific TVM
  version (v0.8) DRP-AI TVM currently vendors. This is a real, exact,
  reproducible fact about public TVM source -- not an estimate.
- **Cannot**: whether an op that passes this gate is actually accelerated on
  the DRP-AI hardware, vs. dispatched to CPU by DRP-AI TVM's own BYOC
  partitioning pass -- that decision is governed by the undocumented
  Translator Manual. Do not read "importable" as "DRP-AI-accelerated".
- **Cannot**: whether a specific attribute/shape combination on an
  importable op_type actually converts successfully -- TVM's per-op
  converter classes (e.g. `Conv.get_converter()`) can themselves raise for
  unsupported attribute values even when the op_type is in the map; this
  module only reproduces the *outer* op_type gate, not each converter's own
  internal logic.
- **Cannot**: anything about R-Car/HyCo. See above.
- Is tied to **TVM v0.8 / ONNX 1.10.1** specifically (DRP-AI TVM's other
  pinned submodule, `3rdparty/onnx` at branch `rel-1.10.1`) -- a several-
  years-old TVM/ONNX pairing. A newer DRP-AI TVM release could move these
  pins; re-run `scrape_tvm_onnx_frontend.py` against whatever `.gitmodules`
  currently says before trusting this against a current install.
"""

from tvm_v08_onnx_frontend_op_support_data import (
    TVM_V08_ONNX_ALWAYS_IMPORTABLE_OPS,
    TVM_V08_ONNX_CONVERT_MAP_OPS,
)

#: Every ONNX op_type TVM v0.8's Relay ONNX frontend can convert at all,
#: including the "Constant" special case `_get_convert_map()` itself doesn't
#: list. This is the set `drp_ai_tvm_simulator.py`'s `partition()` checks
#: node op_types against.
DRP_AI_TVM_IMPORTABLE_OPS = (
    TVM_V08_ONNX_CONVERT_MAP_OPS | TVM_V08_ONNX_ALWAYS_IMPORTABLE_OPS
)
