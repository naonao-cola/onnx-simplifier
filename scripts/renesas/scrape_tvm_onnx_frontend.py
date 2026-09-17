#!/usr/bin/env python3
"""Regenerate `tvm_v08_onnx_frontend_op_support_data.py` from Apache TVM's
own ONNX importer source, at the exact TVM version Renesas's RZ/V DRP-AI TVM
(https://github.com/renesas-rz/rzv_drp-ai_tvm) vendors as its `tvm/` git
submodule (`.gitmodules`: `url = https://github.com/apache/tvm.git`,
`branch = v0.8`, as of the `main` branch checkout this was last run against).

Why this file exists instead of a docs scrape like `scripts/axelera/
scrape_onnx_support_docs.py`: neither Renesas AI toolchain publishes a
per-operator ONNX support matrix the way Axelera's Voyager SDK does.

- R-Car's Hybrid Compiler (HyCo) GitHub repo
  (github.com/renesas-rcar/renesas-rcar-HybridCompiler) is documentation-only
  -- no source, no op-support data, and explicitly gated: "No issues or
  merge requests allowed... contact Renesas Technical Support."
- RZ/V DRP-AI TVM is genuinely open source, but the actual per-operator
  *hardware acceleration* constraint list lives in the DRP-AI Translator
  Manual, Section 4.1 -- a document that isn't published in this (or any
  public) GitHub repo. `docs/Error_List.md` in the repo has known TVM error
  messages/workarounds, and `docs/Model_List.md` lists validated reference
  models with FPS numbers -- neither is a per-op support table.

What *is* public and exact, though, is the ONNX importer TVM itself ships:
`python/tvm/relay/frontend/onnx.py`'s `_get_convert_map()` -- a literal
Python dict whose keys are every ONNX `op_type` TVM's Relay frontend knows
how to convert at all. `GraphProto.from_onnx()` (same file) checks every
graph node's `op_type` against this dict (plus a special-cased "Constant")
before converting anything, and raises `tvm.error.OpNotImplemented` for the
*whole* import if any node's op_type is missing -- not a per-node CPU
fallback, unlike Voyager SDK's AIPU/host partitioning. That exact gate is
what this data (and `drp_ai_tvm_simulator.py`, built on it) reproduces.

**This says nothing about DRP-AI hardware acceleration itself.** An op_type
in this dict just means TVM's frontend can turn it into a Relay op -- DRP-AI
TVM's own BYOC pass then decides, per the undocumented Translator Manual,
which of those Relay ops actually run on the DRP-AI accelerator vs. fall
back to CPU. See `drp_ai_tvm_ops.py` and `drp_ai_tvm_simulator.py`'s module
docstrings for exactly what is and isn't licensed to conclude from this.

Usage: clone apache/tvm at the branch/tag DRP-AI TVM's `.gitmodules` pins
(currently `v0.8`) next to this checkout (or anywhere), then:

    git clone --branch v0.8 https://github.com/apache/tvm.git /tmp/tvm-v0.8
    python3 scrape_tvm_onnx_frontend.py /tmp/tvm-v0.8 \
        > tvm_v08_onnx_frontend_op_support_data.py

To re-check which TVM version current DRP-AI TVM actually pins (it may move
past v0.8 in a future Renesas release), check
`https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/.gitmodules`'s
`[submodule "tvm"]` stanza before re-running this against a different tag.
"""

import re
import sys


def parse_convert_map(onnx_py_path: str) -> list:
    with open(onnx_py_path) as f:
        text = f.read()

    m = re.search(r"\ndef _get_convert_map\(opset\):\n(.*?)\n\n\nclass ", text, re.S)
    if not m:
        raise RuntimeError(
            "couldn't find `_get_convert_map(opset)` in "
            f"{onnx_py_path} -- TVM may have restructured its ONNX "
            "frontend since v0.8; re-check the file by hand."
        )
    body = m.group(1)

    # Every dict entry is `"OpName": <converter expr>,` on its own line;
    # comment-only lines (unimplemented/unmapped ONNX ops Renesas's TVM
    # fork's authors chose not to support) are skipped deliberately --
    # they are exactly the ops this data should NOT claim are importable.
    ops = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        km = re.match(r'"([A-Za-z0-9_]+)":', line)
        if km:
            ops.append(km.group(1))
    return sorted(set(ops))


def main(tvm_root: str) -> None:
    onnx_py_path = f"{tvm_root}/python/tvm/relay/frontend/onnx.py"
    ops = parse_convert_map(onnx_py_path)

    print(
        '"""ONNX op_types importable by TVM v0.8\'s Relay ONNX frontend --\n'
        "auto-generated, do not hand-edit.\n\n"
        "Scraped by `scrape_tvm_onnx_frontend.py` from\n"
        "`python/tvm/relay/frontend/onnx.py`'s `_get_convert_map()` in\n"
        "apache/tvm at the `v0.8` tag/branch, the exact version pinned by\n"
        "renesas-rz/rzv_drp-ai_tvm's `tvm` git submodule (`.gitmodules`) at\n"
        "the time this was run. See `scrape_tvm_onnx_frontend.py`'s module\n"
        "docstring for the full context and caveats -- most importantly,\n"
        "this is TVM's *frontend import* gate, not DRP-AI *hardware\n"
        'acceleration* eligibility.\n"""\n'
    )
    print("TVM_V08_ONNX_CONVERT_MAP_OPS = frozenset(")
    print("    {")
    for op in ops:
        print(f'        "{op}",')
    print("    }")
    print(")")
    print()
    print(
        '#: `GraphProto.from_onnx()` special-cases "Constant" as always '
        "importable, separately from `_get_convert_map()` -- see this "
        "module's docstring."
    )
    print('TVM_V08_ONNX_ALWAYS_IMPORTABLE_OPS = frozenset({"Constant"})')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/tvm-v0.8-checkout", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
