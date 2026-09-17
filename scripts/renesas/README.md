# Renesas R-Car / RZ-V ONNX-toolchain compatibility check

A **TVM-frontend-import-only** estimate of whether `onnxsim.simplify()`'s
output stays loadable by Renesas RZ/V's [DRP-AI
TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) -- no compiler, hardware,
or DRP-AI Translator involved, and **nothing here covers R-Car's Hybrid
Compiler (HyCo)** -- see "What this actually is, and isn't" below for why.

## What this actually is, and isn't

Renesas has two separate ONNX/TVM-based AI toolchains, and they differ a lot
in how much is actually public:

- **RZ/V series (DRP-AI TVM)**: real, installable open-source code -- an
  Apache TVM extension (BYOC-based) with ONNX/PyTorch/TensorFlow frontends.
  But the actual per-operator *hardware acceleration* constraint list (which
  ops the DRP-AI accelerator itself runs, under what attribute/shape limits)
  lives in the **DRP-AI Translator Manual, Section 4.1** -- not published on
  GitHub or, as far as this research found, anywhere public. `docs/
  Error_List.md` in that repo has known TVM error messages/workarounds, and
  `docs/Model_List.md` lists validated reference models with FPS numbers;
  neither is a per-op support table.
- **R-Car series (V3H/V3M/V3U/V4H/V4M, and Gen5's X5H) -- Hybrid Compiler
  (HyCo)**: also described as TVM-based (BYOC, with an R-Car ONNX
  Quantizer front end), but its GitHub repo
  (`renesas-rcar/renesas-rcar-HybridCompiler`) is **documentation-only** --
  no source, no op-support data, and explicitly gated: *"No issues or merge
  requests allowed... contact Renesas Technical Support."*

So, unlike `scripts/axelera/voyager_ops.py` (built from Axelera Voyager
SDK's own public, per-operator, formal-predicate support reference), there
is **no public per-operator support matrix to transcribe for either Renesas
toolchain**. Building one anyway would mean fabricating data -- not done
here.

What *is* public and exact is which TVM version DRP-AI TVM vendors: its
`tvm/` git submodule pins `apache/tvm` at `branch = v0.8`
(`.gitmodules`), and that TVM version's own ONNX importer
(`python/tvm/relay/frontend/onnx.py`) is real, public source with an exact,
checkable list of which ONNX `op_type`s it can convert into Relay IR at all
(`_get_convert_map()`). `GraphProto.from_onnx()` raises
`tvm.error.OpNotImplemented` for the *entire* import if even one node's
`op_type` is missing from that list (and isn't "Constant", handled
separately) -- not a per-node CPU fallback. That hard, whole-graph gate is
exactly what this directory checks:

- `scrape_tvm_onnx_frontend.py` -- scrapes `_get_convert_map()` from a local
  `apache/tvm` checkout at the pinned tag/branch.
- `tvm_v08_onnx_frontend_op_support_data.py` -- the scraped output
  (auto-generated, do not hand-edit).
- `drp_ai_tvm_ops.py` -- thin wrapper exposing the combined importable-ops
  set, with the full caveats in its module docstring.
- `drp_ai_tvm_simulator.py` -- `partition()`/`would_import_succeed()`/
  `coverage()`/`unsupported_op_error_message()` over an `onnx.ModelProto`.

**What this can tell you**: whether `tvm.relay.frontend.from_onnx()` would
raise `OpNotImplemented` for a given graph, at DRP-AI TVM's pinned TVM v0.8.
Real and exact -- not an estimate -- for that one question.

**What this cannot tell you**: whether an op that passes this gate is
actually DRP-AI-*accelerated* (vs. dispatched to CPU by DRP-AI TVM's own
BYOC pass -- decided by the non-public Translator Manual); whether a
specific attribute/shape combination on an importable op_type converts
successfully (TVM's per-op converter classes can themselves raise); or
anything at all about R-Car/HyCo.

## Usage

```python
import sys

sys.path.insert(0, "scripts/renesas")  # or run from this directory

import onnx
import drp_ai_tvm_simulator as sim

model = onnx.load("model.onnx")

print(sim.coverage(model))  # 'full' | 'partial' | 'none', by op-type membership alone
if not sim.would_import_succeed(model):
    print(sim.unsupported_op_error_message(model))
```

## Regenerating the scraped data

The scraped data snapshot tracks whatever `apache/tvm` tag/branch DRP-AI
TVM's `.gitmodules` pinned at the time it was last run, not upstream
automatically. Before trusting this against a current DRP-AI TVM install,
re-check `https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/
.gitmodules`'s `[submodule "tvm"]` stanza -- it may have moved past `v0.8`
in a newer Renesas release -- then:

```bash
git clone --branch v0.8 https://github.com/apache/tvm.git /tmp/tvm-v0.8  # or whatever tag .gitmodules now pins
python3 scrape_tvm_onnx_frontend.py /tmp/tvm-v0.8 > tvm_v08_onnx_frontend_op_support_data.py
```
