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

## There is now a real backend too: `tvm_v08_frontend_backend.py`

Unlike the DRP-AI Translator (gated behind a Renesas account) or R-Car's
HyCo (no public code at all), the TVM version DRP-AI TVM vendors is itself
plain Apache-2.0 `apache/tvm` source -- buildable from scratch with no
account, no login, nothing proprietary. `tvm_v08_frontend_backend.py` wraps
the real, compiled `tvm.relay.frontend.from_onnx()` once that build exists,
and `.github/workflows/renesas-integration.yml`'s `real-tvm-v08-frontend`
job actually builds it (see that workflow's comments for the recipe: Ubuntu
22.04's `llvm-14`/`llvm-14-dev`, `set(USE_LLVM llvm-config-14)`, Python
<=3.8 -- confirmed against a real `cmake -G Ninja ..` configure run of the
v0.8 checkout while writing it, not guessed). `tests/
test_renesas_drp_ai_tvm_real_frontend.py` then:

- Diffs the scraped `DRP_AI_TVM_IMPORTABLE_OPS` snapshot against the *live*
  `_get_convert_map()` dict read straight out of the installed `tvm`
  package -- the strongest check here, since it needs no synthetic graph at
  all to catch a stale scrape.
- Confirms a small importable-op graph really does import via the real
  frontend, and that a deliberately unimportable op (`LayerNormalization`)
  raises the exact `tvm.error.OpNotImplemented` message
  `drp_ai_tvm_simulator.unsupported_op_error_message()` predicts for it.
- Feeds the real frontend a `Conv -> BatchNormalization -> Relu` fixture
  before and after a real `onnxsim.simplify()` call (generated by
  `gen_simplify_fixtures.py` in a separate CI job -- see that script's
  docstring for why TVM v0.8's Python <=3.8 requirement and this repo's own
  Python >=3.11 floor can't share one job) and confirms both import.

Still **not** DRP-AI hardware acceleration or the DRP-AI Translator --
those stay behind the gated Renesas account this repository doesn't have.
This only ever validates the open-source TVM ONNX-*import* step, real
build included.

## Legalizing: fixing what `would_import_succeed()` finds

`legalize.py` replaces an ONNX op absent from TVM v0.8's convert map with
its own ONNX operator schema's function-body decomposition -- extracted
straight from `onnx.defs.get_schema()` (`OpSchema.function_body` /
`get_context_dependent_function(...)`, a real `FunctionProto`) and inlined
with ONNX's own `onnx.inliner.inline_local_functions()`, the same tool
ONNX's reference evaluator and backend test suite use for the same
purpose. Nothing here hand-transcribes an op's spec formula. Unlike
`scripts/axelera/legalize.py`'s rules (fixing an op used *outside* its
documented attribute/shape range), there's no attribute value that would
make an op like `HardSwish`/`Mish`/`LayerNormalization` importable as-is
-- TVM v0.8 (2021) simply predates all three (opset 14/17/18) -- so the fix
is expanding it into ops that are importable, and `legalize_via_onnx_
function()` is the one general rule this reduces to: try extracting a
node's schema function, and only commit the replacement if every resulting
op_type is itself importable (fails closed otherwise). This one rule
already covers all three ops with no op-specific code, and picks up any
other function-defined op (`GroupNormalization`, `Gelu`,
`MeanVarianceNormalization`, ...) the same way. `hardswish_to_primitives`/
`mish_to_primitives`/`layer_normalization_to_primitives` remain as thin,
op_types-filtered wrappers around it, kept only so existing callers/`
--rules` keep their names. See `legalize.py`'s module docstring for the
full reasoning, including what this genuinely improves on versus an
earlier hand-written version of this file: that version required `X`'s
rank statically known (to build `ReduceMean`'s `axes` by hand) and
explicitly punted on `stash_type`'s dtype upcast/downcast and the optional
`Mean`/`InvStdDev` outputs (skipping a node needing either); the
schema-derived function handles all of that correctly, since it *is* the
same computation the real op performs, not a re-derivation of it.

```python
import onnx
import legalize

model = onnx.load("model.onnx")
applied = legalize.legalize(model)  # {"legalize_via_onnx_function": 2}
onnx.save(model, "model.legalized.onnx")
```

Or, run after onnxsim's own simplify loop, inside its fixed point with no
rebuild: `onnxsim.simplify(model, custom_rewriter=legalize.
as_custom_rewriter())` -- same `custom_rewriter` contract
`scripts/axelera/legalize.py`'s own adapter uses. `tests/
test_renesas_legalize.py` checks each rule both structurally (against the
ONNX operator spec's own formula) and numerically (`onnx.reference.
ReferenceEvaluator`, before vs. after), and confirms
`drp_ai_tvm_simulator.would_import_succeed()` flips from `False` to `True`
where the rewrite is supposed to fully resolve a node -- none of that
needs TVM installed or a real DRP-AI TVM/HyCo compiler.

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
