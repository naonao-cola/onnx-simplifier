# Opset version converter: profiling `target_opset_version` and fixing the "back to protobuf" cost

**Goal:** the opset version converter (`target_opset_version`, `onnx::version_conversion::ConvertVersion`)
is opt-in and effectively unused in onnxsim's own test suite (two tiny dedicated
tests). Before considering it for wider use, profile it and fix what's actually
slow.

**Tool:** [`bench/opset_conversion_profile.py`](./opset_conversion_profile.py)
(new) -- synthetic, self-contained (no downloads): a low-opset (9) Conv/BatchNorm/Relu
stack with real numpy-backed initializer weights, at a few sizes, run through
`simplify(model, target_opset_version="latest", profile=...)` in a subprocess,
parsing the same `ONNXSIM_PROFILE` summary table `profile_sample.py` uses.
**Build:** this session's `HEAD`, `onnxsim` built via the normal wheel path
(`ONNXSIM_BUILTIN_ORT=OFF`), vendored `third_party/onnx` at `bea23e9` plus the
fix below (max supported default-domain opset: 28).

## Two changes needed before this was even measurable

1. **`target_opset_version="latest"`** (`onnx_simplifier.py`, CLI `--target-opset
   latest`): resolves to the highest default-domain opset this build's
   compiled-in onnx schema registry supports (a new `C.max_default_domain_opset_version()`
   binding), *not* the pip `onnx` package's `onnx.defs.onnx_opset_version()` --
   onnxsim vendors its own fork, which can differ. The `None` default (no
   conversion) is unchanged; this is purely an added way to opt in.
2. **`ConvertOpsetVersion` was invisible to `ONNXSIM_PROFILE`.** In
   `onnxsim.cpp`, the call ran *before* the `Simplify` root `ProfiledScope`, so
   its cost never appeared in the flame graph or the pass-phase summary table --
   it would have shown up only as unaccounted time before the first pass. Fixed
   by giving it its own `ProfiledScope("ConvertOpsetVersion")`, sibling to
   `Simplify`.

## Finding #1: the "back to protobuf" cost -- confirmed, and fixed

`onnx::version_conversion::ConvertVersion` always went through the *copying*
`ImportModelProto(const ModelProto&)` / default `ExportModelProto(...,
consume_tensor_data=false)`, duplicating every initializer's raw bytes twice
(once into the internal Graph IR, once back out) on every call -- even though
`third_party/onnx/onnx/common/ir_pb_converter.h` already ships moving/consuming
overloads (`ImportModelProto(ModelProto&)`, `ExportModelProto(...,
consume_tensor_data=true)`) built for onnx-optimizer's own
`Optimizer::optimize()`. The version converter simply never adopted them.

**Fix:** added `convert_version(ModelProto&, ...)` / `ConvertVersion(ModelProto&,
int)` consuming overloads (in `third_party/onnx`, purely additive -- the
original const-ref copying overloads are untouched, so `onnx`'s own Python
bindings and any other caller are unaffected) and wired onnxsim's
`ConvertOpsetVersion` to reach them by taking its model by value and having the
call site `std::move` it in.

**Measured (median of 5 trials, `ConvertOpsetVersion` span only, same binary,
`ONNXSIM_BENCH_FORCE_COPYING_CONVERT` env toggle used locally for the A/B and
removed before committing):**

| model | initializer bytes | before (copying) | after (moving) | speedup |
|---|---:|---:|---:|---:|
| tiny (1 block) | 0.0 MB | 4.44 ms | 4.31 ms | ~1.0x (no data to copy) |
| medium (20 blocks) | 3.0 MB | 10.13 ms | 6.29 ms | 1.6x |
| large (40 blocks) | 13.3 MB | 28.89 ms | 7.99 ms | 3.6x |
| xlarge (100 blocks) | 92.5 MB | 143.94 ms | 25.55 ms | **5.6x** |

The gap grows with initializer bytes and nothing else -- consistent with a
data-copy cost eliminated, not an algorithmic one. For any model with
non-trivial weights (i.e. most real models), this alone is a multi-x win on
the opset-conversion step specifically.

## Finding #2: a flat ~4ms per-call floor, unrelated to model size or step count

Isolating the fixed part (tiny model, 3 nodes, varying only the starting
opset so `ConvertOpsetVersion` takes 0, 1, or ~19 steps to reach opset 28):

| start opset | steps to target | `ConvertOpsetVersion` |
|---|---:|---:|
| 28 (== target) | 0 (early-out, `ConvertVersion` never called) | 0.03 ms |
| 27 | 1 | 4.53 ms |
| 9 | ~19 | 4.13 ms |

Going from 0 steps to 1 step costs ~4.5ms; going from 1 step to 19 steps costs
*nothing* measurable. So this floor is not the per-step graph walk -- it's the
fixed cost of constructing a `DefaultVersionConverter` at all: its constructor
rebuilds `all_schemas` from `OpSchemaRegistry::get_all_schemas_with_history()`
(copies every registered `OpSchema`, across all ops and versions) and
re-registers on the order of a thousand hardcoded adapters via
`std::make_unique<...>`, from scratch, on every single `ConvertVersion()` call
-- see `third_party/onnx/onnx/version_converter/convert.h`'s constructor and
`convert.cc`'s `DefaultVersionConverter v;` local.

For anything under a few MB of initializers (i.e. a lot of real models), this
~4ms floor is comparable to or larger than Finding #1's data-copy cost, and
today it's paid on *every* `simplify(..., target_opset_version=...)` call
regardless of model size.

**Not fixed in this pass.** A naive "build the `DefaultVersionConverter` once,
cache it as a process-wide static" is unsound: onnxsim registers additional
op schemas into the *same* live `OpSchemaRegistry` at runtime, per model
(`RegisterCustomDefaultDomainOpSchemas`, `RegisterContribOpSchemas`, and
Python's `import_onnx_schemas()` bridging `onnx.defs.register_schema`), so a
cache built on the first call could go stale and either miss a later model's
custom op (throwing "no registered schema") or silently ignore it. A correct
fix needs a cheap way to know the schema registry hasn't changed since the
cache was built; `OpSchemaRegistry` exposes no such generation counter today
(and its backing map is private), so this needs either:

- adding a mutation counter to `OpSchemaRegistry` itself (touches onnx core,
  bumped in `RegisterSchema`/`DeregisterSchema`), or
- a narrower, onnxsim-side counter bumped at onnxsim's own schema-registration
  call sites, threaded through into the fork's `ConvertVersion()` as an
  optional cache-invalidation hint.

Either is a real architecture change to code this session hasn't stress-tested
sufficiently to ship with confidence, given the correctness stakes (silently
converting a model with a stale/missing custom-op adapter is worse than the
~4ms it would save). Recommended as a follow-up, not attempted here.

## What's shipped

- `onnxsim/onnx_simplifier.py`, `onnxsim/cpp2py_export.cc`: `target_opset_version="latest"`.
- `onnxsim/onnxsim.cpp`: `ConvertOpsetVersion` now profiled.
- `onnxsim/model_prep.{h,cpp}`: `ConvertOpsetVersion` takes its model by value
  and forwards it as a mutable lvalue so the call reaches the new consuming
  overload; call site `std::move`s the model in.
- `third_party/onnx` (branch `claude/onnx-version-converter-perf`, pushed to
  `onnxsim/onnx`, not yet merged/re-pinned): the consuming `convert_version`
  / `ConvertVersion` overloads.
- `bench/opset_conversion_profile.py`: the profiling script used above, kept
  for re-measuring after a Finding #2 fix or against real models.

**Verified:** `tests/test_python_api.py`'s two `target_opset_version` tests and
the full `test_python_api.py` module pass; CLI (`--target-opset latest`) and
Python API (`target_opset_version="latest"`) both smoke-tested end to end.
