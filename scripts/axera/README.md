# Axera Pulsar2/AXCL compatibility check

Verifies that `onnxsim`'s output stays friendly to **Pulsar2**, the compiler
behind Axera's AXCL toolchain that turns an ONNX model into a `.axmodel` for
the AX6xx/AX8xx NPU line. Based on the handoff notes at
[`../../../junk/axcl-axmodel-onnxsim-notes.md`](../../../junk/axcl-axmodel-onnxsim-notes.md),
and since verified against a real **AX650N** (PCIe, via the AXCL host driver
and `axcl_run_model`) and a real compiled `.axmodel`
(`AXERA-TECH/YOLOv8`'s `AX650/yolov8n_640x640_npu1.axmodel`).

## ⚠️ Confirmed on real hardware: onnxsim corrupts compiled `.axmodel` files

**Do not run `onnxsim.simplify()` on an already-compiled `.axmodel`.** This
was verified end-to-end: `axcl_run_model` ran the real file successfully
(~4.8ms/inference on the NPU), then `simplify()` on that same file dropped
its NPU weight/command data and the result failed to even load
(`axcl_run_model` -> "Create model handle failed").

Root cause: the compiled subgraph is a single node, `op_type="neu mode"`,
whose NPU weight/command blobs are ordinary `graph.initializer` tensors
(`npu_params`, `npu_dyn_params`, `<name>_b<N>_neu`) referenced **only** by
name inside a JSON string in the node's `npu_graph_info` attribute --
**not** as a declared node input. Both onnxsim's own constant-folding
cleanup and onnx-optimizer's `eliminate_unused_initializer`/
`eliminate_deadend` passes treat unreferenced-as-input initializers as dead
and drop them; a fresh shape-inference pass also drops the `graph.value_info`
entries describing those tensors, which the real device's loader also needs.

**No combination of `simplify()`'s public parameters avoids this** --
confirmed by exhausting them: `skip_constant_folding=True` alone,
`skipped_optimizers=["eliminate_unused_initializer", "eliminate_deadend"]`
alone, and even both together plus `skip_shape_inference=True`, all still
produced a file `axcl_run_model` refused to load. See `pulsar2_ops.py`'s
docstring for the full record. `pulsar2_ops.has_out_of_band_npu_data()` /
`pulsar2_backend.unsafe_for_simplify()` detect this **before** calling
`simplify()`, and `worker.py` uses it as a hard pre-flight guard
(`pulsar2_unsafe_for_simplify` status) rather than ever calling `simplify()`
on such a model. `tests/test_pulsar2_compat.py::
test_onnxsim_corrupts_a_compiled_npu_subgraph` reproduces the bug against a
synthetic fixture so it's caught in CI without needing the real device --
this confirms the handoff notes' own recommendation to only ever simplify
*pre*-`pulsar2 build` ONNX (approach (b) in the notes), never a compiled
`.axmodel` (approach (a)).

## ✅ Also confirmed on real hardware: approach (b) itself is safe

The real Pulsar2 toolchain (`pulsar2:6.0-lite`, matching the AX650N's
installed firmware) was loaded via Docker and used to actually build two
real `onnxmodelzoo` models end to end -- ONNX -> `pulsar2 build` ->
`.axmodel` -> run on the real AX650N:

- **`resnet18d_Opset18`**: both the original ONNX and its onnxsim-simplified
  twin (onnxsim folded 117 dangling weight-as-input entries down to 1 real
  input, same 56 nodes) compiled to a single NPU subgraph with **identical
  compiler-reported `max_cycle` (1,318,764)**. Running both `.axmodel`s on
  the real device with the same input produced **bit-identical output**
  (`np.array_equal` `True`, max abs diff `0.0`). This is the concrete,
  positive counterpart to the corruption finding above: simplifying
  *pre*-compile ONNX (approach (b)) is safe.
- **`googlenet-6`** (opset 9, uses `LRN`): `pulsar2 build` did not gracefully
  fall `LRN` back to CPU -- it hard-failed the whole build at the frontend
  parse stage (`KeyError('dont support LRN opr in AXOPS/ONNXOPS/CUSTOM_OPS')`)
  before any CPU/NPU partitioning happened. Also below Pulsar2's documented
  minimum opset (11) for AX650. Useful negative data point: an unsupported
  op isn't always "less NPU-friendly," sometimes it's a hard build failure.

This also directly answered an open question from the handoff notes: Axera
publishes the real AX650 NPU op-support list in Pulsar2's own docs
(`appendix/op_support_list_ax650.html`) -- 92 ops, opset >= 11 required.
It's now `pulsar2_ops.AX650_SUPPORTED_OPS` / `AX650_MIN_OPSET`, and
`pulsar2_backend.ax650_build_risks()` uses it to predict (not guarantee) the
two failure modes seen above *before* attempting a real build.

## This one is not like its siblings

[`scripts/qualcomm`](../qualcomm) (QNN), [`scripts/intel`](../intel)
(OpenVINO), and [`scripts/amd`](../amd) (MIGraphX) each wrap a **real**
compiler via a pip-installable ONNX Runtime execution provider, so they
measure actual compile/run behavior. Pulsar2 has neither a PyPI package nor
an ORT execution provider -- it ships as a Docker image -- so there is no
compiler to invoke here for testing *pre*-compile ONNX. (What real hardware
*can* do -- run an already-compiled `.axmodel` via the `axcl_run_model` CLI
-- is a different, narrower thing; see the corruption finding above, which
is exactly what that access was used for.)

So the coverage side of this harness is a **static heuristic**, not a
compiler check: it flags onnx op types that are extremely unlikely to run on
*any* fixed-function NPU (control flow, sequence/optional types, string ops,
data-dependent-shape ops), plus a non-standard ONNX `domain` check that
turned out *not* to be how Axera actually marks a compiled subgraph (see
`pulsar2_ops.py`'s docstring -- the real marker is `op_type="neu mode"` in
the plain default domain). See `pulsar2_ops.py`'s docstring for the full
reasoning and its explicit `CPU_ONLY_OPS` caveats.

## What it checks

For each model:

0. If it already has a compiled Axera NPU subgraph node (`op_type="neu
   mode"`) -> `pulsar2_unsafe_for_simplify`, **without calling `simplify()`
   at all** (see the corruption finding above).
1. Otherwise, `simplify` the model with onnxsim.
2. Compute the static Pulsar2-NPU-blocker set (`pulsar2_ops.blocking_ops`) for
   the original and the simplified graph.
3. If simplification **introduced** a blocking op type that wasn't already
   present -> `pulsar2_regression` (a failure): simplification likely folded
   something into a form Pulsar2's NPU partitioner would reject, pushing more
   of the graph onto its CPU fallback path than before.
4. If simplification dropped NPU weight/command data a compiled subgraph
   node still references -> `pulsar2_data_corrupted` (shouldn't be reachable
   given step 0, checked anyway as defense in depth).
5. If onnxsim's own correctness check reported a mismatch ->
   `simplify_check_failed`.

A model that already has a blocker *before* simplification, or still has one
after but didn't gain a new one, passes (`ok`) -- that's a property of the
input graph, not something onnxsim introduced.

## No-Docker/no-device simulator + compatible quantizer

`pulsar2_simulator.py` and `pulsar2_quantizer.py` turn the confirmed-real
data above into something you can query without the ~1GB Docker image or
physical hardware:

- **`pulsar2_quantizer.quantize_like_pulsar2()`** reproduces Pulsar2's real
  PTQ *numeric convention* -- read directly off a real `quant_axmodel.onnx`
  from the `resnet18d` conversion: **U8 (uint8), per-tensor, asymmetric**
  activations and **S8 (int8), per-channel, symmetric** weights, MinMax
  calibration. It turns out **onnxsim already has a quantizer with exactly
  this convention** -- `onnxsim.quantize_static(method="minmax")`
  (`onnxsim/calibration.py`, an "asymmetric uint8 affine quantization" per
  its own C++ pass's comment) -- so this is now a thin wrapper over
  onnxsim's own quantizer rather than a hand-rolled equivalent built on
  `onnxruntime.quantization`. It does **not** reproduce Pulsar2's actual
  quantized IR: that file's ops are proprietary (`AxQuantizedConv`,
  `AxQuantizeLinear`, ... all in the plain default domain, not standard ONNX
  `QuantizeLinear`/`DequantizeLinear`, and not executable by onnxruntime),
  and onnxsim's quantizer only quantizes Conv/MatMul/"vanilla" Gemm nodes
  where Pulsar2 quantizes essentially the whole graph -- see its docstring.
- **`pulsar2_simulator.py`** adds `partition()`/`coverage()` (per-node
  `AX650_SUPPORTED_OPS` membership -- correctly predicted both real
  conversions: "full" for `resnet18d`, "partial" with
  `{"LRN": 2, "Dropout": 1}` for `googlenet-6`) and `simulate()` (runs the
  quantized graph through onnxruntime's CPU EP as an fp32-vs-INT8 estimate).
  Validated against real hardware: on `resnet18d` with the same input image,
  this simulator's INT8 output had **0.938 cosine similarity** to the real
  device's actual output, close to fp32-vs-real's own **0.949** -- similar
  *magnitude* of quantization noise, but **not** rank/bit-accurate (top-5
  didn't match between fp32, simulated, and real on that input). Both
  degrade gracefully (`SIMULATOR_AVAILABLE`/`PULSAR2_QUANTIZER_AVAILABLE`)
  when `onnxruntime` isn't installed (onnxsim's own `quantize_static` only
  imports it lazily, inside `calibrate()`); `partition()`/`coverage()` need
  only `onnx` and always work.

Use these for a fast first read before spending time on a real
`pulsar2 build` -- always confirm anything that matters on the real
toolchain and hardware, the same way this README's findings were confirmed.

## Real NPU profiling: `chrome://tracing`-compatible trace.json

Confirmed real (this is a genuine Pulsar2 feature, not something this repo
implements): passing `--compiler.npu_perf` to a real `pulsar2 build` writes
`${output_dir}/compiler/debug/subgraph_npu_0/b1/trace.json` -- a standard
Chrome Trace Event Format file (`{"traceEvents": [...], "displayTimeUnit":
...}`, each event `{"ph": "X", "pid": "subgraph_npu_0", "tid": "teng2", ...,
"args": {...}}`) that loads directly in `chrome://tracing` (or Edge's
`edge://tracing`), with one lane per NPU IP (`teng`/`sdma`/`cv`/`conv`) and
one span per hardware task -- op names, dependencies, ddr-swap/load/store
colors. Also pass `--debug.dump_frontend_graph` to get
`frontend/optimized_quant_axmodel.onnx` (openable in Netron) so trace task
labels can be matched back to the algorithm graph. A flat CSV covering the
same data (`op_profile.csv`, one row per op: cycles, bandwidth, tensor
shapes) is written alongside it.

Reproduced against the real `resnet18d_Opset18` build used throughout this
README:

```bash
docker run --rm -v "$PWD:/data" pulsar2:6.0-lite \
  pulsar2 build --target_hardware AX650 \
  --input model/resnet18d.onnx --output_dir output/resnet18d_trace \
  --config config/resnet18d_build_config.json \
  --compiler.npu_perf --debug.dump_frontend_graph
```

This needs a real `pulsar2 build`, not just a compiled `.axmodel` -- it's
generated at compile time from the cycle model, not measured live on-device
by `axcl_run_model`/`ax_run_model` (those only report aggregate min/max/avg
latency). **Automated**: `convert_onnxmodelzoo.py --profile` passes this
through automatically (see below); see Pulsar2's own docs
(`other_tools/profiling.html`) for the full trace-UI reference.

## LLMs: a separate pipeline onnxsim has no hook into

**Confirmed real, end to end** (`pulsar2:6.0-lite` + a real `Qwen/Qwen3-0.6B`
checkpoint + the real AX650N): Axera compiles LLMs through a **completely
different** subcommand, `pulsar2 llm_build` (Pulsar2's newer docs call it
`llm_build2` with a slightly different flag set -- v6.0 only has
`llm_build`; see `pulsar2_docker.llm_build()`'s docstring for the exact
confirmed flags). This is *not* a variant of `pulsar2 build` with an LLM
config -- **`--input_path` is a raw HuggingFace checkpoint directory**
(`*.safetensors`/`pytorch_model.bin` + `config.json`), not an ONNX model.
There is no ONNX step anywhere in this pipeline: the public `ax-llm-build`
project (github.com/AXERA-TECH/ax-llm-build) that Pulsar2's own docs point
to for this workflow contains no model-tracing/export code at all, only
per-architecture config JSONs and small pre/post-processing helper scripts
around the actual (closed-source) `pulsar2 llm_build` call.

**So onnxsim has no direct integration point in Axera's LLM ingestion
path** -- there is no ONNX graph for `onnxsim.simplify()` or any of
onnxsim's GPTQ/AWQ/NF4/`auto_quantize_int4`-family quantizers to act on
before Pulsar2 ever sees the model. `pulsar2 llm_build`'s own
`--weight_type` (`s8` by default, `s4` available) is Pulsar2's own built-in
weight quantization -- unrelated to, and not replaceable by,
`pulsar2_quantizer.py`.

What *is* confirmed and now supported by this harness:

- `pulsar2_docker.llm_build()` wraps the real command. Verified against
  `Qwen/Qwen3-0.6B`: ~7-8 minutes end to end on a 32-core host with
  `--parallel 8`, producing one `<name>_p<prefill_len>_l<N>_together.axmodel`
  per transformer layer (28 for this model) plus one `<name>_post.axmodel`
  (the LM head) -- confirming the original handoff notes' guess that LLMs
  compile to "a directory of small, structurally similar single-block
  graphs," not one big graph.
- Each per-layer file has **two** `neu mode` nodes, not one: a decode
  subgraph (batch-1 shapes) and a prefill subgraph (`prefill_len`-batch
  shapes), sharing one `npu_params` initializer, each with explicit
  `K_cache`/`V_cache` graph inputs *and* `*_out` outputs -- the KV cache is
  ordinary graph tensors the host runtime (`ax-llm`/`axllm`) persists
  between calls, not something hidden inside the compiled blob.
- `pulsar2_ops.py`'s corruption detectors (`has_out_of_band_npu_data()`/
  `missing_npu_data()`) already handle multiple NPU nodes per graph
  correctly with no changes needed. Verified: `onnxsim.simplify()` corrupts
  a real per-layer LLM `.axmodel` the exact same way as the CNN case (3
  initializers -> 0). `models.axera_llm_layer_leaf()` reproduces this shape
  in CI without needing hardware or a real LLM download.
- A per-layer file and the post model both ran successfully on the real
  AX650N via `axcl_run_model` (~1.5ms and ~9ms respectively).

## Real Docker + device conversion driver

`pulsar2_docker.py` and `convert_onnxmodelzoo.py` turn the manual
`docker run ... pulsar2 build` / `axcl_run_model` commands used to produce
every real finding in this README into a reusable pipeline. Unlike
`screen_onnxmodelzoo.py` (static, no Docker/device needed -- run that
first), this does a **real** compile per model, so it needs a loaded Pulsar2
Docker image (see `pulsar2_docker.py`'s docstring for how to get one
matching your device's firmware) and, optionally, a connected AXCL device.

```bash
python scripts/axera/convert_onnxmodelzoo.py \
  --models resnet18d_Opset18 googlenet-6 \
  --profile \
  --output pulsar2-convert.csv
```

For each model: fetches it, `onnxsim.simplify()`s it, `pulsar2 build`s both
the original and simplified ONNX (with `--profile` passing
`--compiler.npu_perf --debug.dump_frontend_graph` through, writing a
`trace.json` per successful build -- see above), and if a device answers,
runs both `.axmodel`s on it with the same input and reports whether the raw
output bytes are bit-identical (this is exactly how the `resnet18d`
bit-identical result in this README was produced). Models are skipped
(`skipped_not_single_image_input`) unless they have exactly one rank-4
input -- NLP/multi-input models need a hand-written config passed to
`pulsar2_docker.build(config_path=...)` directly instead.

One real gotcha worth knowing if you extend this: the Pulsar2 Docker image
must run as root (confirmed: `-u $(id -u):$(id -g)` breaks it -- it needs
root-owned `/root/*.hasplm`/`*.v2c` license files, and a uid absent from the
container's `/etc/passwd` breaks `getpass.getuser()` deep inside a
torchvision import in `pulsar2 version`'s own code path), so everything it
writes under a mounted `work_dir` is root-owned. `pulsar2_docker.
force_rmtree()` handles that (plain `shutil.rmtree` as the host user, falling
back to `docker run --entrypoint /bin/sh <image> -c "rm -rf ..."` on
`PermissionError`) -- use it instead of `shutil.rmtree` for anything under a
Pulsar2 Docker work dir, or root-owned directories accumulate in `/tmp` with
no way for an ordinary user to remove them.

Also note `axcl_run_model -i/-o/-l`'s exact contract, confirmed by trial:
**the input filename must equal the tensor name** (`<in>/0/<tensor_name>.bin`
-- an arbitrary filename fails with "Stimulus file ... is not exist" naming
the tensor). `pulsar2_docker.run_on_device_with_input()` already does this.

## Files

| file | purpose |
| --- | --- |
| `pulsar2_ops.py` | the heuristics and confirmed data: `AX650_SUPPORTED_OPS`/`AX650_MIN_OPSET` (the real, docs-scraped AX650 op list), `CPU_ONLY_OPS` (generic cross-vendor guess), the confirmed `AXERA_NPU_OP_TYPE = "neu mode"` marker, `referenced_const_data_keys()`/`missing_npu_data()`/`has_out_of_band_npu_data()` (the corruption detector), and non-standard-`domain` detection as a fallback for vendor blobs that don't follow Axera's exact convention. |
| `pulsar2_backend.py` | thin wrapper around `pulsar2_ops.py`: `coverage()`, `new_blocking_op_types()`, `stripped_npu_data()`, `unsafe_for_simplify()`, `ax650_build_risks()`. Shaped like the sibling `*_backend.py` modules for interface symmetry (`PULSAR2_AVAILABLE` is always `True` -- there's no external dependency to be missing). |
| `inspect_axmodel.py` | standalone CLI for a **real** `.axmodel` file: loads it with `onnx.load()`, then reports non-standard-domain nodes, op types outside the model's declared opset, and suspiciously large raw attributes -- what originally found the `neu mode` node in the real YOLOv8 file. |
| `models.py` | the shared `scripts/common/synthetic_models.py` suite plus `axera_npu_compiled_leaf` (real CNN `neu mode` node shape) and `axera_llm_layer_leaf` (real per-layer LLM shape: two `neu mode` nodes sharing one initializer) -- no real device needed to exercise the corruption check in CI. |
| `pulsar2_quantizer.py` | `quantize_like_pulsar2()`: a thin wrapper over `onnxsim.quantize_static(method="minmax")`, which already matches Pulsar2's real numeric convention (U8 asymmetric activations, S8 per-channel weights, MinMax calibration). `PULSAR2_QUANTIZER_AVAILABLE` reflects `onnxruntime`'s availability (onnxsim's quantizer needs it internally, imported lazily). |
| `pulsar2_simulator.py` | `partition()`/`coverage()` (real `AX650_SUPPORTED_OPS` membership, no dependency beyond `onnx`) and `simulate()` (fp32-vs-INT8 estimate via `pulsar2_quantizer.py` + onnxruntime's CPU EP). Validated against real hardware -- see above. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_pulsar2_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. No `--require-*` flag or `skipped` status -- unlike the EP harnesses, this needs no vendor package or device, so it always runs. Entry point for `axera-integration.yml`'s `pulsar2-compat` job (stock runner, no Docker/device). |
| `screen_onnxmodelzoo.py` | fast, static, Docker/device-free screening of `onnxmodelzoo` models via `pulsar2_simulator`/`pulsar2_backend.ax650_build_risks()` -- run this first. |
| `pulsar2_docker.py` | real `pulsar2 build` (Docker) + `axcl_run_model` (device) wrapper: `build()` (with `profile=` for `trace.json`), `llm_build()` (the separate, ONNX-free `pulsar2 llm_build` LLM path -- see above), `run_on_device()`, `run_on_device_with_input()`, `force_rmtree()`. Manual/local-only -- needs a loaded Docker image. |
| `convert_onnxmodelzoo.py` | batch driver over `pulsar2_docker.py`: fetch -> onnxsim -> real `pulsar2 build` (orig + simplified, `--profile` optional) -> optional on-device bit-exact diff -> CSV. Entry point for `axera-integration.yml`'s `pulsar2-docker-convert` job -- like `amd-integration.yml`'s MIGraphX check, that job is `workflow_dispatch`-only and targets a `[self-hosted, axcl]` runner this repository doesn't provision, so it's dormant until one exists. |

## Running locally

No extra install beyond onnxsim itself:

```bash
pip install .   # or install an onnxsim wheel

python scripts/axera/run_pulsar2_compat.py --output pulsar2-compat.csv
```

To inspect a real compiled model (and check it for the corruption risk
above before considering running it through onnxsim):

```bash
python scripts/axera/inspect_axmodel.py path/to/compiled.axmodel
```

The in-tree smoke test `tests/test_pulsar2_compat.py` reuses this harness and
needs nothing beyond onnxsim's normal test dependencies (it isn't
skip-guarded like the EP compat tests, since there's no external dependency
to be missing). `tests/test_pulsar2_simulator.py` covers the simulator +
quantizer; its `partition()`/`coverage()` tests are likewise unguarded, but
`simulate()`/`quantize_like_pulsar2()` need `onnxruntime` and skip without it.

To get a fast partition/coverage read or a quantization-noise estimate for a
model, with no Docker or device:

```python
import onnx
from pulsar2_simulator import coverage, simulate  # scripts/axera/

model = onnx.load("model.onnx")
print(coverage(model))          # "full" / "partial" / "none"
print(simulate(model)["close"]) # fp32 vs. simulated-INT8, roughly sane?
```

## Extending

- If the real device/toolchain becomes available again: automate the manual
  `pulsar2 build` + `axcl_run_model -i/-o/-l` (bit-identical output diff)
  flow used for the `resnet18d`/`googlenet-6` conversions above into a real
  `scripts/axera/pulsar2_docker.py` backend, so `worker.py` can do actual
  compiles instead of only the static `ax650_build_risks()` prediction. The
  input/output folder layout for on-device numeric verification is
  `<dir>/0/<name>.bin` + a `list.txt` containing `0` -- see this README's
  git history / session notes for the exact commands used.
- `AX650_SUPPORTED_OPS` only covers AX650; the same docs site has op lists
  for AX620E/AX615/M57/AX637 (`appendix/op_support_list_<chip>.html`) if
  support for those chips is ever needed.
- The real fix belongs in onnxsim itself (or its vendored onnx-optimizer
  fork): some way to mark an initializer as "referenced, don't touch" beyond
  "is a declared node input" -- e.g. recognizing the custom-op placeholder
  schema `model_prep.cpp` already registers for nodes like `neu mode` and
  treating *all* of a model's initializers as roots whenever any such node is
  present, rather than only the ones it happens to declare as inputs.
- `models.py`'s shared suite is intentionally small and self-contained so the
  CI job needs no downloads; a real `.onnx` (pre-`pulsar2 build`) model can be
  layered on by passing an on-disk path as `worker.py`'s second argument, the
  same way `scripts/qualcomm` and `scripts/regression` do.
