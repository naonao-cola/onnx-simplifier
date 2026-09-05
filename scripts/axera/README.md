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

## Digging into a compiled `.axmodel`'s `neu mode` node

Prompted by "could we generate `.axmodel` without Axera tools?" -- short
answer still no (see below), but here's what direct inspection of a real
compiled file, plus a real `--compiler.npu_perf` trace, actually shows.

**The node's own attributes** (from a real `pulsar2 build` of a tiny Mistral
checkpoint via `build_from_hf_checkpoint()`):

```
neu_name: "subgraph_npu_0"
npu_graph_info: {"name": "subgraph_npu_0", "dotneus": [{"neu_key":
                 "subgraph_npu_0_b1_neu", "batch": 1, "extra_inputs":
                 [{"name": "params", "const_data_key": "npu_params"}]}]}
outputs_info: {"lm_head.matmul.94": ["FP32", [1, 8, 32000]]}
version: <int>
```

`neu_key`/`const_data_key` just name ordinary `graph.initializer` UINT8
blobs (see `pulsar2_ops.py`'s docstring for why onnxsim's own dead-code
elimination strips these): `npu_params` (21MB here -- the raw weight
dump, no header, just concatenated tensor bytes at offsets the other blob
names), `npu_dyn_params` (0 bytes for a static-shape model), and the
`<neu_key>`-named blob itself (28KB here) -- the actual compiled program.

**The compiled-program blob is a FlatBuffers container**, confirmed by
hand-decoding its first 32 bytes against the public FlatBuffers spec: byte
0 is a valid root-table uoffset (28), which resolves through a
well-formed vtable (size 24, 7 populated field slots) -- not a coincidence,
a real, spec-conformant FlatBuffers root table. Scanning the blob for
embedded strings surfaces the vtable's field names, present twice (once
right after the header, once duplicated near the very end of the buffer --
consistent with FlatBuffers' bottom-up buffer-construction convention):

```
params, ddr_swap, lm_head.matmul.94_offset, lm_head.matmul.94,
position_ids_offset, position_ids, input_ids_offset, input_ids, _ocm_base
```

i.e. a **tensor I/O offset table**: a `<name>`/`<name>_offset` pair per
graph input/output, plus `_ocm_base` (the AX650's on-chip SRAM base
address) and `ddr_swap` (matches the real, timestamped "add ddr swap..."
compiler pass -- see the build-phase breakdown two sections up). The
remaining ~27KB in the middle of the blob (>95% of it) has no further
embedded strings or hand-decodable structure -- almost certainly the
actual scheduled NPU instruction stream, in a proprietary, undocumented
encoding.

**No usable Axera-provided FlatBuffers schema was found.** Searched (inside
`pulsar2:6.0-lite`): no `*.fbs`/`*.bfbs`/`*_generated.{h,py}` files
anywhere under the image; no `import flatbuffers`/`from flatbuffers` in any
plaintext `.py` file under `/opt/pulsar2` (the only such hits anywhere in
the image are ONNX Runtime's own unrelated `.ort`-format schema, bundled as
a dependency); no relevant field-name strings (`ddr_swap`, `_ocm_base`,
`neu_key`, ...) or `flatbuffers`/`.fbs` mentions in any of the five
`backend/*/*_cmodel.so` libraries (these are almost certainly cycle-accurate
NPU functional simulators used for verification, not the FlatBuffers
writer). The `flatbuffers` PyPI package itself **is** installed in the
image (confirming the container format), but whatever Python code actually
constructs this schema lives inside the Pyarmor-obfuscated `yamain`/
`yasched`/`opset` modules (see `pulsar2_ops.py`'s docstring) -- not
recoverable by inspection.

**A real `trace.json` (see the profiling section above), though, gives away
almost the entire semantic content of that opaque instruction stream --
in plain, readable JSON, no reverse engineering needed.** For the same
tiny Mistral build (`--profile`), the 635 trace events reveal:

- **Five named parallel execution engines** (`tid` values): `conv0`/`conv1`
  (213/210 events -- the MAC/matmul compute units), `cv3` (78 events -- a
  vector/elementwise unit: RMSNorm, RoPE rotation, Softmax), `sdma4` (76
  events -- a system-DMA/prefetch engine), `teng2` (58 events -- handles the
  embedding gather and I/O staging). A real, confirmed heterogeneous
  multi-engine architecture, not a single monolithic "NPU core."
- **Named memory regions**, matching the FlatBuffers offset table above:
  `ocm_base` (871 references -- the dominant, fast on-chip working memory),
  `params` (154 -- DRAM-resident weights), `ddr_swap` (2 -- DRAM staging for
  spilled tensors), plus the three named I/O tensors.
- **Original ONNX op names are preserved end to end** (e.g.
  `model.layers.0.q_rope.rot.41`, `model.layers.0.attn_norm.var_eps.18`),
  each lowered to a small set of NPU primitives: `onnx.FullyConnected`
  (417/635 events -- every projection and the FFN, all lowered to the same
  primitive), `AxQuantizedMatMul` (3), `onnx.Silu` (1).
- **The trace's own time units are NPU cycles** (scaled by 1000, despite
  the file's `displayTimeUnit: "ns"`): summing every event's `dur` gives a
  total schedule span of ~287,231, matching this exact build's own reported
  `max_cycle=287,211` (see `BuildResult.max_cycle`) to within rounding.

Net effect on "could we generate `.axmodel` without Axera tools": no change
to the answer, but a much better-understood boundary. The *container*
(FlatBuffers) and the full *dataflow graph* (`trace.json`, when
`--profile` is used) are both now understood well enough to write a reader
without Docker. Actually *producing* a correct, hardware-loadable
instruction stream from that dataflow graph -- real quantization, tiling,
scheduling, and codegen into an undocumented ISA -- still requires
Pulsar2's own (obfuscated) compiler backend.

### What's op-specific vs. boilerplate, across real Conv/MatMul variants

The single-model dig above raises an obvious question: how much of that
FlatBuffers offset table and instruction stream is generic wrapper vs.
op-specific? Answered by compiling 9 small, hand-built ONNX graphs (plain
`MatMul`, `Gemm` with bias, dense `Conv` 3x3, 1x1 pointwise, depthwise
(`group=C`), grouped (`group=2`), stride-2, dilation-2, and a batched
(rank-3) `MatMul`) through the same real `pulsar2 build --compiler.npu_perf`
and inspecting each result the same way:

- **The FlatBuffers field-name table is identical across every one of the
  9 models**: always exactly `params`, `<input>_offset`, `<output>_offset`,
  `_ocm_base`, and the graph name -- regardless of kernel size, stride,
  dilation, groups, or op family. **No op parameter ever shows up as a
  named field.** Conv's stride/padding/dilation/group and Gemm's alpha/
  beta/transpose flags are entirely opaque, baked into the unlabeled
  instruction bytes -- this table is pure I/O bookkeeping, not a
  semantically rich IR.
- **`ddr_swap` is a real, conditional field**, not a fixed part of the
  schema: present in the earlier 28-layer LLM build (something spilled to
  DRAM), absent from all 9 of these small models (everything fit in OCM).
- **Exactly two compute "primitive families" appear**, cleanly split by op
  family: every `Conv` variant -- dense, 1x1, depthwise, grouped, strided,
  dilated, no exceptions -- lowers to `Pre_AxTranspose` -> `AxQuantizedConv`
  (x6 tiles) -> `Post_AxTranspose` (almost certainly a NCHW<->NHWC layout
  swap around a channel-last-native conv engine); `MatMul` lowers to
  `onnx.FullyConnected` instead. Op parameters change *within* those
  primitives (invisibly) but never *which* primitive gets picked.
- Trace event **naming isn't fully consistent**: `matmul2d`'s compute
  events are labeled `op_1:onnx.FullyConnected_<tile>_<tile>` (primitive
  name visible), but `gemm_bias`'s and the batched `matmul_batched3d`'s are
  labeled directly after their own output tensor (`y_0_0`, `y_1_2`, ...) --
  the same underlying compute, differently named depending on some
  internal fusion/naming decision, not a reliable way to detect op type
  from the trace alone.
- **Cycle cost and weight-blob size don't scale the way FLOP count would
  predict, at this tiny (16x16, 4-8 channel) test size.** 1x1 pointwise
  conv costs nearly as many cycles as full dense 3x3 (2052 vs. 2096) --
  fixed per-tile overhead dominates raw MAC count here. Most strikingly,
  **dilated conv's stored weight blob is 2.6x larger than a plain conv with
  the identical (8,4,3,3) kernel shape** (3.66KB vs. ~1.4KB) despite having
  the same number of logical weight values -- strong evidence the compiler
  materializes a real, zero-expanded ("atrous") kernel footprint for
  dilation rather than an actually-sparse dilated MAC pattern, and it's
  also the most expensive op tested by cycle count (2228).
- Every model tiles into a small, similar instruction count regardless of
  large parameter differences at this scale: all 6 `Conv` variants compile
  to exactly 6 `AxQuantizedConv` sub-tiles each; `matmul2d` to 6
  `FullyConnected` sub-tiles; the batched matmul to 7. Tiling granularity
  here looks governed by fixed hardware tile-size constants more than by
  the specific op's shape/parameters -- this may well change at larger,
  more realistic tensor sizes where tiling actually has to split work up.

### External corroboration: a real hardware teardown

An independent third-party writeup --
[jas-hacks.blogspot.com's AX650N/Sipeed M4N teardown](https://jas-hacks.blogspot.com/2024/09/ax650n-sipeed-maix-iv-axerapi-pro-npu.html)
(not Axera's own documentation; treat specific numbers as one outside
source's reporting, and any interpretive claims -- explicitly flagged
below -- as that author's own inference, not confirmed fact) -- gives real
names and numbers for the hardware this repo's own trace.json digging
above only inferred generically:

- The NPU ("Neutron") is described as 13 execution units + 3 SDMA units:
  3 Convolution Units (handling depthwise/grouped conv, dilation, and
  ConvTranspose), 3 Computer Vision Units (image normalize/resize/clip/
  warp), 3 Tensor Units (activation, pooling, elementwise, reduction), and
  a single Matrix Arithmetic Unit (int8/int16 in, fp16/fp32 out). This
  lines up with this repo's own trace.json engine names in outline --
  `conv0`/`conv1` for the Convolution Units, `sdma4` for an SDMA unit --
  but not exactly: our own LLM trace's `cv3` engine ran RMSNorm/RoPE
  elementwise math, which reads as "Tensor Unit" work by this source's own
  description, not "Computer Vision Unit" work, and we never observed a
  distinct engine for the single Matrix Arithmetic Unit despite compiling
  real `MatMul`/`Gemm` graphs above (both engine-name schemes may not be
  directly comparable, or the compiler may route elementwise math onto
  whichever engine family has spare capacity rather than a fixed
  CV-vs-tensor split). Reported, not reconciled -- a real open question
  for anyone digging further.
- **On-chip memory (OCM): reported as 11.5MB, address space ending at
  `0xAFFFFF`.** `0xAFFFFF + 1 = 0xB00000 = 11,534,336` bytes = exactly
  11MiB by that address range -- close to but not exactly the "11.5MB"
  figure quoted; direct, checkable confirmation that `ocm_base`'s byte
  offsets seen in this repo's own trace.json digging above (all under
  ~3.2MB in our tiny test models) sit well inside a real, multi-megabyte
  on-chip SRAM, not some other memory space.
- 8GB total SoC RAM, split 4GB Linux / 4GB "CMM" (Contiguous Memory Model)
  for peripherals -- CMM is almost certainly what this repo's own findings
  call `params`/DRAM-resident weight storage and `ddr_swap` staging.
- Claimed performance: 72 TOPS mixed precision (18.0 TOPS@INT8, 43.2
  TOPS@INT4 and 10.8 TOPS@INT8 "from NPU alone" per Axera's own SDK docs,
  per that source). **The author's own interpretation** (not a measured
  fact): a single Matrix Arithmetic Unit instance may bottleneck LLM
  inference, since every `MatMul`/`Gemm` in a transformer routes through
  it. That's a plausible complementary explanation for *why* LLM inference
  is slow on this hardware, alongside (not instead of) the very different,
  independently-confirmed bottleneck this repo's own `demo_hf_llm_chat.py`
  measured: `axcl_run_model`'s ~700ms-per-invocation process/model-reload
  overhead, which has nothing to do with the NPU's own compute engines at
  all and would dominate regardless of how many Matrix Arithmetic Units
  existed.
- Real production reference point (**not comparable to
  `demo_hf_llm_chat.py`'s own measured tokens/sec** -- different
  measurement entirely: real `ax-llm` + KV-cache decode via Pulsar2's own
  `llm_build()` path, not this repo's re-run-the-whole-model-per-token
  `build_from_hf_checkpoint()` path, and no per-call CLI-reload overhead
  since it's a persistent server): Phi-3 Mini reported at ~4.4 tokens/sec
  on the AX650N, vs. ~6.46 tokens/sec on an RK3588 for comparison.
- Confirms real ONNX-level `.axmodel` structure from an outside source
  independently: "axmodel files contain a mix of ONNX data and an internal
  graph representation" sent to the NPU kernel driver -- matching this
  repo's own finding of an ordinary ONNX container wrapping an opaque,
  FlatBuffers-framed internal representation.
- **Confirmed, and more specific than reported**: `gemm_bias` above used
  default `alpha=1.0, beta=1.0` and compiled fine; a `Gemm` with
  non-default values (`alpha=2.0, beta=0.5`) **fails outright**, not
  merely "restricted" -- a real `pulsar2 build` on that graph throws
  `KeyError: 'dont support AxQuantizedGemm opr in AXOPS/ONNXOPS/
  CUSTOM_OPS'` before quantization even runs. Default-alpha/beta `Gemm`
  apparently lowers to the same path as a plain `MatMul` + bias-add (hence
  `gemm_bias` succeeding above); any other `alpha`/`beta` maps to a
  distinct, entirely unimplemented `AxQuantizedGemm` op. Confirms the
  blog's suspicion with a precise, reproducible mechanism.

### Differential analysis: how elementwise ops and Conv bias get encoded

The dig above characterizes one model's compiled output; this pushes
further with **differential analysis** -- compiling many near-identical
graphs and byte-diffing the results to locate exactly where a specific,
controlled value ends up. Test graph throughout: `Add`/`Sub`/`Mul`/`Div`
between a `float[1,4]` input and a uniform-broadcast constant, or a `Conv`
with a bias term -- varying only the constant/bias value between builds.

**A "trivial" fast-path exists for small uniform constants, scale=1,
zero_point=0, storing the constant's own integer value as a raw byte** --
but confirmed real by testing across all four ops, **the trivial *set*
is op-specific, not a shared threshold**:

- `Add`: exactly the uniform values `{0, 1, 2}` are trivial; `3` and up,
  any negative value, and any non-integer are not.
- `Sub`: only `{0, 1}` -- `Sub(x, 2.0)` is *not* trivial, unlike
  `Add(x, 2.0)`. Not simply "`Sub(x, c)` lowers to `Add(x, -c)`" either:
  that would predict `Sub(x, 1.0)` (i.e. `Add(x, -1.0)`) to behave like
  `Add`'s confirmed-non-trivial negative case, but it doesn't -- it's
  trivial, storing `01 01 01 01` same as `Add(x, 1.0)`.
  `Mul`: every uniform value tried (`0`, `2`, `3`) was trivial -- no
  "rich" encoding observed for `Mul` at all.
- `Div`: triviality depends on **the constant's reciprocal**, not the
  constant itself -- `Div(x, 0.5)` (reciprocal `2.0`) is trivial, storing
  `02 02 02 02`, while `Div(x, 2.0)` and `Div(x, 3.0)` (reciprocals `0.5`,
  `0.333...`, non-integer) saturate every element to `0xff`. Consistent
  with `Div(x, c)` being compiled as `Mul(x, 1/c)` internally.

**Any non-integer value saturates every element to `0xff` (255)**,
regardless of magnitude -- confirmed across `Add`/`Sub`/`Div`'s non-integer
cases (`0.5`, `3.14159`, and `Div`'s non-integer effective reciprocals).
It's specifically about exact integer-valuedness of whatever value is
actually being quantized (the reciprocal, for `Div`) -- not "small enough."

**A uniform broadcast is required for the trivial path, even when every
individual element already qualifies**: `Add` with the mixed constant
`[1, 1, 2, 2]` (every element in the "trivial" set `{0,1,2}`) still gets
the non-trivial encoding, because the *tensor* isn't a uniform single-value
broadcast. Mixed constants still store their exact literal integer values
per element in the non-trivial path (`[1,2,3,4]` -> `01 02 03 04`), so
"non-trivial" doesn't mean "imprecise" -- it means "not the degenerate
single-value fast path."

**Confirmed real, reproducible compiler bug**: `Mul(x, 1.0)` and
`Div(x, 1.0)` both crash a real `pulsar2 build` with the identical
`NotImplementedError: Seems config of input(y) doesn't exist`. Multiplying
or dividing by the identity constant appears to get eliminated by
Pulsar2's own frontend graph optimizer (`x*1=x`, `x/1=x`) before
quantization runs, leaving the declared graph output with no producing
node. `tests/test_axera_neu_format_arith_ops.py::
test_mul_and_div_by_one_crash_the_real_build` locks this in.

**`Div(x, 0.0)` doesn't error -- it stores literal IEEE-754 `+Infinity`**:
`00 00 80 7f` (float32 `+inf`) repeated once per element, a third,
distinct byte-length class from the other two, and the only case found
where this field holds genuine float32 data instead of an integer code --
a sensible fallback once the "true" quantized value is undefined.

**A field that resists decoding, isolated but not solved**: `Add`/`Sub`'s
non-trivial encoding appends 4 extra bytes past the per-element values.
Ten-plus decodings were tried and rejected (float32, uint32, a `bf16`
pair, an `fp16` pair, `xxhash32`/`xxhash64` of several byte encodings of
the constant, a hand-computed asymmetric output-quantization scale/
zero-point from the real calibration data) -- none matched. What *is*
confirmed: holding the constant fixed (`c=99`) and varying only the
input's calibration scale (x1, x100, x0.01) changed these bytes
completely while the constant's own quantized bytes stayed identical --
so the field depends on the input/output's calibration range, not the
constant alone. The two 16-bit halves are also mathematically coupled:
treating them as `(pair1, pair2)`, `pair2 * scale_y ≈ pair1` held to
within rounding across all three calibration scales, where `scale_y` is
the real `(max-min)/255` output range independently computed from the
actual calibration samples used. A real, non-arbitrary (value,
value-expressed-in-quantization-units) pair -- just one whose absolute
unit/format wasn't identified.

**`Conv`'s bias term, by contrast, decodes cleanly**: byte-diffing five
otherwise-identical `Conv` builds that differ only in bias value locates
the bias-dependent region precisely, and it holds two 4-element
(one per output channel) plain `float32` arrays -- not further-obfuscated
integer codes. One array is small (~0.0026-0.0037) and shrinks
monotonically as the bias value grows, consistent with a per-channel
requantization multiplier `M_channel = input_scale * weight_scale_channel
/ output_scale` (a larger bias widens the calibrated output range, so
`output_scale` grows and `M_channel` shrinks) -- exactly the standard
quantized-conv parameterization real edge-inference runtimes use. The
other array (larger magnitude, ~-159 to 242) plausibly holds a quantized
bias term but wasn't independently re-derived from scratch. Real,
recognizable structure here, in clear contrast to `Add`/`Sub`'s still-
opaque field above.

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
- **Confirmed real, directly from a compiled layer's own declared I/O
  dtypes (a real `HuggingFaceTB/SmolLM2-135M` build, `--help`'s
  `hidden_state_type`/`weight_type` defaults of `bf16`/`s8`): this is
  genuine weight-only quantization, not the full weight+activation INT8
  PTQ the generic path below applies.** Every graph input/output on both
  `neu mode` nodes -- `K_cache`, `V_cache`, the hidden state, and the
  attention `mask` -- is declared `BFLOAT16`; activations never get
  quantized at all. Only `npu_params` shrinks: 3,712,328 bytes for a
  576-hidden-size layer whose real weight element count (q/k/v/o/gate/up/
  down projections + 2 RMSNorm weights) is ~3.54M -- ~1 byte/element,
  confirming S8 weights, not the ~2 bytes/element BF16 would need. This is
  the confirmed, direct explanation for the real accuracy gap found
  below ("Confirmed against a real, full-size model"): the generic
  `pulsar2 build --config` path quantizes *both* weights and activations
  uniformly to INT8 with no smoothing, which compounds into near-random
  output by 30 layers deep; `llm_build()` never quantizes the residual
  stream/KV-cache/attention path at all, only the static weights.
- **`model_type` support is narrower here than the generic path's**:
  `llm_build --input_path` on a real `mistral`-architecture checkpoint
  (`distilabel-internal-testing/tiny-random-mistral`, same one used
  elsewhere in this README) fails outright with `AssertionError:
  model_type error mistral` -- confirming its per-architecture allowlist
  (`yasched/llm_builder/{llama,qwen3,gemma,...}_test.py`, all
  Pyarmor-obfuscated, see above) has no `mistral` entry, unlike
  `reconstruct_hf_graph()`, which treats `mistral` as llama-family-
  compatible. `llama` (confirmed via `SmolLM2-135M`) and `qwen3`
  (confirmed via `Qwen3-0.6B`) both work.

## An alternative LLM path that *does* give onnxsim a hook

The section above is about Pulsar2's own, closed-source `pulsar2 llm_build`
ingestion path, which never touches ONNX. Separately, onnxsim has its own
`onnxsim.reconstruct_hf_graph()` (see `onnxsim/hf_reconstruct.py`) --
builds a runnable ONNX graph directly from a HF checkpoint directory
(`config.json` + safetensors; llama/mistral/qwen2/qwen3 today). Feeding
*that* ONNX graph through the ordinary `pulsar2 build` (the same
CNN/vision ingestion `convert_onnxmodelzoo.py` uses, not `llm_build`) is a
second, independent LLM path with a real onnxsim integration point --
`onnxsim.simplify()`/quantizers can act on the graph before Pulsar2 ever
sees it, unlike the `llm_build` path above.

**Confirmed real, end to end**: a synthetic tiny (2-layer) Llama-shaped
checkpoint, run through `reconstruct_hf_graph()` then a real `pulsar2
build --target_hardware AX650`, compiled cleanly to a single-`neu
mode`-node `compiled.axmodel`, which then ran successfully on a real
AX650N via `axcl_run_model`. Notably, `pulsar2_ops.AX650_SUPPORTED_OPS`
(the doc-scraped op list) flags `Neg` (used by RoPE's rotate-half) as
unsupported, but the real build compiled it without complaint regardless
-- a reminder that the scraped table is a fast pre-screen, not a
guaranteed predictor, once fused patterns are involved.

`pulsar2_docker.build_from_hf_checkpoint()` wraps this whole path:
reconstructs the ONNX graph, auto-generates `Numpy`-format calibration
tars for `reconstruct_hf_graph`'s two inputs (`input_ids`, random token
ids in `[0, vocab_size)`; `position_ids`, `arange(seq_len)`), writes the
two-input quant config Pulsar2 needs (`calibration_format: Numpy` per
`InputQuantConfig`, confirmed from the Docker image's own
`build_config.proto`), and calls `build()`. See
`tests/test_pulsar2_hf_to_axmodel.py` for the full working example.

### Confirmed against a real, full-size model: `HuggingFaceTB/SmolLM2-135M`

Everything above was verified against tiny synthetic or near-random-weight
checkpoints. Compiling a real, genuinely-trained 135M-parameter checkpoint
(30 layers, GQA, 49152-token vocabulary, real BF16 weights) through this
same path surfaced a real bug this repo's own BF16 handling had never hit
before, plus a real accuracy caveat:

- **The `Cast`-in-graph BF16 design was never actually exercised against a
  real `pulsar2 build` until now, and it's fundamentally broken there,
  at any size.** `reconstruct_hf_graph()` (confirmed against the real
  ~1.5GB `Qwen/Qwen3-0.6B` checkpoint, see above) always used a
  graph-level `Cast` node for BF16 weights specifically to keep the
  initializer small and avoid protobuf's ~2.1GB serialization limit -- but
  that Cast node had only ever been run through `onnxruntime`, never a
  real Pulsar2 compile. Compiling `SmolLM2-135M` for real hit
  `Exception: op name: model.embed_tokens.weight.f32.1, Cast, pyrun
  failed.` inside Pulsar2's own frontend constant-folding pass. Isolated
  to a standalone, minimal repro: a bare `Cast<to=FLOAT>` on a BFLOAT16
  initializer fails identically at *every* size tested, from a trivial
  4-element tensor up through the real 49152x576 embedding table --
  ruling out "too large" and confirming it's simply unimplemented for
  this dtype pair in Pulsar2's frontend, full stop.
- **Fixed**: `reconstruct_hf_graph()` now upcasts BF16 weights to FLOAT32
  directly in the stored initializer bytes (`_read_tensor()`), the same
  as the *first* approach that Qwen3-0.6B's size had ruled out -- except
  there is no longer a smaller alternative for real hardware, so the size
  cost is accepted. Confirmed safe in practice for a real small/edge-sized
  checkpoint: `SmolLM2-135M`'s ~269MB BF16 checkpoint upcasts to a
  ~251MB *compiled* `.axmodel` (weights end up INT8-quantized on
  Pulsar2's own side, well under the protobuf limit regardless).
  A checkpoint large enough that the FLOAT32 upcast alone would exceed
  ~2.1GB has no working path through `build_from_hf_checkpoint()` today --
  that's what `llm_build()` (above) is for.
- **Compiled successfully**: ~105s wall time (`pulsar2_build` phase; see
  `BuildResult.phase_timings`), `max_cycle=7,334,676` -- and ran
  successfully on the real AX650N with a real tokenized prompt.
- **Real accuracy is bad, and it's a depth-compounding problem, not a
  calibration problem** -- corrected after actually comparing on-device
  output against the real FP32 reference (an earlier pass here just
  eyeballed logit plausibility and wrongly called it fixed). Across 5 real
  prompts, comparing the compiled model's on-device logits against
  `onnxruntime` running the same `reconstruct_hf_graph()` output: **0/5
  top-1 matches, 0/5 top-5 overlap, average cosine similarity ~0.13** --
  the FP32 reference gets every prompt right (" the" for "The capital of
  France is", " dog" for "...the lazy", " oxygen" for "hydrogen and", ...,
  confirming the reconstruction itself is correct), the on-device output
  is close to random. Two follow-ups ruled out calibration as the cause
  rather than confirming it: real, representative English-sentence
  calibration data (32 real sentences, not random token ids) made it
  *worse* (avg cosine ~0.04), and switching `calibration_method` from
  `MinMax` to `MSE` made it worse again (~-0.12). **The real cause,
  isolated by depth**: the identical reconstruction+quantization approach
  gets 0.999 average cosine similarity and 4/5 top-1 matches on a
  synthetic **1-layer** checkpoint (`distilabel-internal-testing/
  tiny-random-mistral`, same pipeline, same code) -- so per-tensor MinMax/
  MSE INT8 post-training quantization, applied uniformly to every weight
  and activation with no smoothing or outlier handling, works fine at
  shallow depth and compounds into essentially-random output somewhere
  between 1 and 30 sequential transformer layers. This is a well-
  documented, expected limitation of naive full-network INT8 PTQ on deep
  transformers in general (it's exactly why techniques like SmoothQuant/
  AWQ/GPTQ exist), not a bug in `reconstruct_hf_graph()`,
  `build_from_hf_checkpoint()`, or its calibration data. Getting real
  accuracy out of a real-depth LLM through this generic ingestion path
  would need a smarter quantization strategy than what a plain `pulsar2
  build --config` currently applies -- `llm_build()` (above), Pulsar2's
  own dedicated LLM path, presumably has one; this generic path doesn't.

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

## Conv/MatMul variants: what the static heuristic can and can't tell you

Prompted by "how does the axmodel format actually treat different kinds of
Conv/MatMul" -- with no Docker image or AX650N in *this* environment (unlike
the sessions that produced the real-hardware findings above), the honest
thing to check is what the checked-in **static heuristic**
(`pulsar2_ops.AX650_SUPPORTED_OPS` + `pulsar2_simulator.partition()`) can and
can't distinguish, since that heuristic is all a Docker/device-free
environment has to go on. `tests/test_axera_conv_matmul_coverage.py` builds
~16 Conv/MatMul variants via `onnx.parser` and checks them against it:

- **Standard, grouped, depthwise, dilated, strided, `auto_pad`-using, 1-D,
  3-D Conv, and `ConvTranspose`** all read as identical, full NPU coverage.
  So do plain `MatMul`, broadcasting/batched `MatMul`, and `Gemm` under every
  combination of `alpha`/`beta`/`transA`/`transB`. This isn't a bug in the
  test -- it's `partition()`'s own documented design: it classifies purely by
  `node.op_type` membership in `AX650_SUPPORTED_OPS`, the same list
  `inspect_axmodel.py`/`pulsar2_ops.py` scraped from Pulsar2's docs, which
  says nothing about attributes. The docs page itself has per-op
  attribute-level limits (e.g. Conv's `auto_pad` must be `NOTSET`) that
  neither this list nor `partition()` encode -- confirmed absent, not
  confirmed present, since there's no compiler here to check it against.
  Extending `pulsar2_ops.py` with real attribute limits needs a source of
  truth this environment doesn't have (the docs page or a real `pulsar2
  build` failure); making that gap up would be exactly the kind of
  unconfirmed guess this harness otherwise avoids.
- **What *is* checkable without any of that**: none of ONNX's own quantized
  conv/matmul ops (`QLinearConv`, `ConvInteger`, `QLinearMatMul`,
  `MatMulInteger`) are in `AX650_SUPPORTED_OPS` at all, so `partition()`/
  `ax650_build_risks()` flag them as an AX650 build risk regardless of shape.
  This lines up with `pulsar2_quantizer.py`'s separately-confirmed finding
  that Pulsar2's own real PTQ output uses proprietary `AxQuantizedConv`-family
  ops, not standard ONNX quantized operators -- so a graph already quantized
  with ONNX's own vocabulary (e.g. via `onnxsim.quantize_dynamic`, unlike
  `pulsar2_quantizer.quantize_like_pulsar2()`, which stays in QDQ form) is
  something Pulsar2's real frontend has never been confirmed to accept, and
  this heuristic's answer (flag it) is at least consistent with that.
- This also surfaced (and fixed) a real bug in the "no Docker/no-device
  simulator" pitch above: `pulsar2_simulator.py`'s docstring and this
  README both claim `partition()`/`coverage()` "need only `onnx` and work
  regardless" -- but `pulsar2_simulator.py` unconditionally imported
  `pulsar2_quantizer.py`, which unconditionally did `import onnxsim` at
  module scope, so on a checkout where `onnxsim`'s own compiled extension
  isn't built yet (this analysis' own environment, notably -- no real
  `.axmodel`, Docker image, or device either), just importing
  `pulsar2_simulator` for its `onnx`-only `partition()`/`coverage()` raised
  `ModuleNotFoundError` before either function ever ran. Fixed by moving the
  `import onnxsim` inside `pulsar2_quantizer.py`'s existing
  `PULSAR2_QUANTIZER_AVAILABLE` try/except (alongside the `onnxruntime`
  import already there), so a missing/unbuilt `onnxsim` degrades the same
  way a missing `onnxruntime` already did, instead of taking the whole
  import down.

### Confirmed on real hardware: which of these actually build

A later session with real Docker/AX650N access compiled every variant the
static-heuristic analysis above flagged as unverified, through a real
`pulsar2 build`. Results (see
`tests/test_axera_conv_matmul_coverage_hardware.py`):

**Compile successfully, confirming the doc-scraped list under-claims
nothing for these cases**: `Conv` with `auto_pad="SAME_UPPER"` (despite the
docs page reportedly requiring `NOTSET` -- either that limit doesn't hold in
practice for this case, or Pulsar2 silently resolves `auto_pad` to explicit
`pads` before its own limit would apply), 1-D `Conv`, 3-D `Conv`, broadcasting
`MatMul` (rank-3 `A` against a rank-2 `B`), and `Gemm` with `transB=1`.

**Fail outright, with the same "not on `AX650_SUPPORTED_OPS`" mechanism the
static heuristic already predicted**: `ConvInteger`, `QLinearConv`,
`MatMulInteger`, `QLinearMatMul` -- all four real `pulsar2 build` runs threw
the exact `KeyError('dont support <OpType> opr in AXOPS/ONNXOPS/CUSTOM_OPS')`
pattern this repo has seen before (`LRN`, `AxQuantizedGemm` -- see above),
confirming these standard ONNX quantized ops are genuinely unimplemented on
this real toolchain, not merely absent from a possibly-incomplete
docs-scraped list.

**Fails, but with a real, different failure mode neither analysis
predicted**: `ConvTranspose` -- despite being confirmed present in
`AX650_SUPPORTED_OPS` (and passing `partition()`'s coverage check, correctly,
since the op type genuinely is on the list) -- a plain `ConvTranspose`
(kernel 3x3, default strides/padding, upsampling 8x8 -> 10x10) fails during
real quantization with `RuntimeError("Op Execution Error: Y(TargetPlatform.
UNSPECIFIED) - inputs:['X', 'W'], outputs:['Y']")`, not the "dont support"
pattern above. This is exactly the failure mode the static heuristic
structurally cannot see (op-type presence alone says nothing about it) and
is a genuine confirmed gap in `AX650_SUPPORTED_OPS`'s "supported" claim for
at least this shape/parameter combination -- root cause (a missing required
attribute this minimal graph didn't set, a PTQ-engine limitation specific to
transposed conv, or something else) not further diagnosed here.

## Files

| file | purpose |
| --- | --- |
| `pulsar2_ops.py` | the heuristics and confirmed data: `AX650_SUPPORTED_OPS`/`AX650_MIN_OPSET` (the real, docs-scraped AX650 op list), `CPU_ONLY_OPS` (generic cross-vendor guess), the confirmed `AXERA_NPU_OP_TYPE = "neu mode"` marker, `referenced_const_data_keys()`/`missing_npu_data()`/`has_out_of_band_npu_data()` (the corruption detector), and non-standard-`domain` detection as a fallback for vendor blobs that don't follow Axera's exact convention. |
| `pulsar2_backend.py` | thin wrapper around `pulsar2_ops.py`: `coverage()`, `new_blocking_op_types()`, `stripped_npu_data()`, `unsafe_for_simplify()`, `ax650_build_risks()`. Shaped like the sibling `*_backend.py` modules for interface symmetry (`PULSAR2_AVAILABLE` is always `True` -- there's no external dependency to be missing). |
| `inspect_axmodel.py` | standalone CLI for a **real** `.axmodel` file: loads it with `onnx.load()`, then reports non-standard-domain nodes, op types outside the model's declared opset, and suspiciously large raw attributes -- what originally found the `neu mode` node in the real YOLOv8 file. |
| `models.py` | the shared `scripts/common/synthetic_models.py` suite plus `axera_npu_compiled_leaf` (real CNN `neu mode` node shape) and `axera_llm_layer_leaf` (real per-layer LLM shape: two `neu mode` nodes sharing one initializer) -- no real device needed to exercise the corruption check in CI. |
| `pulsar2_quantizer.py` | `quantize_like_pulsar2()`: a thin wrapper over `onnxsim.quantize_static(method="minmax")`, which already matches Pulsar2's real numeric convention (U8 asymmetric activations, S8 per-channel weights, MinMax calibration). `PULSAR2_QUANTIZER_AVAILABLE` reflects both `onnxruntime`'s availability and `onnxsim` itself actually being importable (a checkout with `onnxsim`'s compiled extension not yet built fails `import onnxsim`, not just the lazy `onnxruntime` import inside it -- both are caught the same way so this degrades gracefully instead of taking `pulsar2_simulator.py`'s `import` down with it). |
| `pulsar2_simulator.py` | `partition()`/`coverage()` (real `AX650_SUPPORTED_OPS` membership, no dependency beyond `onnx`) and `simulate()` (fp32-vs-INT8 estimate via `pulsar2_quantizer.py` + onnxruntime's CPU EP). Validated against real hardware -- see above. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_pulsar2_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. No `--require-*` flag or `skipped` status -- unlike the EP harnesses, this needs no vendor package or device, so it always runs. Entry point for `axera-integration.yml`'s `pulsar2-compat` job (stock runner, no Docker/device). |
| `screen_onnxmodelzoo.py` | fast, static, Docker/device-free screening of `onnxmodelzoo` models via `pulsar2_simulator`/`pulsar2_backend.ax650_build_risks()` -- run this first. |
| `pulsar2_docker.py` | real `pulsar2 build` (Docker) + `axcl_run_model` (device) wrapper: `build()` (with `profile=` for `trace.json`), `llm_build()` (the separate, ONNX-free `pulsar2 llm_build` LLM path -- see above), `build_from_hf_checkpoint()` (the hf-config+safetensors -> `onnxsim.reconstruct_hf_graph()` -> `build()` path -- see "An alternative LLM path" above), `run_on_device()`, `run_on_device_with_input()`, `force_rmtree()`. Manual/local-only -- needs a loaded Docker image. |
| `convert_onnxmodelzoo.py` | batch driver over `pulsar2_docker.py`: fetch -> onnxsim -> real `pulsar2 build` (orig + simplified, `--profile` optional) -> optional on-device bit-exact diff -> CSV. Entry point for `axera-integration.yml`'s `pulsar2-docker-convert` job -- like `amd-integration.yml`'s MIGraphX check, that job is `workflow_dispatch`-only and targets a `[self-hosted, axcl]` runner this repository doesn't provision, so it's dormant until one exists. |
| `demo_hf_llm.py` | interactive one-shot demo of `build_from_hf_checkpoint()`: compile, print the phase-timing breakdown (and `--profile`'s trace.json/Netron paths), feed one prompt, print top-5 predicted next tokens. |
| `demo_hf_llm_chat.py` | interactive chat REPL on top of the same compiled `.axmodel`, generating one token at a time and reporting real tokens/sec -- see its own docstring for the confirmed ~700ms-per-step `axcl_run_model` process/model-reload overhead this measures alongside the NPU's actual ~0.6-0.8ms compute latency. |

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
`tests/test_axera_conv_matmul_coverage.py` is the Conv/MatMul-variant
heuristic analysis above -- also unguarded, needing only `onnx`.

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
