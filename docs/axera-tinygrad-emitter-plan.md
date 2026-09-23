# A tinygrad-style code generator for AX650/AX8850: architecture plan

The macro-goal recorded in `tests/test_axera_generator_progress_stocktake.py` is
"a complete tinygrad code based code generator" for the AX650 NPU. This document
maps what tinygrad's code generator actually is (0.14.0, read from source) onto
what this repository has decoded about AX650 compiled models, compares two
architectures, and lays out a staged path to ResNet18 training-step coverage.
It is a plan: no device runs, no Pulsar2 builds, no new code.

Everything cited here comes from this repository's own builds, public
repositories, and black-box device testing. Material derived from
circumventing license or obfuscation protections was not used.

## Recommendation in one paragraph

Build **architecture B (template + patch) now, at graph granularity**, and grow
toward **architecture A (a real MCode renderer) one engine at a time** as
register maps are decoded. B is the only model that works today: every emitter
in `scripts/axera/` retargets a Pulsar2-compiled template, and an external
llama.cpp AX8850 backend reached production speed the same way. B's seam in
tinygrad is **not** the per-kernel `Renderer`: Pulsar2 fuses a whole subgraph
into one program and composition rewrites the MCode
(`docs/axera-compose.md`, `docs/axera-transpose-compose-real.md`,
`docs/axera-conv-compose-real.md`, `docs/axera-stem-gather-reshape.md`), so the
unit of compilation has to be the fused subgraph -- tinygrad's graph/JIT level,
with Pulsar2 behind a cached `Compiler`. A lives at the `ISARenderer` seam
(`Renderer.asm` -> `Ops.BINARY`), but only becomes reachable once the per-engine
register maps, the short-unit encoding, and the cross-engine sync/DMA scheme are
decoded.

## 1. What tinygrad 0.14.0 is (read from source)

The installed package (`uv run --no-project --with tinygrad`, version 0.14.0):

| Stage | Where | What it produces |
| --- | --- | --- |
| Tensor ops -> UOp graph | `uop/__init__.py` (`Ops`), `uop/ops.py` | a DAG of `UOp`s; `Ops.CALL` is an opaque kernel invocation, `Ops.FUNCTION` a gradient-able body |
| Scheduling / kernel split | `schedule/rangeify.py` (`get_kernel_graph`, `bufferize_to_store`, `split_reduceop`), `schedule/__init__.py` (`create_schedule`, `create_linear_with_vars`) | one `Ops.LINEAR` of `Ops.CALL`s; each call's body is an `Ops.SINK` kernel AST (or an `Ops.COPY`) |
| Kernel optimization | `codegen/opt/` (`heuristic.py`, `search.py` BEAM, `tc.py`) | applied opts on the SINK |
| Lowering | `codegen/__init__.py` `full_rewrite_to_sink` | a lowered SINK the renderer accepts |
| Linearize | `codegen/__init__.py` `do_linearize` -> `Ops.LINEAR`; `ISARenderer`s also run register allocation here (`codegen/late/regalloc.py`) | a flat instruction list |
| Render text | `do_render` -> `Renderer.render` -> `Ops.SOURCE` (`renderer/cstyle.py`, `ptx.py`, `wgsl.py`, `llvmir.py`) | program text |
| Or assemble | `do_assemble` -> `Renderer.asm` -> `Ops.SOURCE` + `Ops.BINARY` (`renderer/isa/__init__.py` `ISARenderer`, `renderer/isa/x86.py` `X86Renderer`) | machine code directly, no external compiler |
| Compile | `do_compile` -> `Compiler.compile_cached` (`device.py` `Compiler`, disk-cached by source) | bytes |
| Package | `device.py` `TinyELF` (lib, name, target, signature) | the loadable unit |
| Run | `device.py` `Compiled(device, Allocator, renderers, Program)`; `engine/realize.py` `get_runtime` caches `Device[d].runtime(elf)`; `Program.__call__(*bufs, ...)` | execution |
| Memory | `device.py` `Allocator` (`_alloc/_free/_copyin/_copyout`), `LRUAllocator`, `BufferSpec` | device buffers |
| Whole-graph batching | `engine/realize.py` `get_graph_runtime` (`Ops.CUSTOM_FUNCTION` "graph"), `engine/jit.py` | one launch for many kernels |

The DSP backend (`runtime/ops_dsp.py`) is the closest precedent for an unusual
target. `DSPRenderer` subclasses the C renderer, `DSPCompiler` shells out to a
toolchain, `DSPAllocator` manages shared buffers, and `DSPProgram` submits over
FastRPC. It shows that a backend can keep tinygrad's upper layers unchanged and
replace only renderer, compiler, allocator, and program.

## 2. Prior tinygrad work in this repository

- **`onnxsim/webgpu_tinygrad_codegen.py`** (`docs/webgpu-kernel-dispatch.md`)
  builds a `Tensor` graph, calls `schedule_linear()`, picks the `Ops.CALL`s
  whose body is an `Ops.SINK`, and renders them with `WGSLRenderer` directly,
  without touching the device. The same trick applies here: tinygrad's
  front half runs with no AX650 device present.
- **`onnxsim/tinygrad_uop_export.py`** (`docs/tinygrad-uop-onnx-export.md`)
  round-trips a UOp graph through ONNX under a private domain. It is an
  inspection format, not something Pulsar2 accepts.
- **`scripts/android/tinygrad_hexagon_bridge/`** kept tinygrad's kernel and
  swapped only the transport (TVM's RPC) when tinygrad's own runtime could not
  reach the device. The AX650 equivalent is keeping tinygrad's front half and
  using AXCL (`axcl_run_model`, `axclrtEngineLoad*`) as the transport.

## 3. What an AX650 compiled model is (what this repository has decoded)

- **Compile unit.** Pulsar2 compiles a whole ONNX subgraph into one `neu mode`
  node. Fusion rewrites the MCode: a standalone template's bytes do not
  reappear inside a composed build (sources above).
- **Container.** `.axmodel` = ONNX `ModelProto` with `npu_graph_info`,
  `outputs_info`, `npu_params` (the Wbt weight table), `npu_dyn_params`, and
  the `*_neu` MCode blob. `libax_engine.a`'s DWARF names the `extra_data`
  schema (`docs/axera-bsp-mining.md`).
- **MCode = per-engine queues.** Segments are engine instruction queues
  (`docs/axera-step-attribution.md`: conv, conv, `teng2`, `cv3`, `sdma4`).
  The device log names the EU map: CONV EU0-5, CV EU6-8, TENG EU9-11, MAU EU12,
  SDMA EU13-15 (`docs/axera-teng2-register-decode.md`, #1831).
- **Records.** 8-byte verb records are register writes
  `[verb][unit][reg lo][reg hi][value32]` (`docs/axera-pulsar-v1-mining.md`,
  #1818). The same stream also carries compressed variable-length short units
  that AX620A does not have. `mcode.py` tokenizes both and round-trips
  (`encode(decode(m)) == m`) but does not evaluate them.
- **Register semantics.** Relu's `teng2`: `0x0f60/0x0f70/0x0f80` hold
  `1/input_scale` then `output_scale`, one register per output lane;
  `0x1c20/0x1c30/0x1c40` hold the per-lane clamp floor; the zero point is a
  variable-length byte that reflows the stream (#1831). Add's `teng2` touches
  `0x03b0/0x03c0/0x03d0` and its `cv3` writes `0x0170`
  (`docs/axera-teng2-fault-injection-add.md`). Faults split into AXI address
  errors vs structural errors, so operand bytes are addresses
  (`docs/axera-teng2-fault-injection-relu.md`).
- **`npu_params` tables.** Elementwise DMA tile tables are predicted
  (`dma_tile_predict.py`, `add_tile_predict.py`,
  `elementwise_two_input_tile_predict.py`), as is part of the tiled Transpose
  table (`transpose_tiled_params.py`). Conv weight codes follow a learned bit
  permutation (`emitter.py`, `conv_weight_learn.py`); the requant scaffold is
  per output-channel tile with `tile_width = 36864 // (Cin*K*K)`
  (`conv_scaffold_arithmetic.py`, `docs/axera-conv-256to512-tiled-fix.md`).
- **Runtime.** The AXCL host is an RPC proxy; the card parses the model
  (`docs/axera-axcl-runtime-mining.md`). The device kernel log gives
  per-fault EU and cause (`npu_fault_log.py`).

## 4. Layer mapping

| tinygrad layer | AX650 counterpart | Status |
| --- | --- | --- |
| UOp graph (`Ops`) | ONNX graph (quantized by Pulsar2) | DECODED as a format; quantization is Pulsar2's (`docs/axera-quantizer-reverse-engineering.md`) |
| Schedule / kernel split | Pulsar2's fused subgraph + per-EU task split | UNKNOWN (Pulsar2 decides; `docs/axera-step-attribution.md` gives task counts only) |
| Buffer placement | OCM allocation (3,141,632 B budget), DDR swap, `_ocm_base` RTV | PARTIAL (budget known; allocator not decoded) |
| Linearized program | per-EU register-write streams + short units | PARTIAL (record grammar decoded; a handful of TENG registers decoded; most registers UNKNOWN) |
| Sync / barriers | `a2`/`a3` verbs across queues | PARTIAL (verbs identified; cross-EU scheme not decoded) |
| DMA descriptors | `sdma4` task records, tile tables in `npu_params` | PARTIAL (tile tables predicted for several ops; SDMA record fields UNKNOWN) |
| Renderer output (`Ops.SOURCE`/`BINARY`) | MCode blob: segments + tail/segment table | PARTIAL (`mcode.segments`/`tail_tables`; per-segment word counts known) |
| Constants / weights | `npu_params` (Wbt) | PARTIAL (Conv weight codes 11/11 real shapes; scaffold layout; elementwise tile tables) |
| `TinyELF` packaging | `.axmodel` container | DECODED (#1820 schema names) |
| `Compiler` | Pulsar2 `build` (cached) | available (black box) |
| `Allocator` / `Program` | AXCL buffers + `axclrtEngineLoad*`/`axcl_run_model` | available (public API) |

## 5. Two architectures

### A: a true MCode renderer

An `ISARenderer` subclass whose `asm()` emits per-EU register-write streams,
sync records, and DMA descriptors, plus a packer that writes the segment table
and `npu_params`. It needs, in order of how much they block:

1. **Per-engine register maps** for TENG, CV, CONV, and SDMA: what every
   register in a real program means. Today a few TENG registers are known.
2. **The short-unit encoding.** The compressed variable-length units (the
   zero-point byte among them) must be generated, not just tokenized.
3. **The scheduling layer:** cross-EU sync/barriers, SDMA descriptor fields,
   OCM allocation, tiling. Tinygrad's scheduler would have to produce a
   multi-engine program, which it does not model today.

A also has to match Pulsar2's fusion to be competitive; per-kernel emission
loses the fusion that makes compiled steps fast.

### B: template + patch

`Compiler` = Pulsar2 with a cache keyed by (op signature, shapes, dtypes,
calibration class, toolchain version). `Program` loads the cached `.axmodel`
after applying edits that change values without changing structure: weight
codes and Gather indices in `npu_params`, calibration-dependent values, and
constants. Every emitter in `scripts/axera/` is an instance of this, and the
external llama.cpp AX8850 backend used the same strategy.

B's limits are exactly the decode frontier: an edit is only valid where the
field is decoded and fixed width, and a new shape still needs one Pulsar2 build.

### Where the seams go

| | Renderer | Compiler | Program / runtime |
| --- | --- | --- | --- |
| B | not used per kernel; the backend exports the fused subgraph to ONNX | Pulsar2 build, disk-cached (`Compiler.compile_cached` semantics) | AXCL load/run of the patched `.axmodel` |
| A | `ISARenderer.asm` -> MCode per engine | none (asm produces bytes) | same AXCL transport |

Because the compile unit is the fused subgraph, B should hook in at the
graph/JIT level (a whole captured schedule becomes one `Ops.CUSTOM_FUNCTION`
"graph" with a `graph` runtime), not at the per-kernel `Renderer`.

## 6. Interface sketch

Signatures only; no implementation is proposed in this PR.

```python
@dataclass(frozen=True)
class TemplateKey:
    op_signature: str          # canonical ONNX subgraph hash
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    calibration_class: str     # e.g. "sym", "asym-zp-2byte"
    toolchain: str             # e.g. "pulsar2:7.0-lite"

class TemplateCache:
    def get_or_build(self, key: TemplateKey, onnx_bytes: bytes) -> Path: ...

class AXCompiler(tinygrad.device.Compiler):      # B: Pulsar2 behind the cache
    def compile(self, src: str) -> bytes: ...    # src = serialized fused subgraph

class Edit(Protocol):
    def validate(self, model: onnx.ModelProto) -> None: ...   # refuse unless decoded
    def apply(self, model: onnx.ModelProto) -> None: ...

class EditSet:  # composes existing emitters: memory_emit, emitter, conv_*, tile predictors
    edits: list[Edit]

class AXAllocator(tinygrad.device.Allocator): ...   # AXCL device buffers
class AXProgram(tinygrad.device.Program): ...       # load edited .axmodel, run via AXCL
class AXDevice(tinygrad.device.Compiled): ...       # wires the above
```

`Edit.validate` is where the safety rules live: refuse any field that is not
decoded for that exact template, and refuse any edit that would change a
variable-length unit's width.

## 7. ResNet18 training-step coverage plan

Op counts from `/home/takecheeze/npu-scratch/t6-r18fold/step.onnx` (1104 nodes).
Caveat for every row: trainable weights are graph inputs, i.e. runtime state,
so per-step weight patching is not the training path; the useful edits are
recalibration, indices, and constants.

| Op | Count | Today | Missing | Next decode work |
| --- | --- | --- | --- | --- |
| Mul | 397 | tile table (#1809) | scale registers, compute body | binary-op register semantics (in flight) |
| Reshape | 170 | 17 free bias flattens (`reshape_emit.py`); narrow weight-fold family (#1744) | real DMA programs | affine fields from calibration isolation (#1797) |
| Add | 144 | tile table + Q15 header (#1756) | scale registers, `cv3` role | registers `0x03b0..`, `0x0170` (#1823) |
| Div | 52 | tile table (#1809) | as Mul | as Mul |
| Sub | 46 | tile table (#1809) | as Mul | as Mul |
| ReduceSum | 44 | input tile table matches elementwise rule (#1806) | compute segment | register census |
| Sqrt | 42 | 1-D toy emitter (#1750); cluster extrapolation (#1757, #1805) | scale registers at real shapes | register census |
| MatMul | 41 | frozen-weight bit permutation (`emitter.py`); per-tensor scalar fields (#1794) | live-weight case is runtime state | none needed for weights; scales via recalibration |
| Transpose | 41 | 41/41 standalone templates (#1758); tiled table subset (#1749) | fuses under composition (#1763) | graph-level templates |
| Gather | 41 | 19/19 shapes index retargeting (`memory_emit.py`), calibration recipe (#1761) | stem with Reshape neighbor re-lays indices (#1802) | graph-level templates |
| Conv | 20 | weight codes 11/11 real shapes; full pipeline on several | live weights are runtime state | recalibration path |
| Cast | 19 | none | everything | register census |
| Greater | 18 | none | everything | register census |
| Relu | 17 | scale + clamp registers (#1831) | zero-point width changes | scale-edit emitter (in flight) |
| Others | 12 | none (Softmax 3, Log 2, Neg 2, MaxPool 1, ReduceMean 1, Squeeze 1, Gemm 1, Less 1) | everything | after the above |

The coverage that matters for training is graph-level: one cached template for
the whole step, plus a recalibration edit. Whether a compiled step can be
recalibrated by editing in place is being tested in the in-flight step
recalibration work.

## 8. Phased roadmap (each milestone device-verifiable)

1. **B skeleton.** `AXDevice` with `AXAllocator`/`AXProgram` over AXCL and
   `AXCompiler` = cached Pulsar2, driven from a tinygrad `Tensor` graph exported
   to ONNX at the JIT boundary. Milestone: a tinygrad Relu/Add graph runs on the
   card and matches numpy.
2. **Edits wired in.** Existing emitters exposed as `Edit`s with `validate`.
   Milestone: Gather index retargeting and Conv weight refresh through the
   tinygrad path match native builds on device.
3. **Recalibration.** Calibration-dependent values edited in a cached step.
   Milestone: an edited step matches a native rebuild with the new calibration.
4. **Register maps, engine by engine.** Census plus device confirmation per op.
   Milestone per engine: an edit to a decoded register behaves as predicted.
5. **Renderer (A), smallest engine first.** Emit one engine's queue for a
   single op and splice it into an otherwise Pulsar2-built program. Milestone:
   byte-exact against Pulsar2 for that queue, then correct on device.

## 9. Risks

- **Fusion rewrites MCode.** Per-op templates do not compose; graph-level
  templates are required, which means one Pulsar2 build per graph shape.
- **Zero-point reflow.** A zero point is a variable-length byte, so edits that
  change its width are not value edits.
- **Firmware deaths.** Some bad edits kill the card's firmware (a scale
  register's companion record did, #1831). Recovery is the full guest module
  reload in `scripts/axera/vm/README.md`; every device milestone needs control
  and health runs.
- **Toolchain drift.** Templates are tied to a Pulsar2 version; the vendor's own
  models come from other compiler generations (#1820).
- **tinygrad API drift.** 0.14.0's scheduler output (`schedule_linear`,
  `Ops.CALL`) has changed between versions before; the backend should depend on
  as little of it as possible.
