# TinyEngine static compatibility check

Verifies that `onnxsim`'s output stays friendly to
[**TinyEngine**](https://github.com/mit-han-lab/tinyengine), MIT HAN Lab's
source-code generator for microcontroller inference.

## TinyEngine (this integration) is not TI's edgeai/TIDL, and has no NPU
## -- and, as of March 2026, is *also* not TI's own "TinyEngine NPU"

This deserves its own section because of two separate mix-ups, one from
this project's own history and one from outside it.

First, an earlier request in this project's history conflated MIT HAN
Lab's TinyEngine with TI's TIDL accelerator: a "~2.5 GOPS NPU" figure that
actually referred to TinyEngine was initially attributed to TIDL instead
(see `scripts/edgeai/README.md`, a completely separate integration in this
repo for TI's actual C7x-MMA accelerator). **The TinyEngine this module
checks against -- `github.com/mit-han-lab/tinyengine` -- has no NPU, no
accelerator, and (at the time that history happened) nothing to do with
TI.** It is a pure-CPU code generator for ARMv7E-M (Cortex-M)
microcontrollers -- it compiles a quantized model straight into C source;
the generated C *is* the runtime, with no separate library dispatch and no
accelerator offload of any kind.

Second, and unrelated to that history: **Texas Instruments announced its
own, genuinely new hardware product also named "TinyEngine™ NPU" on March
10, 2026** (at embedded world 2026), integrated into two new MCU families
-- MSPM0G5187 (Cortex-M0+) and AM13Ex (Cortex-M33, motor control). Checked
directly (TI's own press materials, since `www.ti.com` itself is not
reachable from this repo's network policy the way `software-dl.ti.com`/
`downloads.ti.com` are -- see `scripts/edgeai/README.md`), **this is a
different technology, not this project's TinyEngine wearing a new name**:

- TI's TinyEngine NPU is a *dedicated hardware accelerator block* that
  executes neural-network layers (conv/depthwise/pointwise/transposed,
  fully-connected, average/max pooling) in parallel to the MCU's own CPU,
  claiming up to 90x lower latency and 120x lower energy per inference
  than software-only inference on a similar MCU, with <2uA standby draw.
  Its software toolchain is built on the **TVM** (Tensor Virtual Machine)
  compiler framework via TI's "Edge AI Studio".
- MIT HAN Lab's TinyEngine (this module's subject) has no hardware
  component *at all* -- it is a from-scratch C code generator with its
  own hand-written dispatch (`TfliteConvertor.py`), nothing to do with
  TVM, and the CPU itself runs the emitted C.

No licensing relationship, technology transfer, or shared codebase between
the two was found in TI's own announcement coverage -- this looks like an
independently-chosen, identical product name (both draw on the same
"tiny"/edge-AI vocabulary MIT's own TinyML work popularized), not a
rebrand or a derivative. **If a future task in this repo says
"TinyEngine NPU" and means TI's March-2026 hardware part, that is not what
`scripts/tinyengine/` checks -- this module and its whole suite are about
the MIT HAN Lab software project only.**

Real published latency numbers for *this* (MIT HAN Lab's) TinyEngine (from
its own top-level `README.md`, "Measured Results", profiled on an
STM32H743 Cortex-M7 MCU, no dedicated NPU hardware involved) confirm it is
squarely a CPU-microsecond-to-millisecond-scale regime, not the NPU-scale
performance profile TI's newer, unrelated hardware product claims:

| model | latency |
| --- | --- |
| mcunet-vww0 | 27 ms |
| mcunet-vww1 | 51 ms |
| mcunet-vww2 | 234 ms |
| mcunet-in0 | 25 ms |
| mcunet-in1 | 56 ms |
| mcunet-in2 | 280 ms |
| mcunet-in3 | 336 ms |
| mcunet-in4 | 463 ms |

(`mcunet-in4` is listed as out-of-memory on TF-Lite Micro/CMSIS-NN --
TinyEngine's own point of comparison -- which is why it appears here at
all: the whole project is framed around what fits in an MCU's memory, not
around raw throughput.)

## What TinyEngine actually ingests, and why this check can't be exact

TinyEngine has no ONNX importer. `code_generator/TfliteConvertor.py` reads
a **quantized TFLite** flatbuffer and walks its op list; there is no ONNX
path at all. This repository has no ONNX->TFLite conversion step either, so
this heuristic does what `scripts/edgeai/tidl_ops.py`'s `QOPERATOR_OPS`
check does for a different format mismatch: it maps each ONNX op type onto
the TFLite builtin op it would plausibly become, and checks *that* mapping
against TinyEngine's real dispatch table. That is an approximation of a
conversion step that never actually runs here -- "would this survive
conversion and match the whitelist", not a guarantee of either.

## The one dispatch table that actually matters

Read directly from `code_generator/TfliteConvertor.py` and
`code_generator/constant.py` in the real repo (not reconstructed from the
TFLite schema's own op enum -- see "A schema entry is not support" below
for why that distinction matters), `_handleOperator`'s if/elif chain --
run once per node in the graph -- dispatches on exactly:

- `CONV_2D` / `DEPTHWISE_CONV_2D` -- both routed through the same
  `TF_Parser.parse_conv2d`.
- `ADD`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `FULLY_CONNECTED`, `TRANSPOSE`,
  `PAD`, `RESIZE_NEAREST_NEIGHBOR`, and `MEAN` (via `parse_mead1dto2d`).
- `op_code_str in SKIP_OPs: pass` -- `constant.py` defines
  `SKIP_OPs = {"QUANTIZE", "DEQUANTIZE", "RESHAPE"}`. Nothing is generated
  for these; the node is silently dropped. Correct for `QUANTIZE`/
  `DEQUANTIZE` (TinyEngine already runs natively quantized), but
  `constant.py` itself carries a `# TODO: Handle RESHAPE during codegen`
  comment right above the `RESHAPE` entry -- TinyEngine's own authors flag
  that skipping a reshape's actual data movement isn't always safe. This
  harness surfaces that as a caveat (`tinyengine_backend.skip_op_risks`),
  not silently-clean coverage.
- **Nothing else.** The chain ends with a plain
  `else: raise NotImplementedError(f"Unsupported {op_code_str}")`. There is
  no fallback path the way TIDL/QNN/OpenVINO all have one: the generated C
  *is* the only implementation that will ever exist for that graph, so one
  unrecognized op anywhere fails the *entire* compile. `coverage()`
  returning `"partial"` here means "will not compile at all", not "some
  ops fall back to CPU" -- a real difference from every other
  `scripts/<vendor>/*_backend.py` in this repo worth internalizing before
  reading a "partial" result the TIDL/QNN way.

### `MUL` is the one op with no dispatch case of its own

`MUL` never appears in `_handleOperator`'s dispatch. It's handled entirely
through a *lookahead*: when the convertor is looking at a
`DEPTHWISE_CONV_2D` node, `checkIfRequireSEelementmult` scans forward for
the exact three-op sequence `ADD -> MUL -> MUL` and, if found, fuses the
whole triple via `TF_Parser.parse_SEelement` as a squeeze-and-excite gate.
A `Mul` anywhere else -- including the extremely common Sigmoid-gated
Swish/SiLU activation already in this repo's shared synthetic suite
(`scripts/common/synthetic_models.sigmoid_mul_swish`) -- has no dispatch
case and hits the same hard `NotImplementedError`.

This module can't re-run that exact forward lookahead (no notion of "the
op after this one" independent of the graph's own edges), so
`tinyengine_ops.se_window_mul_names` checks the same shape from the graph's
edges instead: walk forward from every `Add` node, and if it feeds a `Mul`
whose output feeds another `Mul`, mark *both* Muls (not just the trailing
one -- the real fusion consumes the whole triple) as part of a matched
window. This is a real, checkable approximation of the same three-op
chain, not the identical algorithm: the real check additionally requires
the chain to originate from a `DEPTHWISE_CONV_2D`'s own output, which this
module does not verify.

### Fused activations: `Relu`/`Clip` need to be a Conv's only consumer

TFLite's own `Conv2D`/`DepthwiseConv2D` op carries its activation as a
`FusedActivationFunction` field on the *same* node -- a `Relu`/`Relu6`
(`Clip(0, 6)` in ONNX terms) immediately following a conv is not a
separate TFLite op once real conversion tooling fuses it in, so
`_handleOperator` never sees it standalone. `is_fusable_activation`
approximates the one condition checkable without a real converter: is this
activation the conv's sole, immediate consumer? A conv output also used
elsewhere (a residual branch, for instance) keeps the raw output around
anyway, which is exactly the shape real fusion passes decline to fuse
through.

### A schema entry is not support

`code_generator/tflite/BuiltinOperator.py` -- TinyEngine's own vendored,
schema-generated bindings -- defines a real enum value for `PRELU`,
`LEAKY_RELU`, `MUL`, `RELU`, and dozens of other ops that appear nowhere in
`_handleOperator`'s dispatch. The generated bindings describe what the
*TFLite format* can represent, not what *this specific code generator*
implements. `tinyengine_ops.UNSUPPORTED_WITH_REAL_TFLITE_OPCODE` names a
few of these deliberately (`prelu_leaf` in the model suite exercises
`PRelu` specifically), since a schema-only lookup would wrongly call them
fine.

## What the static check does, per model

Same shape as `scripts/edgeai/run_tidl_compat.py`: compute the blocker set
before and after `onnxsim.simplify()`, and fail (`tinyengine_regression`)
if simplification introduced a *new* blocking op type that wasn't already
present. The consequence of that failure is different from TIDL's, though
-- see above: it means the whole compile would now fail, not that part of
the graph loses accelerator eligibility.

What it deliberately does **not** claim: that a model with zero flagged
blockers actually converts to TFLite and compiles through the real
`GenerateSourceFilesFromTFlite`, or that its attribute-level details (exact
quantization scheme, buffer/memory-planning limits for a specific MCU) are
satisfied -- only op-type mapping and shape-staticness are checked.

## Two real, worked interactions with onnxsim's own optimizations

Verified by actually running `onnxsim.simplify()` against these fixtures
and inspecting the result, not assumed:

- **`mobilenet_dw_block`: onnxsim's BN-fold *creates* TinyEngine
  eligibility.** Before simplification, each `Clip` (Relu6) sits after a
  BN `Mul`/`Add` pair rather than directly after its `Conv`, so it isn't
  eligible for the Conv-activation fusion above and this heuristic reports
  a blocker (`coverage() == "partial"`). Once onnxsim folds each BN pair
  into its preceding `Conv`'s weight and bias, the `Clip` becomes that
  conv's direct, sole consumer and the blocker clears
  (`coverage() == "full"`) -- onnxsim's fold is what makes the block
  TinyEngine-eligible in the first place, not just a node-count cleanup.
- **`se_gate_block`: the same kind of fold *destroys* TinyEngine
  eligibility.** This fixture's trailing `Mul` matches the `Add -> Mul ->
  Mul` squeeze-and-excite window and has full coverage *before*
  simplification. But onnxsim's affine-fold collapses this block's own
  `Add(bias)`/`Mul(scale)` pair straight into the preceding `Conv` (the
  same fold `mobilenet_dw_block` benefits from), which here *removes* the
  exact `Add`/`Mul` node pair the SE-gate fusion depends on -- leaving a
  standalone trailing `Mul` with nothing left to match.
  `new_blocking_op_types` correctly flags this as a regression
  (`{"Mul"}`), and `coverage()` goes from `"full"` to `"partial"`. This is
  a real tension between a generically-useful onnxsim optimization and one
  specific downstream backend's narrow fusion pattern, not a bug in either
  onnxsim or this heuristic -- see
  `tests/test_tinyengine_compat.py::test_se_gate_block_regresses_after_onnxsim_affine_fold`.
  Because of this, `se_gate_block` is deliberately excluded from
  `models.names()`'s default suite (though still directly reachable via
  `models.se_gate_block()`): the point of the default suite is "every
  model here should simplify to `status == "ok"`", and this one, by
  design, doesn't.

## Files

- `tinyengine_ops.py` -- the ONNX-op-to-TFLite-op-support mapping (direct
  dispatch, `SKIP_OPs`, the SE-gate `Add -> Mul -> Mul` window detector,
  the Conv-fused-activation check, the "real opcode, no dispatch case"
  list), plus the graph-walking helpers that apply them (including
  subgraphs).
- `tinyengine_backend.py` -- the small `coverage()`/`blockers()`/
  `new_blocking_op_types()`/`dynamic_shape_risks()`/`skip_op_risks()` API
  `worker.py` and the tests use, kept separate from `tinyengine_ops.py` for
  the same interface-symmetry reason `scripts/edgeai/tidl_backend.py` is
  split from `tidl_ops.py`.
- `models.py` -- re-exports `scripts/common/synthetic_models.py`'s shared
  suite and adds `tinyengine_dynamic_batch_leaf` (a symbolic batch
  dimension), `mobilenet_dw_block` and `se_gate_block` (the two
  onnxsim-interaction fixtures above), `standalone_mul_leaf` (a `Mul`
  outside the SE window, for contrast), and `prelu_leaf` (the "real
  opcode, still unsupported" case).
- `worker.py` -- checks one model in its own subprocess; see its docstring
  for the exact steps and status values (`ok`, `tinyengine_regression`,
  `simplify_error`).
- `run_tinyengine_compat.py` -- drives `worker.py` over the whole suite (or
  a `--models` subset) and writes a CSV report.
- `../../tests/test_tinyengine_compat.py` -- the pytest suite CI runs.
