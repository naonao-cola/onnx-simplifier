# Where a compiled training step's MCode goes, and what our emitters reach

Question: in a real compiled AX650 training step, how is the MCode split across
ops, and how much of it could the emitters in this repository produce? This
attributes MCode bytes, `npu_params` bytes and cycles to op types on a real
compiled step, checks whether a composed graph's MCode is made of its parts'
MCode, and ranks what to decode next. Reproduce with
`scripts/axera/step_attribution.py` and `scripts/axera/step_compose_probe.py`.
Nothing here emits or predicts MCode.

## What was measured, and what was not

The step is `t8-r18head2`: the ResNet18 dense-head training step (120 ONNX
nodes; BN-folded frozen backbone, trainable classifier head, in-graph Adam and a
distillation loss), 237,368 B of MCode, 11,350,569 B of `npu_params`. It was
rebuilt with Pulsar2 7.0-lite (`pulsar2 build --target_hardware AX650
--compiler.npu_perf --debug.dump_frontend_graph`, same `step.onnx`, config and
calibration data) to get `op_profile.csv` and `trace.json`. Three more compiled
graphs were profiled the same way to test the layout claims: a small
training step (`twd`), a second (`dbg`), and the forward-only ResNet18 (`inf`,
109,712 B of MCode).

Two scoping facts that change how earlier numbers should be read:

- **The 1104-node graph with 41 Gathers, 170 Reshapes and 41 Transposes was not
  attributed, because it has no compiled artifact.** `t6-r18fold/step.onnx` is the
  tap-rewritten step (`act_weight_conv_to_matmul`); the README records that it does
  not build, and no compiled model of it is on disk. Its op tally is what motivated
  the Gather/Transpose/Reshape emitters, but those ops are not what the compiled
  step is made of.
- **The compiled step has no Gather at all.** Its lowered task list holds 1
  `AxReshape` (78 cycles) and 36 `AxTranspose` tasks; the 411 `AxSlice` rows are
  DMA tile loads, not user Slices. The handoff notes for the linearized-conv step
  (`docs/axera-on-device-training-handoff.md`, "_linearize_trainable_convs") say the
  same from the other side: Pulsar2 lowered that step's Gathers to native
  `AxSlice` and no `AxGather` appears in its profile. That step's artifacts
  (`r18_b1`, 190,800 B of MCode) are not on disk and could not be rebuilt here
  (the builder needs `torch`/`timm`), so it is not attributed either.

## The MCode is five engine queues

`mcode.segments` splits every MCode blob into five stream segments. The
profile trace names five engine tracks: `conv0`, `conv1`, `teng2`, `cv3` and
`sdma4`. The segments are those engines' instruction queues:

| build | segment bytes (stream order) | tasks conv0, conv1, teng2, cv3, sdma4 | `0xa3` verbs in the last segment |
| --- | --- | --- | --- |
| `r18head2` | 23,968 / 23,648 / 72,800 / 18,016 / 94,720 | 1347, 1341, 1339, 751, 792 | **792** |
| `twd` | 64 / 32 / 18,304 / 2,400 / 14,624 | 0, 0, 316, 78, 118 | **118** |
| `dbg` | 1,952 / 1,664 / 18,432 / 4,416 / 18,912 | 36, 30, 306, 151, 145 | **145** |
| `inf` | 22,048 / 22,240 / 9,280 / 8,992 / 46,240 | 1275, 1269, 425, 379, 421 | **421** |

Evidence, in order of strength:

- The last segment is 8-byte verbs only (11,836 verb records and 32 stray bytes in
  `r18head2`), and it holds exactly one `0xa3` verb per `sdma4` task in all four
  builds: 792, 118, 145, 421. The DMA queue is the last segment.
- With no convolutions (`twd`), the first two segments shrink to 64 and 32 bytes.
  They are the convolution engines' queues.
- Over the 120 assignments of segments to engines, the one with the most stable
  bytes per task across the four builds (engines with no tasks in a build skipped)
  is conv, conv, `teng2`, `cv3`, `sdma4`. That is an inference with a modest
  margin on the middle two: swapping `teng2` and `cv3` scores 4.2 against 3.7
  (lower is more stable). It is what the estimates below assume; if it is wrong,
  the `teng2` and `cv3` shares swap. Bytes per task are stable only for the DMA
  queue (110-130 B) and `cv3` (24-31 B); `teng2` (22-60 B) and the convolution pair
  (17 B, and 55 B in `dbg`) are not, so per-task byte counts are not a constant
  per engine.
- Which of the two conv segments is `conv0` and which `conv1` is not established;
  the tool treats them as a pair.

Share of the instruction stream by engine (segment bytes over the sum of the five):

| build | conv pair | `teng2` | `cv3` | `sdma4` (DMA) |
| --- | --- | --- | --- | --- |
| `r18head2` (training step) | 20.4% | 31.2% | 7.7% | **40.6%** |
| `dbg` (training step) | 8.0% | 40.6% | 9.7% | **41.7%** |
| `twd` (training step) | 0.3% | 51.7% | 6.8% | **41.3%** |
| `inf` (forward only) | 40.7% | 8.5% | 8.3% | **42.5%** |

The DMA queue is 40.6-42.5% of the stream in every build, whatever the graph. Only
about a fifth of the training step's MCode (20.4%) is in the convolution
engines; four fifths is scheduling glue and data movement.

## Attribution by op type (`r18head2`)

Cycles are exact: the sum of `op_profile.csv` `cycles` per `type` (a sum across
engines, not wall time). Params bytes are exact up to the loads the trace shows
(below). MCode bytes are an estimate: bytes are known per engine segment, so each
segment is split over its engine's tasks by op type, by task count and by
summed task duration. The two weightings bracket the answer; the split inside
one engine is not measured.

| op type | ops | cycles | MCode, task-count | MCode, duration | `npu_params` read |
| --- | --- | --- | --- | --- | --- |
| `AxQuantizedConv` | 164 | 54.0% | 35.1% | 28.9% | 10,576,100 B |
| `AxSlice` (DMA tile loads) | 411 | 13.6% | 27.0% | 31.2% | 485,408 B |
| `AxDequantizeLinear` | 140 | 10.7% | 6.8% | 12.3% | 261,376 B |
| `assign` (state write-back) | 238 | 4.9% | 0.0%* | 0.0%* | 0 |
| `AxQuantizeLinear` | 107 | 4.9% | 4.5% | 5.6% | 0 |
| `AxMatMul` | 2 | 3.9% | 1.5% | 4.5% | 0 |
| `AxAdd` | 118 | 3.3% | 3.4% | 5.4% | 0 |
| `AxClip` | 88 | 1.5% | 4.1% | 5.2% | 0 |
| `AxMaxPool` | 32 | 1.2% | 2.4% | 2.4% | 0 |
| `AxMul` | 87 | 1.0% | 2.3% | 1.8% | 32 B |
| `AxTranspose` | 36 | 0.3% | 2.1% | 1.5% | 0 |
| `AxQuantizedReduceMean` | 16 | 0.1% | 6.0% | 0.1% | 0 |
| `AxQuantizedReduceSum` | 5 | 0.1% | 1.0% | 0.2% | 0 |
| `AxDiv`, `AxSub`, `AxSqrt`, softmax, log, neg, reshape | 95 | 0.4% | 3.4% | 1.1% | 39 B |
| unattributed transfers | 0 | 0.0% | 0.5% | 0.0% | 0 |

\* `assign` tasks are named as stores of the tensor they write, so their MCode is
counted under the op that produced it (the trace does not tag them separately).

How tasks were tied to ops: a task named `<op>_s<k>_gop_<i>_<j>` belongs to the
profile row `<op>_s<k>_gop`. `ld:`/`st:` transfers with no row of their own take
the op type embedded in their name, and parameter/input loads take the type of the
first task that reads the tensor they write. That resolves all but 0.5% of tasks
by count; the rest are runtime-value loads.

Where the two weightings differ by 3 points or more the estimate is weak:
`AxQuantizedConv` (35.1 vs 28.9), `AxDequantizeLinear` (6.8 vs 12.3), `AxSlice`
(27.0 vs 31.2), `AxMatMul` (1.5 vs 4.5) and `AxQuantizedReduceMean` (6.0 vs 0.1: 256
tiny reduce tasks). `AxSlice` (27-31%) and conv (29-35%) are the two largest under
either weighting.

**`npu_params` is fully accounted for.** The `ld:param` loads tile 11,322,955 of
11,350,569 B (99.8%) with no overlap, read as `params:[start:]` plus the length of
the destination OCM range. 93% is read for convolutions (10.58 MB of the 11.32
MB); `AxSlice` reads 485 KB and `AxDequantizeLinear` 261 KB. The 27,614
uncovered bytes are not read by any traced task.

**What a tile-per-task attribution would need, and did not get.** The DMA segment
splits cleanly into 792 groups, each ending in `0xa3`, matching the 792 `sdma4`
tasks in count. If the groups were aligned with the tasks in trace order, bytes
could be attributed per task. Group length (10-41 eight-byte records, 80-328 B)
varies within one task class (`ld:param` runs 12-26 records), and it does not
correlate with any of three task features under trace-index or start-time order
(Spearman 0.08, -0.17, -0.15; shuffled order is no worse). The alignment is
unverified, so no per-task attribution is claimed.

## What our emitters reach

No emitter synthesizes MCode. Each one rewrites parameters of a compiler-built
reference of the same graph; the MCode stays the reference's. "Retarget" and
"metadata" below mean that, and only for the shapes the emitter measured.

| op type | emitter | measured scope | this step's shapes | in scope |
| --- | --- | --- | --- | --- |
| `AxQuantizedConv` | `emitter.py` weight table (learned per shape from k builds); `tiny_emit.py` site-A scales | Conv fixtures at 1-8 channels, 8x8; `conv64_k5_d2`, `conv128_k7_d12` (1-D, dilated) | 3x3/7x7/1x1 convs, 3-512 channels, 7-112 spatial | none |
| `AxMatMul` | `tiny_emit.py` A-scale (same form only); weight table via `emitter.py` | `matmul` 1x8x8 .. 4x16x8; `gemm` with M=1, K 100-1200, N 32-2000 (closest: `gemm_1x512x1000`, constant weights) | `[16,512]x[512,1000]` and `[512,16]x[16,1000]`, live weights | none |
| `AxAdd`, `AxMul`, `AxSub`, `AxDiv` | `tiny_emit.py` scale patchers | two-live-input `[1,16]`, `[1,32]`, `[1,64]`, `[2,32]` | `[16,1000]`, `[1000]`, `[1]`, `[16,1]` | none |
| `AxQuantizedNeg` | `tiny_emit.py` output scale | `[1,8]`-class | `[1,1]` | none known |
| `AxSlice` | `memory_emit.py` static Slice | float32 `[1,8]`, axis 1, steps 1-4 | DMA tile loads of `[1000,512]`, `[16,56,56,64]`, `[16,7,7,512]`, ... | none |
| `AxReshape` | `reshape_emit.py` fused-Reshape relabel | Reshapes Pulsar2 fuses into a Relu neighbour | one `[1,1000]` reshape, no Relu neighbour | none |
| `AxTranspose` | none | not predictable (`axera-transpose-mcode.md`) | 36 tasks, mostly `[1,3,112,224]` | -- |
| `AxDequantizeLinear`, `AxQuantizeLinear`, `AxClip`, `AxMaxPool`, reductions, softmax | none | -- | -- | -- |

Fractions of the step, by the table's op types:

| | op type has an emitter | this step's shapes inside an emitter's measured scope |
| --- | --- | --- |
| MCode (task-count estimate) | 71.3% | 0% |
| MCode (duration estimate) | 72.5% | 0% |
| cycles | 76.0% | 0% |
| `npu_params` | 97.7% | 0% |

The left column is an upper bound that credits any op type an emitter exists
for; the right column is the honest one, because no shape in this step lies
inside any measured family, and even for an op type in scope every emitter needs
a compiler-built reference of the same graph. Emitting this step without a
compile is 0% today. The 97.7% for `npu_params` says the weight table, which is
99.8% of the file, is the part `emitter.py`'s approach is built for; it does not
address the 237 KB of MCode.

## Does a chain's MCode contain its parts' MCode?

`docs/axera-compose.md` found that chains are rewritten. This measures how much,
per engine, on real shapes. The ladder (seven Pulsar2 builds, batch 16, 512
channels, 7x7): `conv1` (3x3, 512 to 512), `relu1`, `conv_relu`, `conv_relu_conv`,
and the classifier head `mm` (`[16,512] x [512,1000]`, both operands live),
`add1` (bias) and `mm_add`. For each engine segment, "chain content found in
parts" is the fraction of the chain segment's 12-byte windows (windows with fewer
than four distinct byte values dropped, so padding does not count) that occur in
the parts' segments. It is exact-byte matching and therefore a lower bound on
similarity: any changed address or immediate breaks a window.

| chain vs parts | conv0/conv1 bytes: chain / sum of parts | `teng2` | `cv3` | `sdma4` |
| --- | --- | --- | --- | --- |
| `conv_relu` vs `conv1` + `relu1` | 2656 / 2720, 2656 / 2688 | 1664 / 3136 | 800 / 1056 | 4128 / 4608 |
| `conv_relu_conv` vs `conv1` x2 + `relu1` | 4832 / 5376, 4832 / 5344 | 1664 / 4800 | 1280 / 1856 | 7904 / 8736 |
| `mm_add` vs `mm` + `add1` | 1536 / 1600, 1312 / 1344 | 1888 / 3008 | 448 / 704 | 576 / 832 |

Standalone program found inside the chain (same 12-byte windows, other direction):

| standalone in chain | conv | `teng2` | `cv3` | `sdma4` |
| --- | --- | --- | --- | --- |
| `conv1` in `conv_relu` | 99-100% | 91% | 100% | 100% |
| `conv1` in `conv_relu_conv` | 76-79% | 85% | 79% | 97% |
| `mm` in `mm_add` | 98% | 75% | 51% | 91% |
| `relu1` in `conv_relu` | 0% | 55% | 11% | 71% |
| `add1` in `mm_add` | 0% | 33% | 44% | 84% |

Reading it:

1. **Relu fuses into the conv.** `conv_relu` compiles to exactly the size of
   `conv1` alone (12,656 B), and a standalone Relu program (3,048 B) is not a
   piece of it: its conv-engine content is 0% present. A standalone-op template
   for Relu would never be used here.
2. **Compute-engine programs carry over almost verbatim and add up.** The
   standalone conv program is 99-100% present in `conv_relu` and 76-79% in the
   two-conv chain (the second conv's addresses differ); `mm` is 98% present in
   `mm_add`. Conv-engine bytes are 0.90-0.99 of the parts' sum.
3. **Glue is rewritten.** `teng2` and `cv3` shrink to 0.35-0.76 of the parts' sum
   because the chain keeps intermediates on chip and shares quantize/load/store
   glue that each standalone build carries for itself; the standalone Relu and Add
   programs are only 11-55% present on `teng2`/`cv3`. DMA (`sdma4`) is 0.69-0.90 of
   the sum.
4. **Consequence for assembling a step from templates.** Even with perfect
   per-op templates and address patching for the compute engines, those queues are
   20% of the step's MCode (`r18head2`); the other 80% is exactly the part the
   compiler re-plans per graph. Whole-step emission needs the DMA and `teng2`
   schedules, not more compute-op templates.

Limits of this test: one chain per family, one shape each, random weights, MinMax
calibration; exact-window matching under-counts similarity; conv output segment
sizes have a small compiler-noise component (rebuilds of one graph differ by a few
bytes in the 301-325 window in small graphs, and by 1000-1500 bytes in composed
graphs per `axera-compose.md`), which this did not separate from real differences.

## What to decode next

By share of the instruction stream, which is the same order across all four
builds:

1. **The DMA queue (`sdma4`), 40.6-42.5% in every build.** It is the largest and
   the simplest structurally: 8-byte verbs only, one `0xa3`-terminated group per
   DMA task, 10-41 records per group. The memory-op work in this repository
   (Slice, Gather, Reshape, Transpose) is the small-scale version of this queue,
   since those ops lower to DMA descriptors. The blocker is the missing link
   from a group to its task. A controlled experiment is the way in: single-DMA
   graphs (a `ld:param` of N bytes, one tile store, one slice) swept over size,
   varying one quantity at a time, to see which verbs and fields encode length,
   stride and address.
2. **`teng2` (31.2% in `r18head2`, 40.6-51.7% in the two small steps).** The
   elementwise, quantize and clip programs: 1,339 tasks, about 54 B each in
   training graphs (22 B in the forward-only one), so per-op cost is not
   constant. `AxDequantizeLinear`, `AxQuantizeLinear`, `AxClip` and `AxAdd` are
   the op types worth isolating, in that order by estimated share.
3. **The convolution queues (20.4%).** The most template-friendly (near-additive,
   verbatim across chains), and `emitter.py` already writes their weights for a
   learned shape. The gap is the MCode itself: an emitted step still needs a
   reference build per conv shape.
4. **`cv3` (7.7%).** 751 tasks at 24 B; smallest and the least characterized.

By op type (estimated bytes): conv 29-35%, `AxSlice` 27-31%, `AxDequantizeLinear`
7-12%, `AxQuantizeLinear` 4.5-5.6%, `AxClip` 4-5%, `AxAdd` 3-5%, `AxMatMul`
1.5-4.5%. In this step `AxTranspose` is 1.5-2.1% and Reshape and Gather are
about 0%, so the Transpose and Reshape decode is not where this step's bytes
are. The earlier profile of the linearized-conv step
(`docs/axera-on-device-training-handoff.md`: `AxTranspose` + `AxSlice` at 26.3% of
cycles before linearization) suggests memory ops matter more when the conv backward
is in the graph; that cannot be measured until that step is rebuilt (needs
`torch`/`timm`) and run through `step_attribution.py`.

## What is not established

- MCode bytes by op type are estimates (two weightings, up to 6 points apart
  on the large types); only engine-level bytes are exact.
- Which conv segment is `conv0` and which is `conv1`.
- Any per-task attribution inside a segment (see the DMA group test above).
- Anything about the conv-backward step with Gathers, for lack of a compiled
  artifact.
- Emission on a device: nothing in this attribution was run on hardware.
- The cycle column sums per-op cycles across engines; it is not wall-clock time
  (the trace's per-engine busy times are 17.6K for each conv engine, 17.8K for
  `teng2`, 7.1K for `sdma4` and 7.3K for `cv3` in `r18head2`).

## Reproduce

```
# real step, profiled (scratch: /home/takecheeze/npu-scratch/t_step_attr/r18head2)
docker run --rm -v $WORK:/data pulsar2:7.0-lite pulsar2 build --target_hardware AX650 \
  --input step.onnx --output_dir out_prof --config config/step.json \
  --compiler.npu_perf --debug.dump_frontend_graph
python scripts/axera/step_attribution.py $WORK/out_prof/compiled.axmodel

# composition ladder
python scripts/axera/step_compose_probe.py build $LADDER
python scripts/axera/step_compose_probe.py compare $LADDER
```
