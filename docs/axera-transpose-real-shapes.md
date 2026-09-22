# A verified template library covers all 41 real ResNet18 Transpose instances

Four independent decode attempts failed to find a shape-to-MCode formula for
a compiled `Transpose` (`docs/axera-transpose-mcode.md`, `docs/axera-
transpose-tiled.md`, `docs/axera-teng2-sqrt-blocks.md`, `docs/axera-teng2-
tiled-repeat.md`): real local structure everywhere, no rule that predicts a
new shape from ones already built. That was the wrong target for this op.
Unlike `Slice`/`Gather`, `Transpose` has **no data-dependent parameter** --
its `npu_params` is either all zero or a pure function of shape (`transpose_
tiled_params.py` decoded that part already), and once shape and perm are
fixed, so is the MCode. A Pulsar2-compiled reference for one exact `(shape,
perm)` pair is not a template to be patched -- it **is** the complete,
correct, reusable artifact for every graph instance that uses that exact
shape. This is a coverage/verification report on building and checking that
library directly, not another sweep for a formula.

## The real step has only 23 distinct shapes

The real ResNet18 training step
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, 1104 nodes) has 41
`Transpose` nodes. Grouping by `(input_shape, perm)` via `onnx.shape_
inference` gives exactly **23 distinct pairs**, with instance counts from 1
to 4 (the four `[1,64,64,9]` weight transposes, the four `[16,1,576,3136]`
activation transposes, down to eleven pairs used only once). Building and
verifying all 23 covers all 41 node-instances.

## All 23 built, all 23 exact on device

16 of the 23 already had a compiled reference from earlier sweep work
(`/home/takecheeze/npu-scratch/t_transpose_sweep/`); the other 7 (`[16,512]`
and `[512,1000]` perm `1,0`, and five mid-size activation/weight shapes) were
built the same way (Pulsar2 7.0-lite, AX650, MinMax calibration, uniform
+/-0.9 samples). **No shape failed to compile** -- unlike the stem Gather
(`docs/axera-stem-gather.md`), nothing here hit an OCM budget error, so no
chunking or decomposition was needed for any of the 23.

All 23 ran on a real AX8850 (`axcl-vm`) against `numpy.transpose`, serialized
through the shared device lock with a control run before and a health run
after. **Every one matched exactly -- `max_err = 0.0` for all 23**, including
the two largest, `[16,1,147,12544]` (29.5M output elements) and
`[16,1,576,3136]` (28.9M). This is different from every other op emitted in
this project so far (Gather, Reshape, Relu/Sqrt): those are quantized, so a
"correct" result is one within an INT8 rounding error (typically 0.003-0.02
max error observed elsewhere in this project). `Transpose` moves FP32 bytes
without a quantize/dequantize pass in between, so there is nothing to round.

| shape | perm | instances | output elements | device max err |
| --- | --- | --- | --- | --- |
| `[1,64,64,9]` | 0,2,1,3 | 4 | 36,864 | 0.0 |
| `[16,1,576,3136]` | 0,1,3,2 | 4 | 28,901,376 | 0.0 |
| `[1,512,512,9]` | 0,2,1,3 | 3 | 2,359,296 | 0.0 |
| `[16,1,4608,49]` | 0,1,3,2 | 3 | 3,612,672 | 0.0 |
| `[1,256,256,9]` | 0,2,1,3 | 3 | 589,824 | 0.0 |
| `[16,1,2304,196]` | 0,1,3,2 | 3 | 7,225,344 | 0.0 |
| `[1,128,128,9]` | 0,2,1,3 | 3 | 147,456 | 0.0 |
| `[16,1,1152,784]` | 0,1,3,2 | 3 | 14,450,688 | 0.0 |
| `[16,512]` | 1,0 | 1 | 8,192 | 0.0 |
| `[512,1000]` | 1,0 | 1 | 512,000 | 0.0 |
| `[1,512,256,9]` | 0,2,1,3 | 1 | 1,179,648 | 0.0 |
| `[16,1,2304,49]` | 0,1,3,2 | 1 | 1,806,336 | 0.0 |
| `[1,512,256,1]` | 0,2,1,3 | 1 | 131,072 | 0.0 |
| `[16,1,256,49]` | 0,1,3,2 | 1 | 200,704 | 0.0 |
| `[1,256,128,9]` | 0,2,1,3 | 1 | 294,912 | 0.0 |
| `[16,1,1152,196]` | 0,1,3,2 | 1 | 3,612,672 | 0.0 |
| `[1,256,128,1]` | 0,2,1,3 | 1 | 32,768 | 0.0 |
| `[16,1,128,196]` | 0,1,3,2 | 1 | 401,408 | 0.0 |
| `[1,128,64,9]` | 0,2,1,3 | 1 | 73,728 | 0.0 |
| `[16,1,576,784]` | 0,1,3,2 | 1 | 7,225,344 | 0.0 |
| `[1,128,64,1]` | 0,2,1,3 | 1 | 8,192 | 0.0 |
| `[16,1,64,784]` | 0,1,3,2 | 1 | 802,816 | 0.0 |
| `[16,1,147,12544]` | 0,1,3,2 | 1 | 29,503,488 | 0.0 |

**23 of 23 shapes covered, 41 of 41 node-instances covered.**

## Trustworthiness: templates were spot-checked against fresh, independent rebuilds

A committed fixture being "the compiled reference" only means something if
rebuilding the same graph reproduces it -- otherwise it is a one-off
artifact of a single Docker run, not something safe to plug into another
model's graph. Three shapes spanning the size range (`[1,64,64,9]`, the
largest `[16,1,576,3136]`, and the 2-D `[16,512]`) were rebuilt from scratch
in a separate output directory, months (well, minutes) after the originals,
and compared:

| shape | `npu_params` equal | MCode diffs | outside the known 301-325 noise window |
| --- | --- | --- | --- |
| `[1,64,64,9]` | yes | 7 | 0 |
| `[16,1,576,3136]` | yes | 9 | 0 |
| `[16,512]` | yes | 5 | 0 |

Every rebuild's `npu_params` was byte-identical and every MCode byte
difference fell inside the compiler-noise window this project has
characterized since the very first Slice emitter
(`docs/axera-memory-op-generator.md`). The three independent rebuilds are
committed alongside the templates as oracle fixtures and checked by
`tests/test_axera_transpose_real_shapes.py` on every push, with no Docker or
device needed to run that check.

## What this is, and is not

- **This is a lookup, not a generator.** `scripts/axera/transpose_real_
  shapes.py`'s `template_path(shape, perm)` returns a fixture path for one of
  the 23 known pairs and raises `ValueError` for anything else -- including
  shapes that look "close" to a known one. There is no retargeting logic
  because `Transpose` has nothing to retarget.
- **It is complete for this step's Transposes**, and only this step: a
  different training step (a different batch size, a different backbone)
  would need its own shapes built and verified the same way. Nothing here
  claims general `Transpose` coverage.
- **It says nothing about composition.** `docs/axera-compose.md` (from the
  Gather composition work) found a chain's MCode is rewritten relative to
  its parts' MCode when ops are compiled together in one graph. Whether
  dropping one of these standalone templates into the real, full training
  step's compiled graph produces a bit-identical result to what compiling
  the whole step directly would produce is untested. What is established is
  narrower but still useful: each of these 23 configurations, compiled and
  run **standalone**, computes the exact transpose Pulsar2 intends.

## Reproduction

```
scripts/axera/transpose_sweep.py build CASES.json /home/takecheeze/npu-scratch/t_transpose_sweep
```

for any of the 23 `(shape, perm)` pairs not already present. Device
verification used a small ad hoc script (not committed -- it is a thin
wrapper around `lxc file push`/`axcl_run_model` matching the pattern already
documented in `scripts/axera/pulsar2_docker.py`'s `run_on_device_with_
inputs`) run under `flock /tmp/axcl-device.lock`, with the project's
established control-run-before/health-run-after discipline.
