# Running the ResNet18 training step with its covered nodes on the AX650

`coverage_report` counts the nodes of the training step
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, 1,104 nodes, batch 16,
distillation loss plus Adam) that our emitters can produce at the step's
predicted calibration (`docs/axera-step-real-calibration.md`). This note is
about actually running the step that way, on the card, and measuring what
comes out.

## The pieces

- `scripts/axera/vm/axcl_batch_runner.c` is a persistent runner built inside
  the LXD guest. One `axclInit`, one device context, and line commands on
  stdin: `LOAD` a model, `RUN` it on input files, `UNLOAD` it. Tensors travel
  as raw files on the guest's virtiofs share (`/mnt/share`), not through the
  pipe. Loading a model takes about 13 ms, and running the Relu health model
  about 25 ms end to end. `axcl_run_model` through the harness instead costs
  a process, a runtime init and an `lxc file push` per call.
- `scripts/axera/axcl_session.py` is the host side (`AXSession`). It holds
  `/tmp/axcl-device.lock` for the session, so other device work queues behind
  it. It also provides `health_check`: the native `x128,y128` Relu template
  on [-1, 1] data, which must come back within 1 LSB.
- `scripts/axera/step_runner.py` turns the step into segments and runs them.
  - Every node `plan_at_calibration` covers becomes an NPU segment: its
    template, retargeted to the predicted calibration.
  - A covered live-operand MatMul runs its whole chain template.
  - Greater/Less -> Cast runs as its pair template.
  - A bias-flatten Reshape runs fused with its ReduceSum (#1908).
  - Everything else runs on the host, one node at a time in onnxruntime.
  - Segments exchange float32 tensors, because every template quantizes and
    dequantizes internally.
  - Every NPU segment is also simulated on the host on the same inputs:
    inputs fake-quantized at the predicted parameters, the float ops, and
    the output fake-quantized. The runner then compares device against
    simulation in output LSBs, and both against float.
  - With `--health-every 1`, the default, the health model runs after every
    device segment, so a bad model is caught at the segment that broke the
    card.
- `tinygrad_ax_backend.register_ax_device()` makes `"AX"` a tinygrad device.
  - `AXAllocator` gives host-staged buffers. AXCL binds device memory to one
    loaded model's IO, so buffers are staged per call.
  - `AXProgram` loads `AXCompiler` output, or any emitted `.axmodel`, into
    the session and runs it on the card.
  - `tests/test_axera_step_runner.py` runs a Relu (an `AXCompiler` request)
    and the step's fc dX MatMul chain through tinygrad `Buffer("AX", ...)`
    objects on the device.
  - tinygrad's own scheduler cannot target the device: no renderer maps
    UOps to templates.

## What running it found

All runs are on 2026-09-24, on the AX8850 through `axcl-vm`, with the step's
first batch (`step1_ref.pkl`) and the committed calibration
(`fixtures/step_calibration/resnet18_step_calibration.json.gz`).

1. **Emitters wrote MCode without updating its size, and that took the
   card down.**
   - `misc_op_record_emit.emit_model`, `reshape_record_emit.emit_step_reshape`,
     `elementwise_scale_emit._write` and `ElementwiseScaleEdit` replaced the
     MCode initializer's bytes but not its `dims`.
   - A zero-point move re-encodes the stream to a different length:
     ReduceSum_62, `[16,1,512,4608]`, went from 120,024 to 120,088 bytes.
   - The runtime reads the size from `dims`. A standalone load fails with
     `0x80300709`. In the first whole-step run it wedged the card instead:
     firmware dead, full stack reload needed.
   - All four emitters now go through `step_recalibrate.with_mcode`.
     ReduceSum_62 then matches its simulation exactly.
   - The native template of the same shape ran fine throughout, which is
     what isolated the fault to the emitted model.
2. **The step Reshape templates compute a Relu (#1891).**
   - The step Reshape templates are `Reshape -> Relu` builds, and the
     emitted model applies the Relu.
   - Reshape_42, on a signed input: 40% of the elements are off by up to
     164 LSB, and the error relative to float is 79%.
   - 100 of the 119 Reshape segments take an input whose calibration range
     is negative. The runner marks them `unsafe` and keeps them on the host.
     The other 19 run on the NPU and match.
   - So `coverage_report`'s Reshape count overstates what is correct by
     those 100 nodes.
3. **Gather templates kept their own calibration.**
   - `GatherIndexEdit` moves the indices but leaves the template's
     `(1/s, s, zp)`. On the device, Gather_57 differed from float at 26% of
     its elements.
   - A Gather is passive, so the runner now retargets it like a Reshape.
     `reshape_record_emit.retarget_scale` gained `zp_regs=GATHER_ZP_REGS`,
     since a Gather writes only 0x1b10/0x1a90. It also now accepts an `s`
     lane that is not exactly `f32(1 / (1/s))`, as a Gather template stores
     141.6667 next to 0.00705882.
   - All 27 Gather segments are now within 1.83 LSB of their simulation.
4. **Five segments disagree with their simulation. Validation pass:
   `--mode npu`, safe segments only, 275 segments, 366 nodes.**
   - 270 of 275 are within 2 LSB.
   - The failures:

     | Segment | What the device did |
     |---|---|
     | `Add_976` (x128,y128,z128) | up to 115 LSB off |
     | `Reshape_371` (fused chain `ReduceSum:16x1x64x3136:axes0,3:k0:reshape64` at s_x = 1.0e-5, s_y = 4.6e-3) | up to 144 LSB off |
     | `Reshape_416` (same chain, s_x = 1.4e-5, s_y = 5.9e-3) | device fault `0x8030070C`; the health check passed right after |
     | `Log_3`, `Log_10` | 17-89 LSB off: the emitter did not move Log's `s_y` lanes. #1910 fixed this; on current master both are exact (0 LSB) |

   - Rechecked on current master: Add_976 is still 115 LSB off, and
     Reshape_371 is still 111 LSB off. Both are emitter defects to chase:
     the Add `x128,y128,z128` retarget at these scales, and the
     `...:reshape64` fused chain at an `s_y/s_x` of about 460.
     `--validated <report>` keeps failing segments on the host in later
     runs.
5. **The loss head cannot be uint8.** The device matches the simulation
   there; the damage is from quantization itself.
   - With Softmax on the NPU, the student's probabilities, which are about
     1e-3 for 1,000 classes, quantize at `s = 3.5e-3` to mostly 0. The
     `softmax - target` gradient is then gone.
   - With every safe segment on the NPU, the median gradient cosine to
     float is -0.62. In a run with only the loss-head/misc segments on the
     NPU, keeping just Softmax on the host brings it back to 0.998.
6. **The optimizer update cannot be uint8 either.**
   - With Adam's `m`, `v`, `sqrt(v)` and `m / (sqrt(v) + eps)` on the NPU,
     the weight update is off by a factor of about 10^6.
   - A per-tensor MinMax range cannot hold `v` of about 1e-8 next to its
     maximum.
   - `--host-optimizer` keeps every node after the gradients in float.

## The numbers

"Grad" is each weight's gradient, the tensor the Adam update consumes,
against a float run of the same batch. "Update" is `w' - w` against the float
step. Medians are over the 42 trainable tensors. Device time is the sum of
`axclrtEngineExecute` wall times inside the guest.

| Run | NPU segments / nodes | Loss (float 17.0582) | Grad cos median / min | Grad rel. err median | Update cos median |
|---|---|---|---|---|---|
| every safe segment (validation pass) | 275 / 366 | 15.864 | -0.62 / -0.78 | 4.4 | -0.11 |
| minus failing segments and the loss head | 265 / 354 | 17.091 | 0.9935 / 0.19 | 0.128 | 0.46 |
| ... and the optimizer on the host | **180 / 269** | **17.091** | **0.9935 / 0.19** | **0.128** | **0.80** |

- In the last run, every one of the 180 device segments was within 2 LSB
  of its simulation, and the health check passed before the first and
  after each one (362 device runs in all).
- The two lowest gradient cosines are the 7x7 stem's weight and bias, at
  0.19. Their backward path is the longest chain of quantized ops; every
  other gradient has a cosine of 0.986 or more.
- The update cosine stays below the gradient cosine because Adam's first
  step is close to `lr * sign(g)`: a small gradient error near zero flips
  the sign.

Time for one step, the last configuration:

| | Seconds |
|---|---|
| NPU execute, all 180 segments | 1.13 |
| Wall, including host ops, per-segment simulation and float checks, file IO and 182 health checks | 63 |
| Host-only float run of the same step | 5.1 |

This is a correctness harness, not a speedup. The NPU nodes are 24% of the
graph, and the Convs, most binary ops and all of the optimizer are still on
the host.

## Reproduce

```sh
PY=/mnt/data/cache/claude-work/tg-venv/bin/python   # numpy, onnx, onnxruntime, tinygrad fork
cd scripts/axera
systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  $PY step_runner.py --mode npu --out /path/validate.json
systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  $PY step_runner.py --mode npu --validated /path/validate.json \
  --exclude '^(Softmax|Log|Neg)_' --host-optimizer --out /path/final.json
```

- `--mode float` checks the runner itself: the loss matches the reference
  to 1e-7 relative, and every gradient has cosine 1.0.
- `--mode sim` runs the whole step on the simulation. It produces NaN,
  because the fake-quantized `Div`/`Log` see zeros that the device
  saturates.
- The peak host memory is 10 GB, from the node-by-node evaluation and the
  per-segment float checks.
