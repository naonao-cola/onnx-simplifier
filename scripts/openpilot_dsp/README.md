# openpilot's models on the Snapdragon 845 Hexagon V65 cDSP (HVX int8): simulator study

openpilot no longer uses SNPE. At master `027770d` (2026-09-24), `modeld` and `dmonitoringmodeld` compile
their ONNX models with tinygrad (`DEV=QCOM IMAGE=1 FLOAT16=1`, `openpilot/selfdrive/modeld/SConscript`) and
run them on the comma device's Adreno 630 GPU. SNPE's DSP runtime used to run them on the Hexagon DSP.
This directory asks whether that can be done again without SNPE, with our own int8 HVX kernels on the
845's V65 cDSP. The GPU would then be free for other work.

**No 845 device was used.** Every number below is one of:

- **measured on the host**: ONNX Runtime accuracy on real route frames, or hexagon-sim cycles;
- **measured on the device in the route's own logs**: `modelExecutionTime` from the comma device that
  recorded the route;
- **estimated**: marked as such, with the assumption stated.

## Summary

| | driving (`driving_supercombo.onnx`) | driver monitoring (`dmonitoring_model.onnx`) |
|---|---|---|
| Compute | 0.85 GMAC: backbone 0.79 (FastViT-style convs), heads 0.053 (23.5 M params, GEMV-shaped) | 0.41 GMAC: backbone 0.41, heads 0.001 |
| GPU today, from the route's rlog (openpilot 0.10.4, older model) | 29.2-29.7 ms mean, p95 ≤ 30.7 ms | 14.3-14.8 ms mean |
| openpilot CI budget (`test_onroad.py`) | mean < 40 ms, max < 60 ms | mean < 50 ms, max < 300 ms |
| Int8 PTQ accuracy (fp32 = reference) | W8A8: **broken** (plan lateral error 4.4 m mean). W8A16 backbone + float heads: 0.07 m mean, lead prob p95 error 0.19 | W8A8: blink decisions agree only 72-94%. **W8A16: ≥ 98.8%**, worst probability head error 1.5% |
| Projected V65 backbone latency, 2 HVX threads at 1.0 GHz (assumed) | W8A8 11.8 ms, **mixed A8/A16 16.2 ms**, W8A16 19.7 ms; heads are extra, ~3 ms int8 (estimated) | W8A8 6.1 ms, **W8A16 10.9 ms** |
| Upstream tinygrad `DEV=DSP` on these models | fp16 graph: fails to link (`__extendhfsf2`/`__truncsfhf2` undefined). fp32 graph: not tried (see the DM column) | fp32 graph: compiles (730 kernels) and matches ORT to 1.4e-5 under qemu. It is scalar float code: V65 HVX has no float |

On compute alone, the DSP looks competitive with the GPU numbers in the route's logs, if the 845's cDSP
really runs 2 HVX threads at around 1 GHz. **Accuracy is the real blocker.** Plain int8 PTQ breaks the
driving model and the DM blink heads. W8A16 is close but still moves the driving model's lead
probabilities. The next step is QAT, or at least a better weight-rounding pass, not faster kernels.

## 1. Models and profile (`profile_models.py`)

The models were fetched through the LFS media URL. Their sha256 matches the pointers at `027770d`.

- **driving_supercombo.onnx**: 377 nodes, 30.0 M fp16 params, opset 20.
  - Inputs: `new_img` u8 `[2,6,128,256]` (narrow and wide road camera, YUV420 as 6 channels) plus the
    recurrent queues `state_img_q` u8 `[2,5,6,128,256]`, `state_desire_q` `[100,1,8]` and `state_feat_q`
    `[96,1,512]`, and `desire`/`traffic_convention`/`action_t`.
  - Output: `[1,2576]` float, split by the `output_slices` metadata.
  - Backbone MACs: 0.794 G, of which stem dense 3x3/s2 24→64 is 113 M, 30 pointwise convs 638 M, and
    29 depthwise 3x3/7x7 43 M (FastViT-style reparameterized blocks, tanh-GELU, layer scale).
  - Heads: 82 Gemm + 8 MatMul (one attention block), 53 M MACs over 23.5 M params.
- **dmonitoring_model.onnx**: 201 nodes, 3.6 M fp16 params.
  - Inputs: `input_img` u8 `[1,1440*960]` (warped luma) and `calib` `[1,3]`. Output: fp16 `[1,553]`.
  - Backbone MACs: 0.411 G, of which 3 dense 3x3/s2 convs are 187 M, 28 pointwise 212 M, and
    15 depthwise 12 M.
  - Heads: 53 Gemm, 1.1 M MACs.
- **fp16 sensitivity**: fp16 vs fp32 on real frames is negligible.
  - Driving: plan lateral error 0.9 mm mean, lead probability error 1e-4.
  - DM: probability errors ≤ 2e-4.

## 2. Real inputs (`prepare_inputs.py`, `read_rlog.py`)

- **Frames**: from the public demo route `5beb9b58bd12b691/0000010a--a51155e496` (the route
  `openpilot/tools` uses), fetched through `api.commadotai.com/v1/route/.../files`.
  - Segments 3 and 12 are for calibration; segment 8 is the held-out evaluation set.
  - Each segment is 600 frames at 20 Hz from `fcamera`/`ecamera`/`dcamera.hevc` (1344x760, device `mici`,
    sensor os04c10).
- **Warps**: modeld's own, reimplemented in numpy.
  - Driving: `get_warp_matrix(rpyCalib, intrinsics, bigmodel)` with nearest-neighbour sampling, then
    tinygrad `compile_warp.py`'s 6-channel YUV420 layout.
  - DM: `cam.intrinsics @ inv(dmonitoringmodel_intrinsics)`, border fill 16.
- **Calibration**: `rpyCalib` comes from each segment's `extrinsicsCalibration` in the rlog (pitch 0.164
  rad). With the nominal zero calibration, the narrow camera shows the hood.
- **Driving run**: sequential, one call per frame, with the model's `next_state_*` outputs fed back as
  in modeld. So a quantized model free-runs on its own recurrent state.
  - Fixed inputs: `desire` 0, LHD, `action_t` = `[0.25, 0.55]`. This is a nominal value; the route's
    delays are not read.
- **Timings**: the rlog also carries the device's own `modelExecutionTime`, which is where the GPU
  numbers above come from. That is openpilot 0.10.4, whose driving model predates master's.

## 3. Int8 PTQ accuracy (`quantize.py`, `run_models.py compare`, `sweep_driving_groups.py`)

Setup:
- Quantizer: onnxsim `quantize_full_qdq` (whole-graph QDQ, int8 per-channel weights, int32 bias),
  calibrated on segments 3+12 and evaluated on segment 8 (600 frames).
- ORT runs at the *basic* optimization level, so QDQ is simulated rather than fused.
- References: the fp32 conversion of each model (`run_models.py to-fp32`), compared through openpilot's
  own `Parser`.

**Driver monitoring**, probability heads (the reference blink rate is 11-13% of frames):

| policy | raw cos | L/R blink decision agreement | eyes | sleep |
|---|---:|---:|---:|---:|
| W8A8 minmax | 0.984 | 0.917 / 0.938 | 0.997 | 0.972 |
| W8A8 percentile | 0.993 | 0.805 / 0.840 | 1.0 | 0.972 |
| W8A8 mse | 0.993 | 0.727 / 0.770 | 1.0 | 0.972 |
| **W8A16 minmax** | **0.9996** | **0.988 / 0.993** | 1.0 | 0.972 |

- **Blink is the int8-sensitive output.** A prefix/suffix sweep of which convs get 16-bit activations
  found no small subset that fixes it. onnxsim's greedy `search_activation_precision_for_budget` ended
  with 191 of 201 nodes in float.
- **`sleep_prob` is a calibration-coverage bug, not a precision one.** Its logit is never positive in
  either calibration segment, so its output range is clamped at 0 and every quantized model outputs
  sleep = 0.5 on the 17 frames where the reference says > 0.5. Fix: keep the DM heads float (1.1 M MACs,
  free on the CPU), or calibrate on drowsy-driver data.

**Driving**. Plan is the lateral position error over the 33 plan points. Lead is `lead_prob[0]`; the
reference has a lead in 96% of frames.

| policy | plan lat mean | plan lat @10 s p95 | lead prob mean / p95 error | lead decision agreement | lead x error |
|---|---:|---:|---:|---:|---:|
| fp16 (deployed precision) | 0.001 m | 0.016 m | 0.0001 | 1.000 | 0.02 m |
| W8A8, heads float | 4.30 m | 54.8 m | 0.125 | 0.957 | 55 m |
| **W8A16 backbone, heads float** | **0.069 m** | 0.91 m | 0.034 / 0.19 | 0.938 | 1.8 m |
| mixed A8/A16 backbone (`policy_driving_mixed.json`), heads float | 0.137 m | 2.08 m | 0.051 / 0.24 | 0.910 | 3.0 m |
| int8 weights only, float activations (backbone convs) | 0.060 m | 0.87 m | 0.032 | 0.945 | 1.9 m |
| int8 weights only, float activations (heads) | 0.043 m | 0.62 m | 0.005 | 0.993 | 0.66 m |
| whole graph QDQ incl. heads (u8 or u16) | ≥ 0.25 m | ≥ 5 m | | | 21-64 m |

Findings:
- **uint8 activations break the backbone.** Activations are heavy-tailed: GELU/residual max/std is
  20-60 (e.g. `gelu_17` reaches 193 with std 3.3).
- **Where it breaks.** A window-by-window sweep (only one window uint8, the rest uint16) puts the damage
  in the stem/stage 0 (windows 0-3) and the last stage (windows 12-13). Windows 4-11 tolerate uint8;
  that is the mixed policy.
- **W8A16's remaining error is mostly the int8 weights, not the activations.** Weight-only int8 already
  gives it. Per group, the first 3 convs (the stem's dense 3x3/s2, then dw 3x3/s2, then 1x1) matter
  most: keeping just those three in float roughly halves the plan and lead error.
- **Heads.** Quantizing the heads' activations crushes the concatenated `outputs` tensor (one scale for
  plan metres and probabilities), so the heads' outputs must stay float. The heads' int8 *weights* are
  mostly fine (lead agreement 0.993).

Not tried yet, and the obvious next step: QAT (the weights and data exist only on comma's side), or
AdaRound/GPTQ-style weight rounding (onnxsim has `adaround.py`/`gptq`) for the backbone convs. Also
16-bit weights for the 3 stem convs, which the kernels can run as a 2-pass `vrmpy` like `pw16`.

## 4. V65 HVX kernels on hexagon-sim (`hvx65/`)

- **Toolchain.** The Qualcomm open-access toolchain 19.0.02 no longer accepts `-mv65`, and hexagon-sim
  only models v68+. So `kernels.c` is compiled by upstream **clang-19 `-mcpu=hexagonv65 -mhvx=v65`**
  (the assembler rejects any instruction V65 lacks), linked with the toolchain's v68 standalone
  runtime, and run on **`hexagon-sim -mv68 --timing`**.
- **Timing.** Cycles come from UPCYCLE (SSR.CE enabled) around one kernel call. They are a
  **V68-pipeline proxy** for V65: same ISA subset, and not validated against 845 silicon.
- **Correctness.** Every kernel is bit-exact against a scalar C reference of the same fixed-point spec
  (`./run_sim.sh <op> ... check`).

| kernel | what | measured (single thread) |
|---|---|---|
| `pw` | 1x1 conv, u8 act x s8 weight, `vrmpy(Vu.ub=32 px x 4 ch, Rt.b)`, 8 output channels per activation load, fused Q31 requant, output in the next layer's layout | 40-64 MAC/cycle on real shapes (K ≥ 128), 25-30 at K = 64 (requant-bound) |
| `pw16` | the same with u16 activations (W8A16): two `vrmpy` passes (lo/hi bytes), combined at requant | ~2x `pw` cycles |
| `dwc` | depthwise kxk (3, 5, 7), stride 1/2, channels-last, `vzxt` + `vmpyacc(Ww, Vh, Vh)`, 2 pixels per iteration | 4-15 MAC/cycle (C padded to 128: C = 64/96/192 layers waste lanes) |
| `gelu` | u8→u8 table, 8x `vlut32` | 4.0 bytes/cycle |

The 4-channel `pw` variant (`pw4`) stalls on the `vrmpy` accumulate latency at 29 MAC/cycle.
Depthwise is the weakest kernel: 7.9-9.2 M of driving's 32-39 M cycles for 5% of its MACs. It is the
first thing to optimize, e.g. a 2-pixel packing for C = 64.

**Projection** (`project_latency.py`). Every backbone conv of both models runs through the simulator on
its real shape; the cycles are cached in `sim_cycles_cache.json` and the per-layer rows are in
`results/`. Some parts are estimated:
- the im2col copy for dense kxk convs, at 64 B/cycle;
- u16 depthwise and u16 elementwise ops, at 2x the u8 kernels;
- 1x1-spatial GEMVs, from weight bytes at 8 GB/s.

| model / policy | single-thread Mcycles (pw / dense / dw / eltwise) | 2 threads @ 1.0 GHz |
|---|---:|---:|
| DM W8A8 | 12.2 (5.1 / 4.5 / 1.5 / 0.8) | 6.1 ms |
| DM W8A16 | 21.7 (8.9 / 7.8 / 3.0 / 1.7) | 10.9 ms |
| driving W8A8 | 23.6 (15.4 / 2.1 / 4.6 / 1.4) | 11.8 ms |
| driving mixed A8/A16 | 32.5 (18.2 / 3.8 / 8.0 / 2.4) | 16.2 ms |
| driving W8A16 | 39.5 (23.6 / 3.8 / 9.2 / 2.8) | 19.7 ms |

Assumptions:
- **HVX contexts.** The 845's Hexagon 685 is listed as having "dual HVX"
  ([Notebookcheck](https://www.notebookcheck.net/Qualcomm-Snapdragon-845-SoC-Benchmarks-and-Specs.299446.0.html),
  [GSMArena](https://m.gsmarena.com/qualcomm_full_specs_snapdragon_845-news-28620.php)). So 2 threads,
  with perfect scaling assumed.
- **Clock.** The cDSP clock is **not known here**; 1.0 GHz is a placeholder, and latency scales
  inversely with it.
- **Heads.** Not included. The driving heads are GEMV-shaped over 23.5 M params, so they are bound by
  weight bandwidth: about 3 ms in int8 at 8 GB/s (estimated), or they stay on the CPU/GPU in fp16.

## 5. Upstream tinygrad `DEV=DSP` (`tinygrad_dsp_check.py`)

tinygrad at openpilot's pinned commit `9d0446a` has a DSP backend (`tinygrad/runtime/ops_dsp.py`):
- it compiles `--target=hexagon -mcpu=hexagonv65 -mhvx=v65 -nostdlib`;
- on the device it talks to `/dev/adsprpc-smd` + ION directly and boots `/dsp/cdsp/fastrpc_shell_3`;
- `MOCKDSP=1` runs the same code under `qemu-hexagon`.

Its CI benchmark (`benchmark.yml`, job `testqualcommdsp`) runs an int8 MobileNetV2 with `DEV=DSP NOOPT=1`
on a self-hosted **comma4** runner, with a `testsig-*.so` symlinked in. So the signed-PD route through
a test signature is what tinygrad uses on comma hardware.

Results here, with `MOCKDSP=1` and openpilot's `compile_onnx.py`:
- **DM fp16 model**: link error, `undefined symbol: __extendhfsf2` / `__truncsfhf2` (`-nostdlib`
  without compiler-rt).
- **DM fp32 conversion**: compiles into 730 kernels and matches ORT on 2 real frames, max |diff| 1.4e-5.
  But V65 HVX has no float, so this is scalar float code. It gives no speed signal (qemu), and it is not
  the int8 HVX path this study needs.

## Going on device (plan)

1. **Access.** comma devices run AGNOS (Linux, root).
   - tinygrad already reaches the cDSP there without a Qualcomm SDK: `/dev/adsprpc-smd` +
     `fastrpc_shell_3` + ION.
   - The dynamically loaded skel needs a **test signature** for the device's serial (`testsig-0x<serial>.so`,
     generated with the Hexagon SDK's `elfsigner`/`signer` tooling). Install it under the DSP search path,
     which root allows. This is the same route as tinygrad's comma4 CI.
   - Alternative: link against AGNOS's `libcdsprpc.so` and use `remote_handle64_open` like
     `../android/tinygrad_hexagon_bridge/native_transport`.
2. **Skel.** `hvx65/kernels.c`'s kernels go into one skel with a fused, static-schedule driver (the
   `native_transport` pattern). Weights stay resident; each 20 Hz frame is one FastRPC call on
   ION/dmabuf buffers.
3. **Measure first.** Before optimizing, get the real cDSP clock (DCVS/turbo vote) and the HVX thread
   count, then check the sim's cycles against the device.
4. **Accuracy.** Needs QAT or AdaRound/GPTQ before int8 on the DSP can replace the GPU's fp16 path.

## Files

| file | purpose |
|---|---|
| `profile_models.py` | op histogram, per-op MACs, I/O |
| `prepare_inputs.py` | route hevc → modeld-exact model inputs (`.npz`) |
| `read_rlog.py` | calibration + on-device model timings from rlogs |
| `run_models.py` | fp16→fp32 conversion, sequential model runs with state feedback, openpilot-parser comparison |
| `quantize.py` | onnxsim `quantize_full_qdq` with real-frame calibration, head-float and policy options |
| `sweep_driving_groups.py`, `policy_driving_mixed.json` | per-window uint8 damage sweep and the resulting mixed policy |
| `hvx65/kernels.c`, `build_sim.sh`, `run_sim.sh` | V65 HVX kernels + hexagon-sim harness |
| `project_latency.py`, `sim_cycles_cache.json`, `results/` | per-layer sim runs → model latency projection |
| `tinygrad_dsp_check.py` | upstream tinygrad `DEV=DSP MOCKDSP=1` vs ORT on real DM frames |

Reproduce (paths are examples):

```bash
export PYTHONPATH=<openpilot checkout>:<this repo>:scripts/openpilot_dsp
python scripts/openpilot_dsp/prepare_inputs.py --road seg8_fcamera.hevc --wide seg8_ecamera.hevc \
  --driver seg8_dcamera.hevc --rpy-calib=-0.00028,0.16415,0.00528 --frames 600 --out inputs_seg8
python scripts/openpilot_dsp/run_models.py to-fp32 driving_supercombo.onnx driving_fp32.onnx
python scripts/openpilot_dsp/quantize.py driving driving_fp32.onnx inputs_seg3.npz,inputs_seg12.npz q.onnx \
  --samples 64 --stride 18 --activation-dtype uint16 --float-after-last-conv
python scripts/openpilot_dsp/run_models.py run driving q.onnx inputs_seg8.npz q.npy
python scripts/openpilot_dsp/run_models.py compare driving driving_supercombo.onnx ref_fp32.npy q.npy
scripts/openpilot_dsp/hvx65/build_sim.sh && scripts/openpilot_dsp/hvx65/run_sim.sh pw 2048 64 192 check
python scripts/openpilot_dsp/project_latency.py driving_fp32.onnx --act uint16 \
  --policy scripts/openpilot_dsp/policy_driving_mixed.json
```
