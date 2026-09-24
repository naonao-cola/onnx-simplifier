# MCC's decoder block through tinygrad's DSP backend (generalizing the hand kernel)

`../` hand-writes MCC's query decoder for the V69 DSP (HMX + 4 HVX threads, 2.6 ms per block at 1024 queries). This
directory runs the same block as **plain tinygrad Tensor code** through the onnxsim/tinygrad fork's DSP backend
(`hvx-hmx`: HVX codegen, qfloat, the HMX TensorCore) on the phone, checks it kernel by kernel, and fixes what that
turned up in the fork (onnxsim/tinygrad#7, branch `hvx-hmx-mcc`). The hand kernel is the target to match.

**Status: correct on the phone (cos 0.9999997 vs float64 at 1024 queries), 572 ms per block** -- about 220x the hand
kernel. It started at "doesn't compile / wrong / faults on the phone"; the table at the end has the steps and the
remaining gap.

## Pipeline

tinygrad's own DSP runtime can't reach this phone (raw FastRPC ioctls; see `../../tinygrad_hexagon_bridge`), and qemu
(MOCKDSP) can't run these kernels either: it can't execute HMX, and its Hexagon decoder aborts on the qfloat code the fork
now emits (`decode_packet: assertion failed`). So the kernels are captured and replayed:

| step | tool | what |
|---|---|---|
| capture | `mcc_tg.py check --no-run --save <bundle>` (`capture.py`) | runs tinygrad under MOCKDSP without executing (and without the mock compile: `CAPTURE_NO_COMPILE=1`, `PARALLEL=0`) and records every kernel call: source (from the PROGRAM uop -- lowering happens in worker processes), argument buffers, the pre-lowering AST |
| memory | `capture.save` | tinygrad's memory planner reuses one allocation for several logical buffers of different sizes (and buffers are views): the replay reproduces the layout as regions of merged address intervals, each region's image built from first-seen contents |
| hexagon-sim | `emit.py sim <bundle> [--ref ...] [--dump]` | every kernel rebuilt for real HMX (no `HMX_REF`), a driver replays the calls on `hexagon-sim -mv69 --mhmx 1` |
| per kernel | `verify.py <bundle>` | re-runs each call's AST on tinygrad's CPU backend from the replay's own inputs (`--dump`) and compares every argument buffer: the first call that differs is the bug |
| phone | `build.sh <bundle>` + `run.sh <bundle> [iters] [ref]` (`replay_rpc.idl`, `replay_impl.c`, `replay_client.c`) | a FastRPC skel with the kernels linked in: regions loaded once, VTCM + HMX acquired (VTCM aligned to 256 KB: the kernels' tile cache assumes one 256 KB window), calls timed one by one |

`tc_test.py` / `ew_test.py` do the same for one matmul / one element-wise op (the unit cases below).

## What running it found (fixed in onnxsim/tinygrad#7 unless noted)

1. **Fusion into matmul operands** (model code, `mcc_tg.py`): a LayerNorm / cast / GELU fused into a matmul's loads hides
   the plain row loads the HMX rewrite needs (it fell back to the generic tile op); GELU fused into fc1's epilogue was
   unrolled per output-tile element (1.5 MB of C, 10+ minutes of clang); S = q K^T fused with the self-score concatenate
   became an 86%-of-the-block scalar kernel. Each matmul operand is materialized (`.contiguous()`).
2. **HMX, single K tile** (fork): with head_dim 32, S = q K^T has K = one tile and no reduce loop; the rewrite took the
   enclosing output-tile loop for the reduction (accumulating across output tiles, a compile error), and its generic
   output path wrote 128-lane vectors over 32-element rows. Now: REDUCE-axis loops only; no reduce loop = begin before the
   op, store after it, row-pair output.
3. **HMX, A-panel prefetch** (fork): A's whole-panel L2 prefetch used B's (32-column) shape, 32*kt rows at A's row
   stride -- 8 MB past A for fc2 (K = 2048), a PD fault (rc 0x4e) on the phone once it reaches an unmapped page; hexagon-sim
   has no MMU and ran it fine. Isolated with `tc_test.py 1 1024 2048 512` (faults) vs `... 512 512` (fine).
4. **qfloat transcendentals** (fork): tinygrad's EXP2 decomposition needs vector int<->float conversion (v73+), so every
   exp / sigmoid / tanh kernel ran scalar on v69 (~190 cycles per element). EXP2 / RECIPROCAL now render as
   conversion-free HVX helpers (magic-number rounding, polynomial, exponent add; bit estimate + Newton), float division as
   a * reciprocal(b), and a lane column may be several vectors' consecutive lanes -- GELU 64x512 3.56M -> 0.19M Pcycles.
5. **Softmax row width** (model code): rows of 225 (224 seen + the self score) don't vectorize; the self term is kept
   apart as in the hand kernel.
6. **Replay pitfalls** (this directory): buffer aliasing (region replay), VTCM window alignment, qemu-free validation.

## Phone (Xiaomi 12S, one decoder block, 1024 queries, `run.sh`)

| version | ms per block | vs float64 |
|---|---:|---|
| first valid run (after 1-3): S on HMX, element-wise scalar | 914 | cos 0.9999997 |
| + qfloat exp2 / reciprocal, softmax 224 + self | (GELU 299 -> 8.4 ms) | |
| + multi-vector lane columns (softmax normalize 368 -> 28 ms) | **572** | cos 0.9999997 |
| hand kernel (`../mcc_block.h`) | 2.6 | cos 0.9999996 |

Per kernel now (ms): softmax row max 125 + sum of exp 171, LayerNorm statistics 2 x (43 + 45), self score 31, softmax
normalize 28, GELU 8.4; HMX: qkv 18.4, S 1.7, P V 45, proj 3.5, fc1 31, fc2 17.6 (the hand kernel's HMX work is 1.14 ms
in total, at 3.07 TMAC/s).

## The remaining gap, largest first

1. **Reductions laid out across rows**: the softmax / LayerNorm reductions upcast 128 *rows* and load one element per row
   at the row stride (128 scalar loads per reduce step); contiguous 32-lane loads along the row with a vector accumulator
   and one horizontal reduce is the HVX shape. A DSP heuristic for contiguous reduce axes, or BEAM timed on hexagon-sim
   (`HEXSIM=1`, already in the fork), should pick it.
2. **HMX kernels from DDR**: every matmul packs its operands from row-major DDR into VTCM tiles per call and unpacks the
   output; the hand kernel keeps activations in VTCM between steps and streams prepacked weights (see
   `../../tinygrad_hexagon_bridge/tinygrad_codegen/hmx`: 2.1x the hand GEMM for one GEMM, far more for small K).
3. **No fusion across ops**: 15 kernels, every intermediate through DDR (the hand kernel: one fused block).
4. **One thread**: the backend runs one HVX context; the hand kernel splits row blocks over 4 and overlaps HMX with HVX.

## Reproduce

```bash
TG=<onnxsim/tinygrad hvx-hmx-mcc checkout>
ENV="PARALLEL=0 CAPTURE_NO_COMPILE=1 HMX=1 DEV=DSP MOCKDSP=1 TC=1 HVX_ARCH=v69 CC=clang-19 HEXAGON_TOOLCHAIN=<Hexagon tools>"
env $ENV PYTHONPATH=$TG:.:.. python mcc_tg.py check --data <../ref.py export --q 1024 dir> --q 1024 --no-run --save $B
python emit.py sim $B --ref <ref_out0.bin rows> --q 1024          # hexagon-sim, real HMX
python emit.py sim $B --dump && THREADS=0 PYTHONPATH=$TG python verify.py $B   # per-kernel check vs the CPU backend
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... NDK_CLANG=... ./build.sh $B
PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh $B 3 $B/ref.bin
```
