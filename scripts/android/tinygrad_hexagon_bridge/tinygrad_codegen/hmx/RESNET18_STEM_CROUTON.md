# ResNet-18 stem: what the crouton attempt found, and why it is unfinished

## Status: no result. Two agent runs hit the turn limit on this.

Baseline (reproduced 2026-09-26, master `a72ed55d`, fork `hvx-hmx-qdq` @ `3605e14ca`,
hexagon-sim `-mv69 --mhmx 1`, `HMX_VTCM_KB=4096`):

```
150 kernel calls {other: 122, hmx: 20, qadd: 8}
hexagon-sim: 0/25088 mismatches vs ORT's semantics; 23945103 pcycles/inference (PASS)
```

The stem (pid 57) is 6,320,833 pcycles = 26.4% of the graph, 4.4x the next largest conv.

## Where the 3.59x of wasted MACs comes from

The stem runs at 67 MAC/cycle, so it is not slow per MAC - it issues 423.6M MACs where the real
math is 118.0M. Decomposed, and verified against the generated kernel:

- The 2x2 phase split makes a 32-lane K block hold 4 phase-pixels x 8 **padded** channels. Only
  3 channels are real, so 12 of 32 lanes carry data: **2.67x**
- 16 phase-grid taps for 49 real taps: **1.35x**
- The output grid is *not* part of it: `P64 = r64(112*112) = 12544` is exactly the valid 112x112.
- The phase split factor is not a free parameter: `f` divides the *output* grid, and only `f=2`
  yields the required 112x112 (f=1 gives 218, f=4 gives 28).

## What was already tried, and failed

Packing 64 phase-pixels x 3 real channels = 192 lanes = 6 K blocks of 32 (100% lane use, no
padding, 2.67x fewer MACs). Implemented, bit-exact, **2.9x slower** at 68,918,887 pcycles.

The reason is the invariant that decides this whole optimisation:

> **The A pack must stay a flat contiguous 2 KB copy.**

An A tile row's 32 lanes were 4 phase-pixels of *one* grid row, so the whole 2 KB activation tile
was one contiguous `__hmx_i8_copy_a`. Spanning the window across grid rows made tinygrad
materialise the gather as its own copy kernel (`E_12928_32_6`, 46,377,606 pcycles = 67% of the
graph), and the stem conv itself only moved 6.32M -> 6.71M. The 2.67x MAC saving was entirely
eaten by losing the copy.

The old estimate in the analysis was wrong: it priced the gather against a whole-graph ~150k
pcycles for activation packing, but the stem's A operand alone is 12.9 MB and gathering it costs
~3.6x the flat copy.

## Where the answer is

The hand reference already has it. `test/external/dsp/hand/hmx/hmx_qconv.h` and `hmx_gemm_u8.h`:
activations live in **crouton form, 64 rows x 32 channels, byte `32*s + c`**. That is exactly the
A tile shape - one contiguous 2 KB - so the hand kernel needs no activation packing at all, and a
k x k conv is `:single` windows over shifted copies of already-packed data.

So the win requires a **graph-level layout change**: produce the stem's activation directly in
crouton form. Re-lane-packing the same grid cannot work, as measured above.

## Findings from the two unfinished runs, worth keeping

- **Removing the fused requant makes the compiler eliminate `buf0`'s stores.** A bisection run that
  dropped the requant to isolate the `__hmx_i8_addq` cost was reading dead code: the accumulator
  array's stores were optimised away, so the "saving" it measured was not the saving. Any future
  bisection on this kernel has to force the stores live before trusting a number.
- The stem's cost is not dominated by the HMX MACs. 6.32M pcycles for 423.6M MACs at 67 MAC/cycle
  leaves a large remainder, and the README's own breakdown of the 3x3 family (requant ~330k of
  682k pcycles, activation packing ~150k, byte-plane adds ~55k, HMX MACs only ~20k) says the
  arithmetic is the small part. **Any attempt that only removes MACs is attacking the wrong term.**

## Reproducing

```
HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC=clang-19 \
  HEXAGON_TOOLS=~/.cache/hexagon-oa-19/Tools HMX_VTCM_KB=4096 PYTHONPATH=<tinygrad> \
  python qdq_net.py <backbone_qdq.onnx> <outdir> --sim input.bin ref.bin
```

`<outdir>/sim_profile.txt` is `graph <pcycles>` then `call <pid> pcycles <n>` per call;
`graph.h` maps each `kNN(...); G_PROF(pid);` line to its kernel file and shape comment.
