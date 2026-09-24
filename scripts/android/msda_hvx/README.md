# Multi-scale deformable attention on the Hexagon HVX (`msda_hvx/`)

A model-agnostic MSDA kernel for the Snapdragon CDSP's HVX (V68+; the test phone is V69), called
over our own FastRPC skel. Its numerics are those of mmcv's `multi_scale_deformable_attn_pytorch`:
`grid_sample` bilinear, `align_corners=False`, zero padding. It covers:
- Deformable DETR, RT-DETR and DINO-style decoders/encoders (one value map, L levels);
- BEVFormer, which averages over several value maps:
  - the SCA: cameras with shared offsets and a per-(camera, query) visibility mask;
  - the TSA: a 2-frame queue with per-frame offsets.

The model owns the Linears and softmax that produce offsets and weights (e.g. on the HTP). The
kernel does everything from "offsets + weights + value" to the attention output. That includes
forming the locations from reference points, which on the HTP is a large broadcast of
elementwise ops.

| file | what |
|---|---|
| `msda_kernel.h` | the kernel, header-only: `msda_args_t`, `msda_check`, `msda_run` (scalar body for any shape, HVX qf32 body where it applies) |
| `msda_shape.h` | packs `msda_args_t`'s sizes into the RPC's `shape` array and back; buffer element counts |
| `msda_rpc.idl`, `msda_impl.c` | the FastRPC skel: shape checks, query blocks over QuRT threads (4 is the sweet spot) |
| `msda_ref.py` | the contract in torch (`msda_reference`), a verbatim copy of mmcv's function it's checked against, case files, synthetic model-shaped cases |
| `msda_io.h`, `msda_host_check.c` | case loader/compare; the scalar body vs torch on the host |
| `msda_sim.c` | the HVX body vs torch on `hexagon-sim`, with cycles (qemu 8.2 can't decode HVX float, so this is the phone-free HVX check) |
| `msda_client.c`, `build.sh` | builds the skel, the client and `msda_stub.o` (for other clients); the phone bench |

Model glue lives with each model, e.g. `../vision_models/bevformer_tiny/msda_hvx/`.

## The contract

For each query `q` and head `h`:

    out[q, h*D:(h+1)*D] = 1 / max(1, #visible maps)
        * sum over visible maps v, levels l, points p of
          attw[q, h, o, l, p] * bilinear(level l of value[v], head h's D channels, at loc)

- `o = v` if offsets/weights are per map (`NO == NV`), else 0 (shared).
- The bilinear sample is at pixel `(loc_x * W_l - 0.5, loc_y * H_l - 0.5)`.
- A tap outside the level reads 0 (zero padding).

The normalized location `loc` comes from the `loc` array according to `mode`:

| mode | `loc` array holds | location | used by |
|---|---|---|---|
| `MSDA_LOC` (0) | normalized locations | `loc` | mmcv's function itself |
| `MSDA_REF_PIX` (1) | raw offsets | `ref.xy + off / (W_l, H_l)` | Deformable DETR 2-d refs, BEVFormer |
| `MSDA_REF_BOX` (2) | raw offsets | `ref.xy + off / P * ref.wh * 0.5` (`RD = 4`) | Deformable DETR / RT-DETR box refs |

For `NV = 1`, no visibility and `MSDA_LOC`, this is exactly mmcv's function. `msda_ref.py
mmcv-check` measures max abs 1.2e-7 between the two.

## C API (`msda_kernel.h`)

```c
typedef struct {
  int NV;                                               /* value maps averaged over (1: plain MSDA) */
  int L;                                                /* levels (<= 8) */
  int H[8], W[8], start[8];                             /* level l = rows [start, start + H*W) of a map */
  int S;                                                /* rows per value map */
  int M, D;                                             /* heads, head dim: a row has C = M * D channels */
  int P;                                                /* points per (head, level) */
  int Q;                                                /* queries */
  int NO;                                               /* offsets/weights: 1 = shared by all maps, NV = per map */
  int mode;                                             /* MSDA_LOC / MSDA_REF_PIX / MSDA_REF_BOX */
  int NVR, RL, R, RD;                                   /* ref (NVR, Q, RL, R, RD): NVR 1 or NV, RL 1 or L,
                                                           point p uses entry p % R, RD 2 (x, y) or 4 (cx, cy, w, h) */
  int vdtype;                                           /* MSDA_F32 (0): value; MSDA_U8 (1): value_u8 + vscale/vzp */
  const float* value;                                   /* (NV, S, C) channels-last */
  const uint8_t* value_u8;                              /* (NV, S, C): real = (u8 - vzp[v]) * vscale[v] */
  const float* vscale;                                  /* (NV) */
  const int32_t* vzp;                                   /* (NV) */
  const float* loc;                                     /* (Q, M, NO, L, P, 2) locations (MSDA_LOC) or raw offsets */
  const float* ref;                                     /* unused for MSDA_LOC */
  const float* attw;                                    /* (Q, M, NO, L, P), softmaxed (over L*P, as the model does) */
  const uint8_t* vis;                                   /* (NV, Q) or NULL = all visible */
  float* out;                                           /* (Q, C) */
} msda_args_t;

int  msda_check(const msda_args_t*);                    /* 0 if the shape is valid */
void msda_run(const msda_args_t*, int q0, int q1);      /* queries [q0, q1); thread-safe on disjoint ranges */
```

**Layouts.** These are mmcv's, with an `NO` axis added. A PyTorch model's
`sampling_offsets(query).view(Q, M, L, P, 2)` and `attention_weights(query).view(Q, M, L*P).softmax(-1)`
are already in this layout (`NO = 1`). The value is `value_proj(...)` channels-last per level,
i.e. `(S, M*D)` with the levels concatenated, as mmcv's `value` is before its `.view(..., M, D)`.

**Precision.** Everything but the value maps is fp32 (offsets, weights, refs, output).

The value maps are fp32 or uint8 (`vdtype`), with one scale / zero point per map. The HTP emits
one per tensor, so pass the same pair for every map.
- **fp32:** the HVX body computes in qf32, which differs from IEEE fp32 only by rounding. It
  matches torch at rel <= 1.5e-5, cos 1.0.
- **uint8:** reads a quarter of the bytes. The HVX body multiplies each tap's zero-extended bytes
  by a Q15 weight into u32 accumulators (exact: sum <= 255 * 2^15 < 2^23). It dequantizes once
  per (query, head, map): `(sum w*u8 - zp * sum w) * scale / 2^15`. Rounding each tap weight to
  Q15 (<= 2^-16) costs up to ~2e-3 of max |out| on 100+ taps: cos >= 0.9999997 vs exact
  dequantized math. Quantizing the values themselves costs much more: per tensor, cos 0.99985 -
  0.99999 vs fp32 on real BEVFormer calls, 0.99994 on synthetic RT-DETR ones.

**Which body runs.** `msda_run` uses the HVX body when all of these hold, else the scalar one:
- `D == 32`, `M <= 32`;
- `M * NO * L * P` is a multiple of 32 and at most 256;
- every level has `H * W < 65536`;
- for uint8, `C` is a multiple of 128;
- the value buffer, `loc`, `attw` and `out` are 128-byte aligned.

rpcmem buffers are page-aligned. Both bodies handle any valid shape correctly; the scalar one is
just slow. The HVX body keeps ~24 KB on the stack; `msda_impl.c` gives its threads 64 KB. Batch > 1:
one call per batch element. `NV` averages its maps; it's not a batch axis.

**How the HVX body works.** Per query and value-map iteration, it has two phases:
1. **Coordinates.** 32 points per vector: location, `floor`, the 4 bilinear weights (x attention
   weight / #visible), tap addresses. V69 has no vector float->int convert (V73+ has one), so
   `floor` adds 1.5 * 2^23 in qf32, converts to sf and reads the integer from the mantissa; a
   +-1 fix-up makes it exact.
2. **Multiply-accumulate.** 4 independent qf32 accumulators per head, one 128-byte vector (a
   head's 32 channels) per tap.

Building the next item's taps is pipelined behind the current MAC. With shared offsets
(`NO = 1`), each visible map is its own iteration, so invisible (map, query) pairs cost nothing.

## FastRPC (`msda_rpc.idl`)

```
run(value, value_u8, vscale, vzp, loc, ref, attw, vis, shape, flags, rout result, rout dsp_us)
```
- `shape` is `msda_shape_pack()`'s int32 array: `NV L S M D P Q NO mode NVR RL R RD has_vis vdtype`,
  then `H W start` per level.
- Pass `value` for fp32 and `value_u8` + `vscale` + `vzp` for uint8, the other(s) empty.
- Pass empty sequences for `ref` (mode 0) and `vis` (all visible).
- `flags`: bits 0-7 are threads (use 4); bits 8-15 are queries per job / 16 (default 32 queries).
- Use rpcmem buffers so the DSP maps them instead of copying.
- A client links `msda_stub.o` (from `build.sh`) and `-lcdsprpc`, and opens
  `file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp` after
  `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`.

## Checks

```sh
python3 msda_ref.py mmcv-check                                   # msda_reference vs mmcv's code
for k in rtdetr_decoder bevformer_tsa bevformer_sca loc_small; do
  python3 msda_ref.py synth $k cases/$k; python3 msda_ref.py synth $k cases/${k}_u8 --u8; done
python3 msda_ref.py quantize-case <fp32 case> <out>             # a real call with uint8 values
cc -O2 -o msda_host_check msda_host_check.c -lm && ./msda_host_check cases/*          # scalar body vs torch
hexagon-clang -mv69 -mhvx -mhvx-length=128B -O2 msda_sim.c -o msda_sim.elf -lm -lhexagon
hexagon-sim -mv69 --simulated_returnval msda_sim.elf -- cases/rtdetr_decoder            # HVX body vs torch
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... CASES="cases/rtdetr_decoder" FLAGS=1,4 \
  PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./build.sh                 # phone
```

`tests/test_msda_hvx.py` runs the first four phone-free. It runs the HVX body on hexagon-sim when
`HEXAGON_TOOLS` is set.

The synthetic cases mix in edge cases:
- points off the map;
- coordinates exactly on pixel centers and borders;
- invisible maps;
- a query no camera sees.

hexagon-sim, all four kinds, HVX body:
- fp32: rel <= 1.5e-5, cos 1.000000000.
- uint8: within the Q15 bound above, cos >= 0.9999997.

## Phone (Xiaomi 12S, SM8475 / V69)

Median of 10 under the host's phone lock. "DSP" is the kernel's time on the DSP (HAP timer);
"wall" also includes the FastRPC call. Jobs are 32 queries unless noted. Every row matches its
torch reference within the tolerances above.

| call | shape | value | 1 thread DSP | 4 threads DSP / wall | 4 threads, 16-query jobs |
|---|---|---|---|---|---|
| RT-DETR-r18 decoder cross-attention (synthetic) | Q 300, M 8, L 3 (80², 40², 20²), P 4, D 32, box refs | fp32 | 3.94 | 1.42 / 1.74 | 1.28 / 1.60 |
| | | **uint8** | 3.41 | **1.16 / 1.50** | **1.08 / 1.40** |
| BEVFormer-tiny TSA (real frame) | Q 2500, 2 maps 50x50, per-map offsets, P 4 | fp32 | 17.3 | 4.52 / 4.89 | 4.89 / 5.26 |
| | | uint8 | 12.7 | 4.10 / 4.45 | 4.34 / 4.68 |
| BEVFormer-tiny SCA (real frame) | Q 2500, 6 cameras 15x25, shared offsets, R 4, P 8, 19% visible | fp32 | 13.5 | 4.12 / 4.47 | 4.33 / 4.69 |
| | | uint8 | 13.1 | 4.27 / 4.63 | 4.48 / 4.84 |

uint8 makes RT-DETR's decoder layer 18% faster and BEVFormer's TSA 9% faster. It also shrinks
the value maps 4x for the HTP -> DSP handoff. The SCA is the exception (+4%): it dequantizes per
(query, head, camera) over only 32 taps.

The 6-thread results are within noise of 4. What made the HVX body fast, in hexagon-sim cycles
per point (single thread), from a first scalar version at 168 cycles:
- **No `%` in the point loop.** `p % R` is a libcall on Hexagon, and the call spilled the HVX
  accumulator to the stack and back on every point.
- **The per-point coordinate and weight math runs 32 points per vector.** As scalar code it was
  117 of the 168 cycles per point: a serial chain of dependent float ops.
- **Per-lane constants are hoisted out of the per-query work:** level sizes, starts, map bases
  and scales.
- **The multiply-accumulate reads tap lists with 4 independent accumulators**, pipelined one
  item ahead.

What didn't help:
- a head-major work order;
- a TURBO clock vote;
- more than 4 threads;
- specializing `M = 8` at compile time.
