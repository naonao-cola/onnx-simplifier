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
  const float* value;                                   /* (NV, S, C) channels-last */
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

**Precision.** fp32 in and out; that's what the HTP pieces hand over. The HVX body computes in
qf32, which differs from IEEE fp32 only by rounding. On real BEVFormer and synthetic RT-DETR
calls it matches torch at rel <= 1.5e-5, cos 1.0. fp16 value maps would halve the L2 traffic and
are the obvious next step; `vcvt` hf -> sf is V68+.

**Which body runs.** `msda_run` uses the HVX body when all of these hold, else the scalar one:
- `D == 32`, `M <= 32`;
- `M * NO * L * P` is a multiple of 32 and at most 256;
- every level has `H * W < 65536`;
- `value`, `loc`, `attw` and `out` are 128-byte aligned.

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
run(value, loc, ref, attw, vis, shape, flags, rout result, rout dsp_us)
```
- `shape` is `msda_shape_pack()`'s int32 array: `NV L S M D P Q NO mode NVR RL R RD has_vis`,
  then `H W start` per level.
- Pass empty sequences for `ref` (mode 0) and `vis` (all visible).
- `flags`: bits 0-7 are threads (use 4); bits 8-15 are queries per job / 16 (default 32 queries).
- Use rpcmem buffers so the DSP maps them instead of copying.
- A client links `msda_stub.o` (from `build.sh`) and `-lcdsprpc`, and opens
  `file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp` after
  `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`.

## Checks

```sh
python3 msda_ref.py mmcv-check                                   # msda_reference vs mmcv's code
for k in rtdetr_decoder bevformer_tsa bevformer_sca loc_small; do python3 msda_ref.py synth $k cases/$k; done
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

hexagon-sim, all four kinds, HVX body: rel <= 1.5e-5, cos 1.000000000.

## Phone (Xiaomi 12S, SM8475 / V69)

Measured under the host's phone lock, median of 10, 4 threads. See the table below (and
`../vision_models/bevformer_tiny/README.md` for BEVFormer in context).
