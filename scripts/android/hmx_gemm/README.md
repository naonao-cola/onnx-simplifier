# HMX GEMM on Hexagon V69 (Xiaomi 12S / SM8475), from an unsigned FastRPC skel

Follow-up to `../hmx_probe/` (PRs #1890, #1905), which found the working HMX recipe for an unsigned PD:
HVX power + **`HAP_power_set_HMX` power_up** (without it the first tile op wedged the cDSP until a
reboot) -> `HAP_compute_res` (VTCM + hmx param) -> `HAP_compute_res_hmx_lock` on the issuing thread ->
tiles in VTCM -> `bias = mxmem` column table -> `{activation = mxmem(a,Rt):deep; weight = mxmem(w,Rt)}`
-> `mxmem(o,0):after... = acc`. `hexagon-sim -mv69 --mhmx 1` models HMX (standalone code sets SSR bit
26, which QuRT's hmx_lock does on the phone) and matched the phone on every sequence, so every new
sequence here was developed in the simulator first.

## Tile layouts (all derived on hexagon-sim with one-hot operands, 0 mismatches over every byte)

32x32 tiles. `IDX(i, j) = 64*(i/2) + 2*j + i%2` ("2x1 interleave": row pairs, one 128-byte vector per
row pair).

| | fp16 (`.hf`) | int8 (`activation.ub` x `weight.b`) |
|---|---|---|
| activation A(r, k) | halfword `IDX(r, k)` | byte `2*IDX(r, k) + 1` = `128*(r/2) + 4*k + 2*(r%2) + 1`: **odd bytes only**, even bytes are never read |
| weight W(k, c) | halfword `IDX(k, c)` | byte `128*(k/4) + 4*c + k%4` |
| output C(r, c) | halfword `IDX(r, c)` (`:after.hf`) | `:sat.uh acc:2x1`: u16 `IDX(r, c)`; `:sat.ub`: byte `2*IDX(r, c) + 1` (= the activation layout, so a u8 output tile is directly the next layer's activation tile) |
| bytes per K=32 block | A 2048, W 2048 | A 2048, W 1024 |
| K > 32 | consecutive blocks for both operands, one `:deep` load pair with `Rt = (K/32)*2048 - 1` (checked to K=576) | same `Rt` for both; the weight reads `Rt/2` bytes (checked to K=512) |

Output column table (`bias = mxmem(T)`, 256 B, words 0..31 = columns 0..31, words 32..63 unused):

- fp16 out: word c high 16 bits = fp16 **bias** of column c (added exactly); low 16 bits: leave 0.
- int8 out: word c low 16 bits = fp16 **scale** `s`; `:sat.uh` = `min(65535, floor(acc * s / 2))`
  (0x4000 = x1), negative accumulators clamp to 0; `:sat.ub` = `min(255, floor(acc * s / 2 / 256))`.
  High bits unused. There is no signed int8 output and no int-accumulator -> fp16 output on v69
  (`:after.hf` after int8 loads writes only the table's bias), so signed int8 GEMMs need an offset
  K-block (activation 255 x per-column positive weights) folded into the output zero point.
- fp16 accumulation is exact (e.g. `1000 - 1000 + 30*2^-10` comes out exact) and rounded once on store.

Simulator programs (`sim/`, run with `./sim/run.sh sim/<prog>.c [args]`): `map_int8.c`, `map_fp16.c`
(one-hot layout maps), `scale_probe.c`, `scale_probe2.c`, `round_probe.c` (column table / conversion
modes), `gemm_sim.c` (the GEMM below vs a double reference).

## GEMM (`hmx_gemm.h`, header-only)

`hmx_gemm_f16(A, Wp, bias, C, M, K, N, vtcm, vtcm_bytes)`: `C[M,N] = A[M,K] . W[K,N] + bias`, fp16
row-major A/C, W prepacked once with `hmx_pack_w_f16` into per-32-column blocks of K/32 tiles; K, N
multiples of 32, any M. A is packed into VTCM once, then per 32-column block the weight tiles are copied
into VTCM and every 32-row block is one `:deep` MAC over all of K plus one tile store.

hexagon-sim, vs a double reference (tolerance = fp16 rounding of the result): 32x32x32, 45x576x96 +
bias, 128x576x192 + bias -- 0 elements beyond fp16 rounding.
