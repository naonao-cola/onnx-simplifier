# HMX probe on Hexagon V69 (Xiaomi 12S, SM8475)

Can our own **unsigned** FastRPC skel drive HMX (the HTP's matrix unit) on V69, the way MNN's
`source/backend/hexagon/htp-ops-lib` does on v73+ (fp16 HMX, inline asm, `HAP_compute_res_hmx_lock`;
arXiv 2509.23324)?

## Findings so far

| step | result |
|---|---|
| toolchain | Hexagon_open_access 19.0.02 (login-free) and the SDK's 19.0.04 both assemble HMX for `-mcpu=hexagonv69 -mhmx`: `{ activation.ub = mxmem(Rs,Rt):deep ; weight.b = mxmem(Rs,Rt) }`, the `.hf` pair, `bias = mxmem(Rs)`, and the accumulator stores `mxmem(Rs,Rt):after[:sat].ub/.hf/.uh:2x1 = acc` (+ `:before`/`:retain`/`:cm`/`:pos`). `cvt.* = acc(Rs)` (MNN's v73 read-out) needs v73; f8 needs v79. |
| 1. VTCM + HMX lock from an unsigned PD | **works**: `HAP_compute_res_attr_set_vtcm_param_v2(256 KB)` + `attr_set_hmx_param(1)` + `HAP_compute_res_acquire` -> context, 256 KB VTCM mapped; `HAP_compute_res_hmx_lock` = 0, `hmx_unlock` = 0 (all return codes 0). Known-good skel passes afterwards. |
| 2. first int8 tile MAC | **hangs the cDSP**: all-ones `activation.ub`/`weight.b` loads (`Rt` = 2047) then `mxmem(o,0):after:sat.ub = acc`, no `bias = mxmem` first, on a QuRT thread holding the HVX lock at a priority above the RPC thread. The call never returns (client killed at 60 s); the next FastRPC client (a known-good msda skel, separate process) blocks in `fastrpc_wait_for_completion` and no SSR happened. Results after this point are void. |

## Suspects for step 2 (not yet tested -- the DSP must recover first)

- **No scale/bias loaded before the store.** HMX converts the accumulator through a per-channel
  scale/bias table (`bias = mxmem(Rs)`; MNN always sets it via `mxmem2` on v73). An unset table may
  not by itself hang, but it's the first difference from MNN's sequence.
- **`Rt` / tile span.** The V69 int8 tile geometry is unknown; `Rt` = 2047 assumed MNN's fp16
  2 KB tile. A span the unit doesn't accept could stall the load.
- **Store operand.** `mxmem(Rs,Rt) = acc`'s `Rt` was 0 (MNN passes 0 to the v73 `cvt` store).
- **Thread priority.** The probe thread ran one level above the RPC thread; a stalled HMX op on a
  high-priority thread can starve the rest of the DSP. Run at the RPC thread's priority next time.

Next attempts should change one of these at a time, each followed by the health check, and only on
a freshly recovered DSP.

## Round 2 (codex/android-hmx-probe2): simulator first

The v69 ISS (`libhexagonissv69.so`, `hexagon-sim -mv69 --mhmx 1`) models HMX, so sequences can be
checked with zero risk before they touch the phone (`hmx_sim.c`, `sim.sh`; standalone, no QuRT):

| sim experiment | result |
|---|---|
| int8 `{activation.ub=mxmem(a,2047):deep; weight.b=mxmem(w,2047)}` + `mxmem(o,0):after:sat.ub=acc`, HMX not enabled | exception 0x18 at the first load (badva = VTCM) |
| same, SSR bit 26 set (HMX enable; QuRT's `hmx_lock` does this on the phone) | **completes**, writes 2048 B, all 0 (no scale table) -- so the missing `bias = mxmem` does **not** by itself hang |
| int8 + `bias = mxmem` with 256 B of word `0x00004000`, store `:after:sat.uh = acc:2x1` | **0x0020 = 32** in every output = K of one all-ones tile: the int8 MAC is correct |
| int8 scale word sweep | the low 16 bits are the per-column scale (0x4000 -> x1 for `uh`; >= 0x7fff saturates); high bits had no visible effect with this data |
| fp16 `{activation.hf ...:deep; weight.hf ...}` of 1.0s + `mxmem(o,0):after.hf=acc`, no scale table | **0x5000 = 32.0**: fp16 HMX works on v69 in the sim |
| fp16 + scale table word `0x3c000000` | 0x5020 = 33.0: high fp16 half acts as a bias |

So the phone hang's cause is outside the ISA: the untested difference is **HMX power** --
the first probe voted HVX power and DCVS only, never `HAP_power_set_HMX` (`power_up = 1`), which
MNN does before any HMX use. Round 2's first phone variant changes only that (`HMX_POWER=1`).

### Round 2 on the phone (each variant followed by the known-good msda health check: all PASS)

| variant | change | result |
|---|---|---|
| P0 | step 1 + `HAP_power_set_HMX` `power_up=1` (`HMX_POWER=1`) | power vote rc 0 from the unsigned PD; lock/unlock 0 |
| **P1** | **round 1's exact hanging int8 sequence + the HMX power vote (only change)** | **completes**: 2048 B written, all 0 (same as the sim without a scale table). **Round 1's hang = HMX not powered.** |
| P2 | + scale table (256 B of word `0x00004000`), store `:after:sat.uh = acc:2x1` | **0x0020 = 32 in every output** -- int8 HMX MAC correct |
| P3 | fp16: `activation.hf`/`weight.hf` of 1.0, store `:after.hf`, no scale table | **0x5000 = 32.0 in every output** -- fp16 HMX works on V69 |

Recipe that works from an unsigned FastRPC skel on V69: `HAP_power_set` HVX + **`HAP_power_set_HMX`
(`power_up = 1`)** -> `HAP_compute_res` acquire with VTCM + `attr_set_hmx_param(1)` ->
`HAP_compute_res_hmx_lock` on the thread that issues HMX -> tiles in VTCM -> (optional) `bias = mxmem`
scale table -> `{activation.* = mxmem(a,Rt):deep; weight.* = mxmem(w,Rt)}` -> `mxmem(o,0):after... = acc`.

## Files

`hmx_rpc.idl`, `hmx_impl.c` (skel: acquire, lock, one parameterized HMX sequence, release; return
codes), `hmx_client.c`, `build.sh` (needs `HEXAGON_SDK_ROOT` for qaic/headers and a
`HEXAGON_TOOLCHAIN` with `-mhmx`), `run.sh` (`setup` / `probe` / `health`; run under the host's phone
lock; `health` runs a known-good msda_hvx skel).

```
./run.sh setup
./run.sh probe 0 0 0 1 - - - 0          # step 1: acquire / hmx_lock / unlock / release
./run.sh health
./run.sh probe 1 0x00 2047 1 =1x8192 =1x8192 - 8192 o.bin   # step 2 (hung the DSP on V69)
```
