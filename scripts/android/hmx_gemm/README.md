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
| K > 32 | consecutive blocks for both operands, one `:deep` load pair with `Rt = (K/32)*2048 - 1`, **at most 32 blocks (K <= 1024) per load pair** (K=1056 is wrong); longer K = consecutive load pairs, which keep accumulating until the store (exact to K=4096) | same `Rt` for both; the weight reads `Rt/2` bytes (exact to K=256; wrong at K=1024 -- max depth not bisected) |

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

Two rules that only showed up on the phone (hexagon-sim does not model the first):

- **An operand span must not cross a 256 KB VTCM boundary.** A load pair whose A or W span crosses
  one takes a user-PD page fault at the boundary (`Bad VA` = VTCM base + 0x40000 / 0x80000; the PD
  dies with rc 0x4e, the cDSP stays healthy). `hmx_valloc` places every span inside one 256 KB window.
- At most 32 K-tiles per load pair (above), also true in the simulator.

## GEMM (`hmx_gemm.h`, header-only)

`hmx_gemm_f16(A, Wp, bias, C, M, K, N, vtcm, vtcm_bytes)`: `C[M,N] = A[M,K] . W[K,N] + bias`, fp16
row-major A/C, W prepacked once with `hmx_pack_w_f16` into per-32-column blocks of K/32 tiles; K, N
multiples of 32, any M. A is packed into VTCM once, then per 32-column block the weight tiles are copied
into VTCM and every 32-row block is one `:deep` MAC over all of K plus one tile store.

Packing/unpacking is HVX: one `vshuff(row1, row0, -2)` of a row pair gives the row-pair vector of two
K blocks; output tiles unpack with `vdeal h` + `vmux`/`vror` into 64-column row segments. Weight tiles
stream DDR -> VTCM with HVX copies, l2fetch-prefetched two 16 KB chunks ahead.

hexagon-sim, vs a double reference (tolerance = fp16 rounding of the result): 32x32x32, 45x576x128 +
bias, 128x576x192 + bias, 32x1056x64, 45x1536x128 + bias, 128x4096x64 + bias -- 0 elements beyond
fp16 rounding.

### Phone (Xiaomi 12S, one HMX thread, turbo, `hmx_gemm_client`; every run PASS vs a double reference)

Pure MAC + tile store with A and W resident in VTCM (`mode 1`): **3.07 TMAC/s fp16** at 128x576x1536,
512^3 and 1024^3 (36.9 / 43.8 / 349 us) -- the single-thread HMX issue rate (QNN's HTP reaches ~16
TMAC/s on this phone with both HMX units and its own scheduling).

End to end from DDR (`mode 0`: pack A, stream every weight tile into VTCM, unpack C to DDR), per call,
with the phase split in core cycles (~1.5 GHz):

| M x K x N | us | TMAC/s | pack A | W copy | MAC + store | unpack C |
|---|---|---|---|---|---|---|
| 128 x 576 x 576 | 53.5 | 0.79 | 11.6 k | 29.1 k | 7.3 k | 32.0 k |
| 128 x 576 x 1536 | 161.6 | 0.70 | 24.3 k | 113.4 k | 19.3 k | 84.5 k |
| 128 x 1536 x 576 | 205.2 | 0.55 | 114.6 k | 118.9 k | 8.6 k | 65.0 k |
| 512 x 576 x 1536 | 376.3 | 1.20 | 58.8 k | 121.5 k | 48.0 k | 334.6 k |
| 1024 x 1024 x 1024 | 1407 | 0.76 | | | | |

The MACs are 5-12% of the time: fp16 weights stream at ~22 GB/s (1.77 MB in ~80 us) and C goes back
to DDR at ~7 GB/s. HMX pays off where operands stay in VTCM across calls (fused pipelines), not as a
standalone GEMM over DDR buffers. Run with `./run.sh setup; ./run.sh gemm <mode> <M> <K> <N> [iters]
[bias]; ./run.sh health` under the phone lock (`build.sh` needs `HEXAGON_SDK_ROOT` for qaic/headers and
`HEXAGON_TOOLCHAIN` for -mhmx).

**Two HMX threads:** a second `HAP_compute_res_acquire` with the HMX parameter, while the first is held
by the same unsigned PD, is refused (ctx 0; `mode 2` of `hmx_gemm_client`), so this setup drives one
HMX context (2.91 TMAC/s MAC loop at 256x512x512, `mode 3`). Whether QNN's ~16 TMAC/s comes from a
second unit or from its own scheduling is not visible from here.

## First use: SmolLM2-135M prompt processing (prefill) projections on HMX

`llm_prefill.py prep` runs SmolLM2-135M (fp32 torch) on a 128-token prompt, captures every layer's real
projection inputs with forward hooks (fp16), fuses q|k|v (576 -> 960) and gate|up (576 -> 3072), and
prepacks the fp16 weights; `hmx_gemm_llm_client` runs all 120 GEMMs (4 per layer x 30) on the phone in
one FastRPC session; `llm_prefill.py compare` checks them against torch (fp16 inputs, float64
reference). Phone, turbo, under the phone lock, health check clean:

| GEMM per layer | M x K x N | DSP time per call | x 30 layers |
|---|---|---|---|
| q\|k\|v | 128 x 576 x 960 | 114 us | 3.42 ms |
| o | 128 x 576 x 576 | 81 us | 2.42 ms |
| gate\|up | 128 x 576 x 3072 | 302 us | 9.07 ms |
| down | 128 x 1536 x 576 | 200 us | 5.99 ms |
| **all 120** | | | **20.9 ms** on the DSP (0.65 TMAC/s), 64-118 ms wall |

Every output matches torch to fp16 rounding (worst cos 1.0000000, worst max-relative error 4.6e-4).

**Versus the HTP:** QNN runs the *whole* fp16 prefill (these projections plus attention, RMSNorm,
RoPE, SwiGLU) in 24.5 ms (`../llm_tinygrad/`, #1886). The HMX projections alone already take 20.9 ms,
so a DSP-side prefill built on this GEMM would at best tie the HTP: the fp16 weights (212 MB per
prefill) stream into VTCM at ~22 GB/s and the outputs go back to DDR at ~7 GB/s, while the HMX MACs
are only ~12% of the time. The wall time is 3-6x the DSP time because FastRPC copies each call's
weights in and out; a real prefill would keep the weights resident in the DSP heap like the decode
loop (`../llm_tinygrad/hvx/decode_impl.c`, 101 tok/s). What would make it win: int8 weights (half the
bytes; needs the offset-K-block trick for signed int8 output on v69), keeping activations and outputs in
VTCM across a layer instead of DDR, and a second HMX context if the PD can get one.

Reproduce (host: `~/.cache/llm-venv` with torch + transformers; phone steps under the phone lock):

```
python llm_prefill.py prep --work $W                              # 120 GEMMs: $W/<layer>.<proj>.{a,wp,ref}
./build.sh && ./run.sh setup
adb push hmx_gemm_llm_client $D/; (cd $W && tar cf - manifest.txt *.a *.wp) | adb shell "mkdir -p $D/llm/out && cd $D/llm && tar xf -"
adb shell "cd $D && ADSP_LIBRARY_PATH=$D ./hmx_gemm_llm_client 'file:///hmx_gemm_rpc.so?hmx_gemm_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' llm 1"
adb pull $D/llm/out/. $W/phone/ && python llm_prefill.py compare --work $W --phone phone
```

`tests/test_hmx_gemm.py` runs `sim/gemm_sim.c` on hexagon-sim (32x64x64, 45x576x128 + bias,
45x1056x128 + bias) when `HEXAGON_TOOLS` points at a Hexagon toolchain.

## Handoff for code generation (tinygrad DSP backend: HMX as a TensorCore)

Everything a generated kernel needs, split so the generator emits only the kernel:

| file | what | emitted by a code generator? |
|---|---|---|
| `hmx_block.h` | the block primitive: `hmx_blk_set_table`, `hmx_blk_mac_f16` / `hmx_blk_mac_u8s8` (load K tiles, accumulating; splits K into legal load pairs), `hmx_blk_store_f16` / `_u16` / `_u8` (store one 32x32 tile with conversion, clears the accumulator); full layout + table + precondition notes in its header comment | yes -- this is the TensorCore "instruction" |
| `hmx_runtime.h` | `hmx_rt_power` (HVX + **HMX** power vote + turbo), `hmx_rt_acquire` (VTCM + HMX context), `hmx_rt_lock/unlock` (HVX + HMX lock on the issuing thread), `hmx_rt_release` | no -- the runtime (skel) does this once |
| `hmx_gemm.h` | reference GEMM built on the block: HVX pack/unpack, VTCM windows, weight streaming | as a reference to match |
| `sim/block_ref.c` | **bit-exact** reference test of the three store modes (random operands, any K) | the diff target |

Rules a generated kernel must follow (all measured; the first two only on the phone):

1. Vote `HAP_power_set_HMX` power_up before any HMX op, and issue HMX ops only on a thread holding the
   HVX lock and `HAP_compute_res_hmx_lock` (a QuRT worker thread with >= 64 KB stack, not the FastRPC
   thread). Without the power vote the first tile op wedged the cDSP until a phone reboot.
2. Operands in the HMX context's VTCM, tiles 2048-byte aligned, and **no operand span may cross a 256 KB
   VTCM boundary** (user-PD page fault; see `hmx_valloc`).
3. At most 32 K-tiles per fp16 load pair and 8 per int8 load pair; longer K = more load pairs before one
   store (they accumulate).
4. Layouts: `IDX(i, j) = 64*(i/2) + 2*j + i%2`. fp16 A(r,k) / W(k,c) / C(r,c) at halfword `IDX`;
   int8 A(r,k) at byte `2*IDX(r,k)+1` (odd bytes, 2048 B / K-block), W(k,c) at byte
   `128*(k/4)+4*c+k%4` (1024 B / K-block), C at u16 `IDX` (`_u16`) or byte `2*IDX+1` (`_u8`, which is the
   int8 activation layout -- chainable).
5. Column table (256 B, word c = column c): fp16 out = `rne_fp16(exact acc + bias_c)` with bias in the
   high half; int8 out = `floor(acc * s_c / 2)` (`_u16`) or `floor(acc * s_c / 512)` (`_u8`), `s_c` fp16
   in the low half, saturating, negative acc -> 0 (use the offset-K-block trick for signed products).

Harness: `./sim/run.sh sim/block_ref.c <ktiles>` builds with the login-free Hexagon_open_access 19 tools
(`HEXAGON_TOOLCHAIN`, default `~/.cache/hexagon-oa-19/Tools`; `NCSHIM` = a dir with libncurses.so.5)
and runs `hexagon-sim -mv69 --mhmx 1`; it prints per-mode mismatch counts and PASS (K = 32, 256, 384
all 0 mismatches). A generated kernel should replace the `hmx_blk_*` calls in it and keep PASS. Standalone
sim code must set SSR bit 26 (HMX enable) itself; `hmx_lock` does it on the phone. `tests/test_hmx_gemm.py`
runs `gemm_sim.c` and `block_ref.c` in CI when `HEXAGON_TOOLS` is set.

Targets measured on the phone (one HMX context, turbo):

| what | best |
|---|---|
| MAC + tile store, operands resident in VTCM (fp16, 128x576x1536 .. 1024^3) | **3.07 TMAC/s** |
| same, int8 (`hmx_probe` issue-rate loop) | ~3.7 TMAC/s |
| full GEMM from DDR, fp16 (512x576x1536) | 1.20 TMAC/s (weights ~22 GB/s in, C ~7 GB/s out) |
| SmolLM2-135M prefill projections, 120 GEMMs | 20.9 ms (HTP whole prefill: 24.5 ms) |
| QNN / HTP practical int8 ceiling on this phone | ~16 TMAC/s |

What the remaining gap to QNN looks like (not implemented here): the MACs are 5-12% of a DDR-fed GEMM,
so the win is keeping HMX fed -- (a) stream weight tiles DDR -> VTCM asynchronously while HMX works on
the previous block (user DMA, or other HVX threads doing the copies/l2fetch while the HMX thread only
issues MACs; the copy is ~80 us per 1.77 MB against ~13 us of MACs), double-buffered in two VTCM
windows; (b) keep activations and outputs in VTCM across ops (fusion) instead of packing from and
unpacking to DDR; (c) int8 weights to halve the bytes; (d) a second HMX unit: a second HMX compute_res
acquire in the same PD is refused, so either QNN's ceiling comes from scheduling a single unit better
or the second unit is not reachable from one unsigned PD (untested: `HAP_compute_res_hmx_lock2` with
`HAP_COMPUTE_RES_HMX_SHARED`, as MNN's v73 code uses).
