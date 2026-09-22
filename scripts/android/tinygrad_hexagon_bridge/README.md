# Bridging tinygrad's Hexagon codegen onto real hardware through TVM's transport

Follow-up to `scripts/android/maskrcnn_e2e/README.md`'s "Diagnosed but not fixed" 1x1-conv
finding: asked to eliminate that using tinygrad instead of hand-writing a TVM schedule fix. This
is the record of what that actually took, what's real and working, and what isn't yet.

## Why tinygrad can't reach this phone's DSP directly

tinygrad's Hexagon backend (`tinygrad/runtime/ops_dsp.py`) has a real (non-mock) driver that
talks to `/dev/adsprpc-smd` via raw FastRPC ioctls, and a `MOCKDSP=1` mode that instead runs the
same generated C under `qemu-hexagon-static` on the host -- useful for codegen/correctness, not
a timing signal.

The real driver doesn't work on this phone: it's a production Android build (`adb root` is
refused -- *"adbd cannot run as root in production builds"*), SELinux is `Enforcing`, and
`/vendor/dsp/cdsp/fastrpc_shell_3` (the bootstrap binary the driver unconditionally reads) exists
but is permission-denied to the unprivileged `shell` user `adb shell` runs as. There's no
privilege-escalation path available without rooting/unlocking the bootloader -- an invasive
action on physical hardware, out of scope here.

TVM's own `tvm_rpc_android`, by contrast, reaches the DSP fine as a plain `adb shell`-launched
process (confirmed: `readelf -d` on the deployed binary shows it links `libcdsprpc.so`, the
vendor's *official* user-space FastRPC library) -- a different, already-permitted access path
than tinygrad's raw-ioctl one.

## The bridge

Since TVM's transport works and tinygrad's own doesn't, but TVM's Hexagon RPC has a **generic**
module loader (`session.load_module` -> `tvm.hexagon.load_module`, not tied to relay-compiled
graphs -- any correctly-built `.so` can be loaded and its exported functions called directly),
the bridge compiles tinygrad's own generated kernel with the same Hexagon toolchain TVM uses,
wraps it in a thin TVM `PackedFunc` ABI shim, links both into one `.so`, and loads/calls it
through TVM's existing session. Only the *transport* is TVM's; the *kernel* is tinygrad's own
codegen, unmodified.

```
capture_kernel.py --hw 512 --cin 64 --cout 256 --beam 2 --out kernel.c
    # runs a uint8 GEMM through DEV=DSP MOCKDSP=1, captures the rendered C source (verified
    # correct against a numpy reference under qemu), strips the mock-only entry boilerplate

bridge_and_test.py --kernel kernel.c --hexagon-toolchain $HEXAGON_TOOLCHAIN \
    --tvm-root <tvm source root> --hw 512 --cin 64 --cout 256
    # compiles kernel.c with the real (non-mock) Hexagon toolchain, generates and compiles
    # wrapper_template.c (a plain-C TVM PackedFunc shim -- not C++, since TVM's
    # TVM_DLL_EXPORT_TYPED_FUNC macro pulls in tvm::runtime::Array/Map, which need more libc++
    # than this freestanding Hexagon build provides), links via
    # tvm.contrib.hexagon.tools.link_shared, uploads+loads+calls it via a real
    # HexagonLauncher session, and times it
```

Verified end-to-end, real hardware, `cin=64, cout=256, hw=512` (the pathological shape from the
1x1-conv finding):

| Kernel | Median | Throughput | Correct |
|---|---:|---:|---|
| tinygrad, naive (BEAM=0) | 2137.8 ms | 0.42 GMAC/s | yes, bit-exact |
| tinygrad, BEAM=2 (found a real 4x4 register-tiled microkernel) | 8.4 ms | 1.00 GMAC/s | yes |
| tinygrad, vrmpy TensorCore (see below) | 29.9 ms | 0.28 GMAC/s | yes |
| stock TVM (hand-tuned `vrmpy` schedule) | 3.8 ms | 2.18 GMAC/s | yes |

BEAM search gave a genuine 2.37x improvement over naive codegen, confirmed on real hardware (not
just qemu's instruction-count proxy) -- real signal. But even BEAM=2's best kernel is ~2.2x
*slower* than TVM's hand-tuned schedule: tinygrad's Hexagon renderer had (before this session)
zero references to `vrmpy` or any dot-product intrinsic anywhere in its source -- it relies
entirely on LLVM's generic auto-vectorizer to recognize a scalar multiply-accumulate loop and
turn it into hardware SIMD, which it doesn't do for Hexagon's `vrmpy`. Higher BEAM values (4, 6)
converge to the identical kernel as BEAM=2 -- confirmed by diffing the generated source
byte-for-byte -- so more search width doesn't help; the renderer's action space has no vrmpy
option to find.

## Adding real vrmpy support to tinygrad

`vrmpy_tensorcore.patch` (PR: https://github.com/onnxsim/tinygrad/pull/1, against a
`v0.14.0`-pinned branch of the `onnxsim/tinygrad` fork) adds `hexagon_v65`, a real `TensorCore`
declaration for `V6_vrmpyub_acc_128B` / `V6_vrmpybusv_acc_128B`:
`D(int32x32) = C(int32x32) + dot4(A(u8x4 broadcast scalar), B(u8x128, 32 groups of 4))`, one HVX
vector instruction.

This is the **first TensorCore in tinygrad with `threads=1`**. Every existing definition
(CUDA/AMD/Metal) assumes warp-cooperative execution -- the `TensorCore` dataclass's `threads`,
`opts` and `swizzle` fields all encode how work splits across a warp's lanes. `vrmpy` needs none
of that: it's a plain single-thread SIMD instruction. There's no existing single-thread example
to copy the `opts`/`swizzle` fields from; they were derived directly from
`TensorCore.__post_init__`'s assertions (`local_axes=0` since `2**0==threads`; all 5
axis-doubling opts needed for `N=32` are `"u"`-type since `M=1` needs none) and worked on the
first attempt.

Two real bugs surfaced and fixed along the way, both specific to being the first
tensor-core-capable-but-warp-incapable backend:
1. `MockDSPRenderer.__init__` has its own override that never calls the base class's, silently
   dropping `tensor_cores` -- the TC was registered on `DSPRenderer` but invisible to the qemu
   path used for verification.
2. `hand_coded_optimizations` unconditionally attempts an `OptOps.LOCAL` opt right after
   successfully applying a TC (an "improve ILP" step), **uncaught** -- crashes outright on any
   backend with tensor cores but no local-memory support, which every backend until now has had.
   Guarded behind `renderer.has_local`.

**Verified correct**: bit-exact under qemu, and bit-exact on real Hexagon v73 hardware via the
bridge above (`tinygrad_gemm` -> genuine `__builtin_HEXAGON_V6_vrmpyub_acc_128B` calls, compiled
with the vendor Hexagon toolchain, no errors).

**Not yet fast**: 0.28 GMAC/s, slower than even naive scalar codegen. Root cause, precisely
diagnosed: `devectorizer2`'s `do_stack_wmma` (`codegen/__init__.py`) unconditionally decomposes
*every* WMMA's accumulator into 32 individual scalar element-loads before rendering. That's the
*correct* behavior for every other backend -- each GPU thread genuinely only ever holds a few
scalar elements of a warp-distributed fragment, so there's nothing to "keep as a vector" at the
per-thread level. It's *wrong* for Hexagon: the "32 elements" is one HVX vector register that a
single thread (`threads=1`, no warp) processes atomically in one instruction, and it should stay
vector-resident across the whole reduction loop instead of being rebuilt from 32 scalar reads
before every accumulate call. BEAM search independently corroborates the regression: across 15+
candidates explored at BEAM=2, it never selects the WMMA path, consistently preferring plain
scalar codegen -- exactly matching what the real-hardware measurement shows.

**Correction**: an earlier version of this note claimed a "minimal, non-interleaved single-TC-call
test" still showed the scalar rebuild, "confirming" the issue wasn't about interleaving. That test
used a hand-rolled monkeypatch of `postrange.apply_opts` to try to isolate a bare `Opt(OptOps.TC)`
application -- the monkeypatch silently wasn't being hit (confirmed later via the official
`NOOPT=1` control, which behaves differently), so that test was actually exercising the normal,
unmodified path the whole time and proved nothing about interleaving specifically. See "Three
more attempts" below for what a *reliable* trace found instead.

### Three more attempts at a fix, each real and each a dead end

Further digging (tracing via a monkeypatch of `heuristic.hand_coded_optimizations` directly,
which -- unlike patching `apply_opts` -- reliably fires) found the actual mechanism: `applied_opts`
after a real TC application is `[Opt(TC, ...), Opt(UPCAST, axis=0, amt=4)]` -- `hand_coded_optimizations`'s
own "improve ILP by upcasting M and N" step is what groups multiple independent M-row accumulators
into one interleaved buffer. Three attempts followed, each tested for correctness and, where
correct, for real speed on hardware:

1. **Skip the M-upcast for backends without locals** (guard `for tc_dim in [1,0]` down to `[0]`
   when `not tk.ren.has_local`): eliminates the interleaving cleanly -- WMMA call count drops from
   4 to 1 at the same shape, and the accumulator becomes a plain contiguous 32-element buffer
   slice instead of a stride-4 interleaved one. **Measured on real hardware: slower**, not
   faster -- 0.19 GMAC/s vs. 0.28 GMAC/s. Losing the M-row interleaving's instruction-level
   parallelism costs more than the simpler accumulator access saves. Reverted.
2. **Reorder the `TensorCore.swizzle` field's upcast axis list**: swept several permutations of
   `('u0','u1','u2','u3','u4')`. Every permutation except the original either produced **wrong
   results** (confirmed against a numpy reference) or left the accumulator's offset pattern
   completely unchanged. Conclusion: `swizzle` governs `A`/`B`'s data layout, tightly coupled to
   `vrmpy`'s actual hardware lane semantics (permuting it independently breaks which output lane
   receives which input) -- it does not control `C`'s (the accumulator's) element ordering at all.
3. **Remove the reversal in `TensorCore.base_upcast_axes()`** (the actual, separate mechanism that
   *does* order `C`, via `_apply_tc_opt`'s `tc_upcast_axes` slicing): even with only Hexagon's TC
   registered, this **crashes** a different invariant -- `codegen/late/coalesce.py`'s
   `memory_coalescing` pass asserts (`attempting multiple stores: 4`) on the resulting UOp
   structure. The reversal isn't an arbitrary convention; something else downstream depends on it
   being there, even though its only in-repo justification is a one-line comment ("this is defined
   in the swizzle... first we use the upcast axes, then the reduce").

All three are consistent with one pattern: every lever that touches the accumulator's construction
or ordering is entangled with some *other* invariant in the generic, shared kernel-optimization
pipeline that every backend goes through -- not something a local, Hexagon-only tweak can safely
adjust without either the fix itself regressing (attempt 1) or breaking a different, seemingly
unrelated assumption (attempts 2 and 3). A real fix within the shared `TensorCore`/`WMMA`
machinery needs a maintainer's-level map of what `_apply_tc_opt`, `base_upcast_axes`, and
`memory_coalescing` jointly assume -- more archaeology than this session had time for.

### Modeling Hexagon as its own accelerator, not a `TensorCore` variant

Given the shared-machinery entanglement above, the more promising direction is to **stop routing
through `Ops.WMMA`/`TensorCore` entirely** for Hexagon and instead give it its own, hand-written
lowering -- sidestepping `do_stack_wmma`, `base_upcast_axes`, and `memory_coalescing` altogether
rather than trying to make them single-thread-aware.

tinygrad has an established, precedented mechanism for exactly this: `Tensor.custom_kernel`
(`UOp.custom_kernel`) lets you hand-build a kernel's UOp graph directly -- explicit `UOp.range`
loops, `.index()`/`.store()`/`.load()`, and `Ops.CUSTOM`/`Ops.CUSTOMI` for injecting a raw,
target-specific intrinsic as a format string (`arg.format(*rendered_srcs)`). `tinygrad/llm/kernels/amd.py`
is a full, real, ~500-line module built this way for AMD-specific quantized-linear/attention
kernels, including register-resident (never memory-array-decomposed) accumulators via
`UOp.placeholder(shape, dtype, addrspace=AddrSpace.REG)` threaded through the reduction loop with
`.store()`/`.after()` -- precisely the "stay vector-resident across the loop" property `do_stack_wmma`
fails to give Hexagon's accumulator. `_amd_dp4a` (`amd.py`) is the exact same shape of problem as
`vrmpy`: a 4-element dot-product-accumulate hardware intrinsic, injected via one `Ops.CUSTOMI` call.

`custom_kernel_attempt.py` was a from-scratch attempt at the Hexagon equivalent -- one `vrmpy`
call, accumulating a real `int32x32` register value across a K-reduction, entirely bypassing
`Ops.WMMA`. It got stuck for a while (see the file's own header comment for that history), but
**this now works**, in `hex_gemm_kernel.py` -- see "It works" below. The bugs found and fixed
along the way, each a genuine, distinct issue (not the same wall hit repeatedly):

1. **UOp shape-broadcast mismatch** (`(32,)`/`(128,)`/`(1,)` couldn't unify): caused by calling
   `.load()` on the wide A/B slices before feeding them to the `Ops.CUSTOMI` intrinsic call, which
   gives them their full array shape. Traced `Ops.CUSTOM`/`Ops.CUSTOMI`'s shape rule directly
   (`case Ops.CUSTOM | Ops.CUSTOMI: if self.dtype is dtypes.void: return None` in `uop/ops.py`) and
   `_amd_load`'s real pattern in `amd.py` to find the fix.
2. **Fixed** by passing the raw `Ops.INDEX` (an address, shape `()`) directly into the `CUSTOMI` op
   instead of a `.load()`'d array value, matching `_amd_load`'s actual usage exactly -- then hit a
   **UOp spec-verification failure** (`uop/spec.py`'s `type_verify`) on `acc.load()`: loading an
   `Ops.AFTER`-wrapped placeholder directly isn't a form the verifier recognizes.
3. **Fixed** by dropping the explicit `.load()` entirely -- re-reading
   `_amd_flash_attention_decode_partial`'s real, working accumulator idiom
   (`prev_acc = acc.after(offset)[head]`, `acc[head].store(...)`, no `.load()` calls anywhere) showed
   that a `AddrSpace.REG` placeholder, once `AFTER`-scoped to its dependency, *is* the loadable value
   directly -- indexing/using it does the load implicitly.
4. Past both of those, hit **a genuine tinygrad control-flow bug**: `codegen/late/linearizer.py`'s
   scheduling pass asserted (`assert y.src[1] not in x.backward_slice_with_self`) on an adjacent
   comment reading *"TODO: this can happen! it causes infinite loop in shufflenet"*. **Fixed** by
   calling `.end(kc)` on the reduction range exactly once (only where the accumulate step's update
   value is constructed), not again in the final store's `.end(...)` call -- the double-`.end()` on
   the same range was the trigger, not an unrelated tinygrad bug after all.
5. Next error was mundane: `AxisType.GLOBAL` for the outer `M` loop rendered to `Ops.SPECIAL` with a
   GPU-style workitem code (`'g'`) `ClangRenderer` doesn't implement (`has_threads=False`, no grid
   dispatch). **Fixed** by using `AxisType.WEAK` (plain sequential loop) instead.
6. Then a **pointer-cast bug**: the format string wrote `*(unsigned char128*)&{1}`, but `{1}`
   (an `Ops.INDEX`) already renders as a pointer expression -- taking `&` of it is invalid. **Fixed**
   by dropping the `&`.
7. With that fixed, the *compiled* code revealed the real semantic bug: the `(32,)`-shaped
   `Ops.CUSTOMI` result was still being decomposed by the generic elementwise devectorizer into 32
   separate calls, each with a spurious `+lane_index` appended to the result -- `Ops.CUSTOMI`,
   despite representing one opaque hardware call, gets treated exactly like any other `(32,)`-shaped
   elementwise computation, which assumes each of the 32 output elements is an independent
   per-lane result. **Fixed** by never constructing a `(32,)`-shaped *value* at all: the whole
   accumulate step became one `dtypes.void`-dtype `Ops.CUSTOM` **statement** (`case Ops.CUSTOM |
   Ops.CUSTOMI: if self.dtype is dtypes.void: return None` in `uop/ops.py` -- `void` dtype exempts
   it from shape/broadcast tracking entirely), addressing the accumulator only by its base pointer
   (`acc[0]`, shape `()`, from indexing the `AddrSpace.REG` placeholder -- same address-not-value
   pattern as step 2) and doing the load-vrmpy-store round trip as raw C inside the format string.
8. Final snag: the format string referenced `unsigned char128`/`int32` -- vector typedefs the
   renderer only auto-emits when a shaped *value* of that type appears somewhere, which no longer
   happens once accumulation is a `void` statement. **Fixed** by using
   `__attribute__((vector_size(128)))` inline in the cast expression instead of a named typedef,
   sidestepping the need to inject a declaration into the kernel prologue at all.

### It works

`hex_gemm_kernel.py` is the resulting, working, from-scratch `vrmpy` GEMM kernel: `C[M,N] =
A[M,K] @ B[K,N]` (uint8 x uint8 -> int32), correct under qemu and **correct on real Hexagon v73
hardware** (via the TVM-transport bridge), at the *exact* pathological shape this whole
investigation started from -- `scripts/android/maskrcnn_e2e/README.md`'s ranked-profile finding,
`cin=64, cout=256`, spatial `200x272` (`M=54400`):

| | Median | Throughput |
|---|---:|---:|
| tinygrad, naive (BEAM=0) | -- | 0.42 GMAC/s (isolated 512-row proxy shape) |
| tinygrad, BEAM=2 (tiled) | -- | 1.00 GMAC/s (isolated 512-row proxy shape) |
| tinygrad, `Ops.WMMA`/`TensorCore` (broken accumulator) | -- | 0.28 GMAC/s (isolated 512-row proxy shape) |
| stock TVM (hand-tuned `vrmpy` schedule) | 254.1 ms | 3.51 GMAC/s (**real shape**) |
| **tinygrad, hand-written `custom_kernel`** | **29.363 ms** | **30.35 GMAC/s (real shape)** |

**8.65x faster than stock TVM's hand-tuned schedule, bit-exact correct, at the real shape** --
not a synthetic proxy. This is the single largest line item in the original ranked profile
(rank 2, 13.4% of the isolated-timing total, one of the "small-channel 1x1 convs at 10.1x below
the throughput of large-channel 3x3 convs" from the systemic-pattern finding); at 30.35 GMAC/s it's
now well past that comparison point entirely.

`B` needs pre-packing into `hex_gemm_kernel.pack_b()`'s layout before use (128 contiguous bytes
per `(N-tile, K-chunk)`, matching what one `vrmpy` call reads in one shot) -- a one-time,
host-side repack of the (static) weight tensor, not something done per-inference.

**Bug found and fixed**: originally failed whenever `cout == 32` exactly, or more generally
whenever the N-tile count is 1 (also affects `build_strided_kernel()`, same underlying pattern).
Root cause: `_reg_i32`'s accumulator-init scoped its `.after()` dependency to only the innermost
enclosing range (`nt_rng`); tinygrad's optimizer *eliminates* degenerate extent-1 ranges entirely,
which silently dropped the "reset the accumulator once per row" dependency down to "reset once,
ever" -- corrupting every row after the first. Fixed by depending on every enclosing range (`M`
can never degenerate to extent 1 in practice, so this is robust). This directly unlocked the tiny
`cout=12`/`cout=3` coverage below, via `cout` zero-padded to 32.

### Coverage: the other small-channel 1x1 convs in the backbone

`maskrcnn_e2e/profile_data/conv_profile.json` lists every unique conv shape in the backbone. The
same kernel (just re-parameterized by `cin`/`cout`/`M`, no code changes) was verified correct and
measured on real hardware against the next four largest small-channel 1x1-conv shapes by
isolated-timing impact:

| `cin` | `cout` | spatial | stock TVM | `custom_kernel` | speedup | correct |
|---:|---:|---|---:|---:|---:|---|
| 64 | 256 | 200x272 | 3.51 GMAC/s | 30.35 GMAC/s | **8.65x** | yes |
| 128 | 512 | 100x136 | 6.03 GMAC/s | 38.66 GMAC/s | **6.41x** | yes |
| 64 | 64 | 200x272 | 2.88 GMAC/s | 14.36 GMAC/s | **4.99x** | yes |
| 256 | 64 | 200x272 | 7.60 GMAC/s | 18.15 GMAC/s | **2.39x** | yes |
| 512 | 128 | 100x136 | 12.72 GMAC/s | 25.45 GMAC/s | **2.00x** | yes |

All five bit-exact correct, all faster than TVM's hand-tuned schedule. Together with the strided
shape below, they account for ~2217 ms of the ~7620 ms isolated-timing total from the original
ranked profile -- **about 29% of the entire backbone's isolated-timing sum**, from the same
`hex_gemm_kernel.py`/`build_strided_kernel()` code, re-parameterized by shape each time.

### Coverage: the strided 1x1 conv

The one strided shape in the profile (`cin=256, cout=128, stride=2 @200x272`, 82.9 ms) needed
real 2D spatial indexing -- `build_strided_kernel()` in `hex_gemm_kernel.py` takes the flat,
*unstrided* input (`ih*iw, cin`) and two `(oh, ow)` `WEAK` ranges instead of one flat `M` range,
computing the strided source row as `(oh*stride)*iw + (ow*stride)` directly in the index
expression -- no change to the accumulate/vrmpy step at all, only to how `A`'s address is
computed. Correct on the first attempt (qemu and real hardware), no new bugs:

| `cin` | `cout` | spatial | stride | stock TVM | `custom_kernel` | speedup |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 128 | 200x272 | 2 | 5.38 GMAC/s | 24.86 GMAC/s | **4.62x** |

### Coverage: the tiny RPN/mask-head convs (`cout=12` or `cout=3`)

Total impact across all spatial sizes of these two shapes is ~182 ms (not ~14 ms as an earlier
version of this note estimated -- corrected here), dominated by the largest spatial size
(`200x272`) for each: 83.7 ms (`cout=12`) and 52.1 ms (`cout=3`), together ~136 ms, ~75% of the
group's total. `cout` isn't a multiple of 32, so the kernel's N-tile loop doesn't apply directly;
covered by zero-padding the weight matrix's `cout` up to 32 (wasting most of a `vrmpy` lane, but
functionally correct and simple: `hex_gemm_kernel.build_kernel()` used completely unchanged with
`cout=32`, and only the first `real_cout` output columns are read). This is exactly the `NT=1`
case the bug above blocked -- fixing that bug is what unlocked this coverage:

| `cin` | `cout` (real) | spatial | stock TVM | `custom_kernel` (padded to 32) | speedup |
|---:|---:|---|---:|---:|---:|
| 256 | 12 | 200x272 | 2.00 GMAC/s | 3.89 GMAC/s | **1.94x** |
| 256 | 3 | 200x272 | 0.80 GMAC/s | 0.97 GMAC/s | **1.22x** |

Both bit-exact correct, both faster than TVM despite computing (and discarding) 20 or 29 unused
output lanes per call -- the `vrmpy` instruction's fixed 32-lane width means the "wasted" lanes
cost nothing extra beyond the one instruction already being issued. The smaller spatial sizes of
these same two shapes (`13x17` through `100x136`, ~46 ms combined) aren't separately verified but
should behave the same way (same kernel, same correctness argument, only the M-loop trip count
differs) -- not measured individually given their small individual impact.

### Coverage: the larger-channel 1x1 convs (`cin`/`cout` > 128)

The earlier coverage passes only looked at shapes with `cin<=128 or cout<=128` (the "small-channel"
systemic-pattern group). Re-checking the *entire* `conv_profile.json` (not just that filtered
subset) by total isolated-timing impact turned up several **larger**-channel 1x1 convs still
running well below the backbone's ~50-80 GMAC/s ceiling -- an earlier version of this note missed
these entirely by filtering too narrowly. The kernel needed zero code changes (it was never
channel-count-limited, only the earlier *search* for targets was) -- verified correct and measured
on real hardware exactly as before:

| `cin` | `cout` | spatial | stride | stock TVM | `custom_kernel` | speedup |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 512 | 200x272 | 2 | 7.54 GMAC/s | 43.80 GMAC/s | **5.81x** |
| 256 | 256 | 200x272 | 1 | 9.78 GMAC/s | 38.00 GMAC/s | **3.89x** |
| 512 | 256 | 100x136 | 1 | 14.40 GMAC/s | 36.73 GMAC/s | **2.55x** |
| 256 | 1024 | 50x68 | 1 | 20.10 GMAC/s | 44.53 GMAC/s | **2.22x** |
| 1024 | 256 | 50x68 | 1 | 24.43 GMAC/s | 35.15 GMAC/s | **1.44x** |

All five bit-exact correct, all faster than TVM -- including the strided `cin=256,cout=512` case
(the `build_strided_kernel()` variant), which got the *largest* speedup of any shape covered so
far (5.81x) despite TVM already doing reasonably well on it in absolute terms (7.54 GMAC/s is
mid-pack, not obviously "broken" the way the smallest-channel shapes were).

**Total real coverage across all twelve verified shapes**: ~3562 ms of the ~7623 ms isolated-timing
total from the original ranked profile -- **about 47% of the entire backbone's isolated-timing
sum**, every one bit-exact correct and faster than TVM's hand-tuned schedule.

### Coverage: a real 3x3 conv (`hex_conv3x3_kernel.py`) -- correct everywhere, fast on some shapes

Every kernel above handles 1x1 convs (a plain GEMM once the spatial dims are flattened into `M`).
3x3 convs are `2509 ms` of the `6851 ms` conv-only profile total (the biggest bucket after 1x1),
so they're the natural next target. `hex_conv3x3_kernel.py` extends the same `vrmpybusv`
accumulate-loop pattern to a genuine spatial convolution:

- **Padding pushed out of the kernel**: the real backbone's 3x3 convs are all `stride=1, pad=1`
  ("same" output size). Rather than branch on boundary conditions inside the hot loop, the input is
  zero-padded by 1 pixel on each spatial side *before* the kernel runs (`pad_input()`) -- once padded,
  `a_row = oh + kh` / `a_col = ow + kw` directly index the padded array for `kh, kw in 0..2`, no `-1`
  offset or bounds check needed. (A real integration into the backbone would need to pad with the
  input's actual quantization zero-point, not literal 0, to match `qnn.conv2d` semantics exactly --
  noted in the file, not yet done, since this file's own correctness check compares against a
  reference padded the same literal-0 way.)
- **Combined reduction range**: one `REDUCE` range of extent `9 * (cin//4)`, decomposed inside the
  kernel via `//`/`%` into `(kh, kw, kc)` -- reuses the exact `_reg_i32` multi-range-dependency
  accumulator-init fix from `hex_gemm_kernel.py` (needed here too: `cin==4` makes this reduction
  range itself degenerate-extent-9, and separately `cout==32` still makes `nt` extent-1).
- **Weight packing extended to 9 positions**: `pack_weight_3x3()` packs each of the 9 kernel
  positions with `pack_b()`'s exact per-position layout (`Wp[pos, nt, kc, n_lane*4+ks]`), so each
  `vrmpybusv` call still reads one contiguous 128-byte weight slice.

Verified **bit-exact correct on real hardware** (`vrmpybusv_acc_128B`, uint8 activation x signed
int8 weight, matching the real backbone's QNN quantization) at the three highest-impact 3x3 shapes
in the profile, at their real spatial sizes. Initial (untiled, `ow_tile=1`) results were a **mixed
result**: correct everywhere, but the single biggest 3x3 bucket in the whole profile
(`cin=cout=256`) was *slower* than stock TVM, the two smaller-channel shapes roughly even to
modestly faster:

| `cin` | `cout` | spatial | stock TVM | untiled (`ow_tile=1`) | speedup |
|---:|---:|---|---:|---:|---:|
| 256 | 256 | 200x272 | 49.30 GMAC/s (0.651 s) | 33.41 GMAC/s (0.960 s) | **0.68x (slower)** |
| 64 | 64 | 200x272 | 20.86 GMAC/s (0.096 s) | 28.96 GMAC/s (0.069 s) | **1.39x** |
| 128 | 128 | 100x136 | 31.87 GMAC/s (0.063 s) | 32.19 GMAC/s (0.062 s) | **1.01x** |

**Why, most likely**: this kernel did zero explicit cache-blocking or output-tile reuse -- a direct
nested loop (`oh -> ow -> nt -> reduction`) with no register tiling across neighboring output
pixels, unlike the 1x1 kernels where every "row" (`M`) is independent and TVM's own baseline was
already memory-bound in a way a single-accumulator loop matches well. A 3x3 conv's per-output-pixel
weight working set is 9x an equivalent 1x1's (`9*256*256 = 589824` packed weight bytes at the
biggest shape -- past a typical Hexagon L1's size, so every output pixel's full reduction re-streamed
weight data from L2/memory with no reuse across pixels).

### Fixing it: output-tile reuse (`ow_tile`)

`build_kernel()`'s new `ow_tile` parameter processes several adjacent output columns per
accumulator group: for each reduction step, the packed weight slice (shared across all tiled output
columns -- it depends only on `nt`/`kc`, not on the output column) is loaded into a local vector
once and reused across `ow_tile` separate `vrmpybusv_acc` calls, one per tiled column with its own
`AddrSpace.REG` accumulator, instead of being re-fetched from memory once per output pixel. Both the
fused accumulate step and the fused output write-back are single `Ops.CUSTOM` statements covering
all `ow_tile` columns at once (see the file for why: an earlier version tried chaining `ow_tile`
separate per-column `CUSTOM` statements via `.after()`, which looked right -- each later write's
source embeds a dependency on the earlier write -- but silently dropped every write except the last
from the rendered output, since `.after()` only orders two nodes that are *already* reachable from
the sink; it doesn't itself make an otherwise-unreferenced void statement reachable. Confirmed by
inspecting the generated C for `ow_tile=2`: only the last column's write appeared, so half the
output columns were simply never written -- exactly the 50% mismatch that bug produced before the
fix).

Tile size was picked using the `HEXSIM=1` BEAM-search timing mode (see "Timing BEAM search
candidates with hexagon-sim instead of raw instruction counting" elsewhere in this file, or PR
https://github.com/onnxsim/onnxsim/pull/1780) rather than guessed -- a genuine, deliberate test of
that infrastructure, not just a demonstration. **Correction**: the sweep below was described as "at
the real `cin=cout=256, 200x272` shape" -- it's actually a small spatial proxy (`8x32`) at the same
`cin=cout=256` (the dimension that actually drives the cache-pressure effect being measured);
`hexagon-sim --timing`'s cycle-accurate simulation of the real kernel's ~31M reduction steps at full
`200x272` scale didn't finish in several minutes (confirmed directly: killed after running past that
without completing), so the sweep never ran at true full scale. `hexagon-sim --timing`'s
Pcycles-derived cost was **non-monotonic** in tile size, real signal a qemu-instruction-count proxy
would not have shown:

| `ow_tile` | Pcycles-derived time | vs untiled |
|---:|---:|---:|
| 1 (untiled) | 7244.75 us | -- |
| 2 | 8265.37 us | **worse** |
| 4 | 5319.82 us | 1.36x better |
| 8 | 4745.00 us | **1.53x better** |

`ow_tile=8` won clearly and was verified bit-exact correct on real hardware at all three profile
shapes, with a uniform ~1.3x real-hardware speedup over the untiled kernel and no regressions:

| `cin` | `cout` | spatial | stock TVM | untiled | tiled (`ow_tile=8`) | vs TVM | vs untiled |
|---:|---:|---|---:|---:|---:|---:|---:|
| 256 | 256 | 200x272 | 49.30 GMAC/s | 33.41 GMAC/s | **44.16 GMAC/s** (0.727 s) | 0.90x (still slower) | 1.32x |
| 64 | 64 | 200x272 | 20.86 GMAC/s | 28.96 GMAC/s | **38.01 GMAC/s** (0.053 s) | **1.82x** | 1.31x |
| 128 | 128 | 100x136 | 31.87 GMAC/s | 32.19 GMAC/s | **42.44 GMAC/s** (0.047 s) | **1.33x** | 1.32x |

Reported honestly: tiling closes *most* of the gap at the biggest shape (0.68x -> 0.90x of TVM) but
not all of it, while pushing the two already-ahead shapes further ahead (1.39x -> 1.82x, 1.01x ->
1.33x). `ow_tile=8` needs 8 concurrent `(32,)` `int32` accumulators live at once (8 HVX vector
registers, 1KB) plus the weight/activation working set -- likely close to Hexagon v73's real
register budget, which is a plausible reason larger tiles weren't swept further.

### Sweeping further: `ow_tile=8` is a real ceiling, not just an unswept guess

Follow-up (same `8x32,cin=cout=256` `HEXSIM=1` proxy as above, for fast iteration -- see the
correction above about why full `200x272` scale isn't practical for this): both of this section's
own "Not done" items were actually tried, and both came back negative, turning the earlier
speculation ("likely close to Hexagon v73's real register budget") into a confirmed, evidenced
finding rather than a guess:

- **Sweeping `ow_tile` past 8**: `ow_tile=16` measured *flat* (0.00475139 s, 0.13% worse than
  `ow_tile=8`'s 0.004745 s -- noise-level) and `ow_tile=32` measured **clearly worse** (0.005719 s,
  20% worse). No further gain past 8; the register-budget hypothesis holds.
- **2D (row+column) tiling at the same total accumulator count**: an `oh_tile=2, ow_tile=4` 2D tile
  (still 8 total accumulators, so the same register pressure as the winning 1D `ow_tile=8`, testing
  whether 2D activation-data locality helps independently of the weight-reuse count) measured
  0.005024 s -- **5.9% worse** than the pure 1D `ow_tile=8` result, not better. The weight-reuse
  factor is what the tiling amortizes (fixed at 8 either way here), and the 2D tile's less-sequential
  activation addressing (a row-stride jump instead of contiguous columns) cost more than any
  locality benefit it might have added.

So this is a genuine ceiling, not an unexplored direction: at this kernel's current register/loop
structure, `ow_tile=8` (already the committed, merged value) is the real local optimum among the
sweep tried, both along the 1D axis and against a same-budget 2D alternative. Closing the remaining
`cin=cout=256` gap (0.90x of TVM) further would need a structurally different approach -- e.g.
reducing the *weight* working set some other way, or restructuring the reduction to touch less
memory per output tile -- not more of the same tiling.

**Still not done**: real end-to-end backbone impact isn't measured (would need the same
splice-into-`relay.build()` mechanism `backbone_splice/` uses, currently blocked on that work's own
unresolved RPC loading bug -- see `backbone_splice/README.md`).

### Coverage: the ResNet stem 7x7 conv (`hex_stem7x7_kernel.py`) -- the last uncovered conv shape

`conv_profile.json`'s last remaining uncovered *conv* bucket (excluding the 3x3 tiling refinement
above): the single ResNet stem conv, `ishape=[1,3,800,1088]`, `wshape=[64,3,7,7]`, `stride=[2,2]`,
`pad=[3,3]`, 308.5 ms. `hex_stem7x7_kernel.py` generalizes `hex_conv3x3_kernel.py`'s pattern from
9 kernel positions to 49, and adds real output-vs-input spatial size mismatch (stride=2, so unlike
the 3x3 kernels' "same"-size case, `(oh, ow)` need their own ranges computed from `(ih, iw)`, not
reused directly -- the same `(oh, ow)` vs `(ih, iw)` split `build_strided_kernel()` in
`hex_gemm_kernel.py` already established for strided 1x1 convs).

One genuinely new wrinkle: `cin=3` isn't a multiple of 4, the K-chunk width every `vrmpybusv` call
in this project assumes. Handled the same way `hex_gemm_kernel.py`'s tiny-`cout` coverage handled
`cout` not being a multiple of 32 ("Coverage: the tiny RPN/mask-head convs" above) -- zero-pad, but
on the reduction (`cin`) axis instead of the N (`cout`) axis this time: `cin` padded up to 4 with
an all-zero weight channel, so the padding lane always contributes `activation * 0 == 0` regardless
of what's in the corresponding activation padding byte. Free for the same reason N-axis padding
was free: `vrmpybusv`'s K-chunk width is fixed at 4 regardless, so `cin=3` already pays for a
4-wide reduction step; padding to `cin=4` just makes that width explicit.

Verified **bit-exact correct on real hardware** (device `239dbd8f`, `vrmpybusv_acc_128B`, uint8
activation x signed int8 weight) at the exact real profile shape, at full scale (`oh=400, ow=544`,
~10.66M reduction steps) -- also bit-exact under qemu at that same full scale (6 seconds to
generate+verify, not just the smaller shapes checked during development):

| `cin` | `cout` | spatial (in) | kernel | stride | stock TVM | `custom_kernel` | speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 64 | 800x1088 | 7x7 | 2 | 2.27 GMAC/s | 31.92 GMAC/s | **14.08x** |

The largest speedup of any shape covered so far, and it makes sense why: TVM's packed NCHWc
schedule pads `cin` up to a full 32-wide input-channel block regardless of the real channel count
(the same packed-layout convention `chunked_kernel_test.py` inspected earlier in this project),
so at `cin=3` it's doing >10x more multiply-accumulate work than necessary padding to 32; this
kernel only pads to 4, the minimum `vrmpybusv` needs, wasting a much smaller fraction.

This closes out every *conv* shape in the profile except the 3x3 tiling refinement above. Adding
this unambiguous win (308.5 ms) to the twelve 1x1 shapes' running total (~3562 ms) gives ~3870.5 ms
of the 7623 ms grand total covered by a kernel that's faster than TVM everywhere it's been
measured -- **about 51%** (the 3x3 conv's ~2509 ms is left out of this figure, same as before,
since one of its three measured shapes is a real, reported loss against TVM, not an unambiguous
win -- see its own section above for the honest breakdown rather than folding it into one summary
percentage).

**Not covered by this file**: the small elementwise/pooling ops (`add`/`maxpool`/`sigmoid`,
~176.7 ms combined, the only remaining uncovered items in `noncon_profile.json` besides `resize`,
which was already handled separately via a TVM-schedule-level fix earlier in this project, not a
`custom_kernel`) -- out of scope here, left for a follow-up.

## Removing TVM as a transport dependency

The bridge above still depends on TVM for two separate things: (1) **transport** -- getting bytes
to and from the DSP at all (`tvm.rpc.tracker.Tracker`, `tvm.contrib.hexagon.build.HexagonLauncher`,
`tvm_rpc_android`, TVM's own `libhexagon_rpc_skel.so`, and TVM's MinRPC wire protocol tunneled
through it), and (2) **codegen coverage** -- convs/ops this session hasn't hand-written a kernel
for yet still come from `relay.build()`. This section is about (1) only.

### What TVM's transport actually is

Reading `src/runtime/hexagon/rpc/hexagon_rpc.idl` and `hexagon_rpc_skel.c` (both in TVM's own
source tree) settles this precisely: the FastRPC interface TVM defines has exactly **five**
methods -- `open`, `close`, `send`, `receive`, `init` (`hexagon_rpc_skel_handle_invoke`'s switch
statement, cases 0-4; anything else returns `AEE_EUNSUPPORTED`). Every actual TVM RPC operation --
loading a module, calling a `PackedFunc`, running a graph -- is **not** a separate FastRPC method.
It's a byte stream TVM's `MinRPCServer` (`src/runtime/minrpc/minrpc_server.h`, instantiated in
`rpc_server.cc`'s `HexagonRPCServer`) decodes on the DSP side, fed through nothing but repeated
`send`/`receive` calls. So "TVM's transport" is really two independent, very differently-sized
things bundled together: a **thin, generic FastRPC shell** (5 methods, entirely reimplementable
from `hexagon_rpc.idl`'s pattern) carrying TVM's own **MinRPC application protocol** (a real wire
format for module loading, `PackedFunc` calls, `DLTensor` argument marshaling -- substantial to
reimplement, and not needed at all if the DSP side doesn't need to run arbitrary `PackedFunc`s).

Given this project's actual use case -- call one fixed, hand-written kernel function with a
handful of buffer pointers, no generic dispatch needed -- reimplementing MinRPC is the wrong
amount of engineering. The right amount is: **write our own tiny FastRPC interface**, generated by
the same `qaic` compiler Qualcomm's SDK ships (the same tool TVM's own build uses to generate
`hexagon_rpc_skel.c`/`hexagon_rpc_stub.c` from `hexagon_rpc.idl`), and drive it from a from-scratch
native client using only `libcdsprpc.so` (Qualcomm's own official, public FastRPC library --
present at `/vendor/lib64/libcdsprpc.so` on-device, not a TVM artifact). See `native_transport/`.

### Proof: a complete, TVM-free round trip on real hardware

`native_transport/mini_rpc.idl` defines a two-method interface (`run_add`, a trivial sanity check;
`run_kernel`, a buffer-in/buffer-in/buffer-out shape matching what a real vrmpy kernel needs).
`qaic` generates `mini_rpc_skel.c` (DSP-side dispatch) and `mini_rpc_stub.c` (host-side call
marshaling) from it -- no TVM code involved in generating either. `mini_rpc_impl.c` is the
DSP-side implementation (currently a placeholder byte-add, standing in for where a real kernel
call would go); it's compiled with `hexagon-clang` and linked with `hexagon-link` directly (the
same two flags TVM's own `tvm.contrib.hexagon.tools.link_shared` passes -- `-Bdynamic -shared
-export-dynamic` plus `libgcc.so` -- read out of `tools.py` and replicated without importing it).
`client_main.c` is a from-scratch native ARM64 program, cross-compiled with the Android NDK,
linked only against `libcdsprpc.so`, that calls the `qaic`-generated stub directly. Verified on
real hardware end to end (device `239dbd8f`), zero TVM Python or C++ in the loop:

```
enable_unsigned_pd rc=0
open OK, handle=0xb400007e59006450
run_add(40,2) rc=0 result=42
run_kernel rc=0 out=11 22 33 44 55 66 77 88
closed OK
```

### Two real bugs found getting here

1. **`AEE_ECONNREFUSED` (114) on every `remote_handle64_open` call, even against TVM's own
   already-proven `libhexagon_rpc_skel.so`** (tested directly, to isolate whether the problem was
   our `.so` or something about the calling process -- it was the latter). Root cause: unsigned
   (unsigned = not Qualcomm-signed) module loading on the CDSP is refused by default and must be
   explicitly enabled *per FastRPC session*, before the first `open`, via
   `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &{domain: CDSP_DOMAIN_ID, enable: 1},
   ...)` (`remote.h`). TVM's own launcher/session code (`apps/hexagon_launcher/launcher_android.cc`,
   `src/runtime/hexagon/rpc/android/session.cc`'s `enable_unsigned_pd()`) already does exactly this
   -- which is *why* the TVM-transport bridge never had to think about it -- but a from-scratch
   client has to make this call itself. This is not the same bug as the `AEE_EUNSUPPORTED`/"method
   5" failure in `../backbone_splice/README.md` (that one happens on an *already-connected*
   TVM session, well after this same enable call has already succeeded internally); still open.
2. **`AEE_EUNABLETOLOAD` (0x406) after fixing (1)**: the generated skel has two *undefined*
   external symbols, `mini_rpc_open`/`mini_rpc_close` -- the `open`/`close` handlers every
   `remote_handle64`-based interface needs (see `hexagon_rpc_skel.c`'s own `case 0`/`case 1`,
   dispatching to `hexagon_rpc_open`/`hexagon_rpc_close`) -- that `mini_rpc_impl.c` simply hadn't
   defined yet. The DSP-side dynamic loader can't load a `.so` with unresolved externals; adding
   trivial `mini_rpc_open`/`mini_rpc_close` implementations (allocate/free a 1-byte handle, mirroring
   TVM's own `hexagon_rpc_open`/`_close` in `rpc_server.cc`) fixed it.

### What this does and doesn't prove

Proven: the FastRPC transport layer -- everything from `libcdsprpc.so` down to a loaded, callable
DSP-side `.so` -- has **no TVM dependency at all**. `tvm.rpc.tracker.Tracker`,
`tvm.contrib.hexagon.build.HexagonLauncher`, and `tvm_rpc_android` can be replaced outright by
`native_transport/`'s pattern for this project's actual use case (call fixed, hand-written
kernels, not arbitrary graphs).

Not yet done: wiring a *real* `hex_gemm_kernel.py`-generated vrmpy kernel (not the placeholder
byte-add) into `mini_rpc_impl.c`'s `run_kernel`, and measuring it end to end through this path
instead of through `bridge_and_test.py`'s TVM-RPC bridge. `run_kernel`'s signature (three
buffer-in/out args) was deliberately shaped to make this a straightforward follow-up: point it at
one of `hex_gemm_kernel.py`'s generated kernel functions instead of the placeholder body.

## Timing BEAM search candidates with hexagon-sim instead of raw instruction counting

`MOCKDSP=1`'s `qemu-hexagon-static` path (used throughout this whole investigation for fast
host-side iteration) times BEAM search candidates via QEMU's `inscount()` pseudo-register -- a
raw instruction count, not a cycle-accurate signal, and structurally blind to Hexagon-specific
pipeline behavior (HVX multi-cycle vector ops, dual/triple-issue slotting) or memory-hierarchy
effects. The 3x3 conv coverage work (`hex_conv3x3_kernel.py`, `codex/hex-gemm-3x3-conv`) found
exactly the case where this matters: the same kernel *shape*, same instruction-count profile, ran
faster than TVM at `cin=64` and **slower** than TVM at `cin=256` on real hardware, purely from
cache/working-set effects a plain instruction count can't see (no output-tile reuse across
pixels -- the fix is real, just not yet built). BEAM search using QEMU's inscount as its cost
signal has no way to detect a difference like that.

### hexagon-sim, and specifically its `--timing` mode

`hexagon-sim` is Qualcomm's own instruction-set simulator (ships in the Hexagon SDK's
`HEXAGON_Tools/*/Tools/bin/hexagon-sim` -- the same simulator this project's separate
`scripts/android/hexagon_sim_harness.py` already uses for TVM-compiled-kernel cycle counts, with
sim-vs-phone speedup correlation previously validated at 2.4x/4.5x/2.5x tracking the real phone's
1.8x/5.1x/2.0x). Its default mode reports a PMU-derived `Pcycles=` total at process exit, but
that default mode turned out to be a fast functional-only estimate, not meaningfully better than
instruction counting for this purpose. `hexagon-sim --help` reveals a separate, undocumented
(from this project's prior usage) `--timing` flag ("Run timing mode") alongside `--timing_nodbc`
("...without data backed cache") and per-component cache trace flags (`--dcachetrace`,
`--l2cachetrace`) -- strong evidence of a real pipeline/cache-hierarchy model gated behind that
flag, distinct from the default.

**Confirmed empirically**, with a minimal, decisive test: two synthetic kernels with the
*identical* instruction count (a fixed 65536-iteration load-increment-store loop), differing only
in whether their working set is a 4KB (L1-resident) or 4MB (cache-hostile, strided) buffer:

| Working set | `--timing` Pcycles-derived time |
|---|---:|
| 4KB, cache-resident | 0.000393 s |
| 4MB, cache-hostile | 0.010925 s |

**~27.8x apart, despite executing the exact same instructions in the exact same order.** Raw
instruction counting (what `MOCKDSP` gives BEAM today) would report these as identical. This is
real evidence `--timing` mode sees a real class of effect instruction counting cannot, in
principle, ever see.

### The tinygrad-side change

`HEXSIM=1` is a new third `DSPDevice` mode (alongside the real phone and `MOCKDSP=1`), landed on
the `onnxsim/tinygrad` fork's `vrmpy-hexagon-support` branch
(https://github.com/onnxsim/tinygrad/pull/1, same PR as the earlier `vrmpy` `TensorCore` work --
stacked onto it rather than a new PR, matching this project's usual pattern for an already-open,
unmerged PR). Mechanism, in `tinygrad/runtime/ops_dsp.py`:

- `HexagonSimRenderer` emits a hosted `main()` (hexagon-sim's standalone-OS mode has real libc,
  unlike `MOCKDSP`'s bare-metal `trap0`-syscall entry) with zero-filled static buffers. This is
  safe because cycle count for a fixed-control-flow kernel (true of every kernel this project
  generates -- conv/gemm with static loop bounds, no data-dependent branches) doesn't depend on
  data *values*, only on the shapes/loop-bounds already baked into the generated source -- so no
  live buffer data needs copying from the caller at all.
- Reading a cycle-counter register live from inside a standalone-sim binary doesn't work (the
  PCYCLE control register pair reads back 0 in this mode -- the same finding
  `hexagon_sim_harness.py` already made), so `HexagonSimCompiler.compile()` compiles the *same*
  kernel wrapper twice (`REPEAT=1` vs `REPEAT=2`, calling the kernel body once vs. twice) and
  `HexagonSimProgram.__call__` runs both ELFs under `hexagon-sim --timing`, returning the
  *difference* in each run's total Pcycles -- the simulator is deterministic, so this exactly
  isolates one kernel invocation's cost and cancels the fixed process-startup overhead (same
  differencing technique `hexagon_sim_harness.py`'s `run_kernel(..., measure_cycles=True)` uses).
  Both ELFs are cached together via the normal `Compiler.compile_cached()` path, so the real
  hexagon-clang + hexagon-sim round trip only happens once per distinct kernel source, not once
  per `__call__`.
- This needed **zero changes to BEAM search itself** -- `Program.__call__`'s return value (a
  float, lower is better) is the only integration point, and it already treats that value as an
  opaque timing signal regardless of which `DSPDevice` mode produced it.

Verified: deterministic across repeated runs of the same compiled ELF pair; scales correctly with
workload (doubling a kernel's inner-loop trip count measured ~1.93-1.97x, not exactly 2x --
plausible with pipeline/dual-issue overlap now being modeled); a real tinygrad-rendered GEMM
kernel run end to end through `Tensor.realize()` under `HEXSIM=1` reports a sane, nonzero,
Hexagon-pipeline-derived per-kernel time through the normal `DEBUG=2` display.

Usage: `HEXSIM=1 DEV=DSP HEXAGON_TOOLS=<Hexagon SDK Tools dir> BEAM=2 python3 your_script.py` --
same shape as `MOCKDSP=1`'s existing usage, just swap the env var. Prefer `HEXSIM=1` over
`MOCKDSP=1` whenever a BEAM search decision plausibly involves a memory/cache tradeoff (tile
sizes, blocking, working-set-sensitive reduction orders); `MOCKDSP=1` remains the faster option
(no real hexagon-clang/hexagon-sim round trip) when only correctness verification is needed, or
when candidates differ only in raw compute-instruction mix with comparable working sets.

## Files

- `capture_kernel.py` -- capture tinygrad's rendered Hexagon C for a shape, verified under qemu.
- `wrapper_template.c` -- the plain-C TVM PackedFunc ABI shim template.
- `bridge_and_test.py` -- compile with the real toolchain, link, deploy via TVM RPC, run, time.
- `vrmpy_tensorcore.patch` -- the tinygrad-side vrmpy `TensorCore` support (also at
  https://github.com/onnxsim/tinygrad/pull/1, branch `vrmpy-hexagon-support`); superseded in
  practice by `hex_gemm_kernel.py` below, which is both correct and fast, but kept as the record
  of the `TensorCore`-machinery dead ends (see "Three more attempts" above).
- `custom_kernel_attempt.py` -- the intermediate, still-broken steps of the `custom_kernel`
  investigation (see its own header comment for the bug-by-bug history); superseded by
  `hex_gemm_kernel.py`.
- `hex_gemm_kernel.py` -- **the working result**: a correct, hand-written Hexagon `vrmpy` GEMM
  kernel via `Tensor.custom_kernel`, measured 8.65x faster than stock TVM's hand-tuned schedule at
  the real Mask R-CNN pathological shape (see "It works" above). Bridge its output through
  `bridge_and_test.py` (after pre-packing `B` via `pack_b()`) to run it on real hardware.
- `hex_conv3x3_kernel.py` -- a real spatial 3x3 conv via the same `vrmpybusv`/`custom_kernel`
  pattern, extended to a 9-position reduction, pre-padded input addressing, and (`ow_tile`)
  output-tile reuse across neighboring output columns. Bit-exact correct at every shape tested;
  faster than stock TVM at two of the three real profile shapes, ~0.90x of TVM (up from 0.68x
  before tiling) at the biggest one (`cin=cout=256`) -- see "Coverage: a real 3x3 conv" above.
- `hex_stem7x7_kernel.py` -- the ResNet stem 7x7 conv, generalizing `hex_conv3x3_kernel.py`'s
  9-position pattern to 49 and handling `cin=3` (not a multiple of 4) via reduction-axis
  zero-padding. Bit-exact correct and 14.08x faster than stock TVM on real hardware at the real
  profile shape -- see "Coverage: the ResNet stem 7x7 conv" above.
- `native_transport/` -- a from-scratch, TVM-free FastRPC transport: custom `qaic`-generated
  interface, a native ARM64 client using only `libcdsprpc.so`, verified end to end on real
  hardware. See "Removing TVM as a transport dependency" above; `native_transport/build.sh`
  reproduces the whole pipeline from source.
