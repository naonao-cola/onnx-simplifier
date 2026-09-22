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

**Known limitation**: fails when `cout == 32` exactly (a degenerate single-iteration N-tile loop)
-- not investigated further since it doesn't affect the real target shape (`cout=256`, 8 N-tiles).

This is real evidence the approach itself is sound (it gets *past* both the shape system and the
UOp spec verifier, further than any of the three `TensorCore`-machinery attempts got before hitting
their own, different dead ends) -- what's blocking it now is either a genuine tinygrad scheduler bug
or a kernel-structure choice this investigation hasn't found yet. Picking this back up means
understanding the linearizer's sibling-range ordering logic well enough to route around it, or
fixing the underlying scheduler issue upstream (which would also fix it for `amd.py`'s equally
affected `shufflenet` case, per the comment).

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
