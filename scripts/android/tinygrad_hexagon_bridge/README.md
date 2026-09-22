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
before every accumulate call. Confirmed this isn't specific to BEAM's automatic
"interleave-for-ILP" upcast step -- a minimal, non-interleaved single-TC-call test (bypassing
`hand_coded_optimizations` entirely, applying only `Opt(OptOps.TC, ...)`) still shows the same
scalar rebuild. BEAM search independently corroborates the regression: across 15+ candidates
explored at BEAM=2, it never selects the WMMA path, consistently preferring plain scalar codegen
-- exactly matching what the real-hardware measurement shows.

A real fix needs either the generic devectorizer or `postrange.py`'s `_apply_tc_opt`
(accumulator axis-ordering/layout) to recognize single-thread, vector-native tensor cores as a
distinct case from warp-distributed ones -- both are shared across every backend tinygrad
supports, so this is real, if bounded, upstream-tinygrad-core work, not a local patch. Not
attempted here; documented as the precise next step, including a stopping point that was reached
deliberately rather than a task that was abandoned mid-diagnosis.

## Files

- `capture_kernel.py` -- capture tinygrad's rendered Hexagon C for a shape, verified under qemu.
- `wrapper_template.c` -- the plain-C TVM PackedFunc ABI shim template.
- `bridge_and_test.py` -- compile with the real toolchain, link, deploy via TVM RPC, run, time.
- `vrmpy_tensorcore.patch` -- the tinygrad-side vrmpy `TensorCore` support (also at
  https://github.com/onnxsim/tinygrad/pull/1, branch `vrmpy-hexagon-support`).
