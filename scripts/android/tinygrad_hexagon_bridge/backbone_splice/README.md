# Splicing a hand-written kernel directly into a real `relay.build()`

Follow-up to `../hex_gemm_kernel.py`'s isolated-shape results: instead of running the custom
`vrmpy` kernel as its own standalone op (bridged in via TVM's transport, as `../README.md`
documents), this splices it *directly into TVM's own compiled backbone*, replacing one op's
compute in an otherwise normal `relay.build()` -- so the whole backbone, kernel included, comes
out as a single compiled artifact deployable the normal way.

## The key finding: zero-copy compatible layouts

TVM's Hexagon int8 conv2d packs its buffers before scheduling:

```
data_vec:   [1, cin//32, H, W, 32]                      uint8
kernel_vec: [cout//32, cin//32, kh, kw, 8, 32, 4]        int8
```
(`kh=kw=1` for the 1x1 convs this targets; verified by direct inspection, not just reading the
compute definition's docstring.)

For fixed `(oc_chunk, ic_chunk, kh=0, kw=0)`, `kernel_vec`'s last two dims `(32, 4)` are exactly
128 contiguous bytes in `(n_lane, k_sub)` order -- **exactly** the layout `hex_gemm_kernel.py`'s
`pack_b()` produces. And `data_vec`'s `32`-wide last dim, for a fixed `(ic_chunk, h, w)`, is
exactly 32 contiguous input-channel bytes, subdividing cleanly into 8 four-byte `vrmpy` groups.
This isn't a coincidence -- both are shaped by the same `vrmpy` hardware width -- and it means the
kernel can read TVM's own packed buffers **directly, with zero repacking**, via chunked address
arithmetic (`ic_chunk = kc // 8`, `ic_within = kc % 8`) instead of the flat-`M`/`pack_b()`
addressing `hex_gemm_kernel.py` uses standalone.

`gen_chunked_prod.py` builds and verifies this chunked-addressing kernel variant at the real
backbone scale (`cin=64, cout=256`, `H=200, W=272`) against a numpy reference -- correct.

**Bug found along the way**: `vrmpybusv` (the signed-weight `vrmpy` variant real backbone weights
need, vs. the unsigned `vrmpyub` used in `hex_gemm_kernel.py`'s earlier tests) has a *different*
calling convention than documented by analogy to `vrmpyub` -- its "A" operand must be
broadcast to a full 32-wide vector first (matching TVM's own `dot_vrmpy` intrinsic exactly), not
passed as a plain scalar. Fixed via a `.format()`-escaped vector-literal broadcast in the
`Ops.CUSTOM` format string.

## The splice

`splice_test.py` monkeypatches `topi.hexagon.conv2d.conv2d_NCHWc_int8` (the *compute* definition,
not just the schedule -- letting TVM's own `AlterOpLayout` pass produce the packed buffers
normally) to return a `te.extern(...)` node calling the precompiled kernel by name, for shapes
matching the target `(cin, cout)`; everything else falls through to TVM's stock compute.

**Proven working**: `relay.build()` accepts the splice and successfully lowers/compiles the whole
graph (`te.extern`'s raw `tir.call_extern` survives TVM's own compilation pipeline intact).
Linking also works, via a custom `fcompile` wrapper needed for two reasons: (1) `export_library`'s
kwarg-forwarding doesn't match `link_shared`'s TVM-FFI-wrapped (`PackedFunc`) calling convention,
needing a plain-Python shim; (2) `export_library` sometimes hands the linker a `devc.c` module-
packing fallback file directly, which `link_shared` (a pure linker wrapper, not a compiler) can't
handle -- the wrapper compiles any `.c` inputs with the Hexagon toolchain first.

**Not yet resolved**: loading and running the spliced `.so` via `session.load_module()` +
`tvm.contrib.graph_executor.create(...)` fails at runtime -- narrowed significantly, still open.

### Diagnosing the runtime failure

Bisecting the failing call step by step (`diag1.py`/`diag2.py`/`diag3.py`, not committed --
scratch scripts, reproducible from the pattern below) showed:

```
session.upload(...)          OK
session.load_module(...)     OK   -- the module DOES load and IS runnable on the DSP
graph_executor.create(...)   OK
gm.load_params(...)          OK
gm.set_input(...)            OK
gm.run()                     FAILS
```

So the module itself loads fine -- the failure is specifically in *running* the graph. `adb
logcat` around the failure shows the real underlying error, one level below TVM's opaque
`hexagon_rpc_send failed: N` wrapper:

```
tvm_rpc_android: .../fastrpc_apps_user.c:1323: Error 0x27: remote_handle64_invoke failed
  for handle 0x68004180, method 5 on domain 3 (sc 0x5000100) (errno Success)
```

`0x27` = 39 = Qualcomm's `AEE_EUNSUPPORTED` -- the DSP-side FastRPC skeleton is rejecting
"method 5" for this call, repeated many times for the same handle (once per graph-execution
step, i.e. per op/buffer-management call TVM's runtime makes while running the graph -- not a
crash in the kernel's own compute, which is never reached).

**Ruled out** (both implemented and tested on real hardware, neither fixed it):
- **HVX alignment**: `te.extern`'s auto-allocated in/out buffers might not get the 128-byte
  alignment HVX vector load/store needs (unlike TVM's own vrmpy-tensorized buffers, which do).
  Passed explicit `tvm.tir.decl_buffer(..., data_alignment=128, offset_factor=128)` via
  `te.extern`'s `in_buffers`/`out_buffers` params. No change.
- **`call_extern` return-type ABI mismatch**: the kernel function is `void`
  (`__attribute__((noinline)) void hex_gemm_...(...)`, matching every other kernel in this
  session), but `tvm.tir.call_extern("int32", "hex_gemm_...", ...)` declares it as returning
  `int32` -- a real ABI mismatch, fixed by changing the dtype to `"void"`. Still failed
  identically afterward, so this wasn't the (sole) cause either, though it's a correct fix to
  keep regardless.

Since the error happens on a repeated `method 5` FastRPC call *before* the kernel's own compute
would run, and *not* while the (correctly loaded, per `load_module` succeeding) module's data is
being used, the most likely remaining explanation is a **structural difference between how
`relay.build()`'s own internal Hexagon codegen links a module and how this session's manual
`export_library(fcompile=..., addons=[...])` does** -- e.g. missing device-module metadata
(`__tvm_dev_mblob`-style embedding, or similar) that the DSP-side skel's dispatch path for
"method 5" (likely graph-executor-specific buffer/workspace setup, not a data compute call)
relies on and that TVM's *own* internal Hexagon link step (invoked inside
`codegen_hexagon.cc`, confirmed to call `tvm.contrib.hexagon.link_shared` itself as part of
`relay.build()`, before this session's script ever gets a chance to intervene) embeds
correctly but a from-scratch `export_library` re-link does not reproduce.

## Third pass: still open, but significantly narrowed, plus a real fallback landed

A third investigation pass (following up on the two above) did the C++-level reading option (a)
named below, and used the meanwhile-completed `../native_transport/` work (a from-scratch,
TVM-free FastRPC client) as new leverage this bug's first two passes didn't have. Two real,
permanent fixes landed in `splice_test.py` regardless of the outcome below:

- **The `call_extern` return-type ABI bug (int32 vs the kernel's actual void) was never actually
  fixed in the committed script** -- the "fixed to `void`... still failed identically" note above
  described a fix that only ever existed in the uncommitted scratch diagnostic scripts
  (`diag1.py`/`diag2.py`/`diag3.py`), never landed in `splice_test.py` itself. Fixed for real this
  time (`"int32"` -> `"void"` in the `call_extern` call).
- **Isolated the spliced-vs-stock comparison into two separate `Session`s** instead of one --
  mixing a manually-loaded custom `.so` with `get_executor_from_factory()` in a single session was
  independently found (in this project's elementwise-add coverage work) to cause spurious
  `hexagon_rpc_send` failures of its own, unrelated to either module. `splice_test.py` was doing
  exactly that; splitting it removes that confound from this test permanently.

### Decisive new test: the manual loading path itself is not the problem

The standing hypothesis (a structural difference between `relay.build()`'s own internal Hexagon
link step and this session's manual `export_library(fcompile=..., addons=[...])` re-link) was
never actually tested in isolation before -- both prior passes only ever ran the *spliced* module
through the manual path. Running a **completely unmodified, stock TVM-compiled module** (no
splice, no custom kernel, same `fcompile_wrapper`) through the exact same
`session.load_module()` + `graph_executor.create()` + `gm.run()` sequence, in a fresh session,
**succeeds outright** -- `RUN OK`, correct output, no `AEE_EUNSUPPORTED`, nothing. This
conclusively rules out "the manual loading path is generally broken" as an explanation: it works
fine on its own. Whatever's wrong is specific to the *spliced* module, not the loading pattern.

### The failure, decoded precisely

Rerunning the spliced test (with both fixes above applied) still fails at `gm.run()`, same
symptom as before -- but fresh `adb logcat` this time around shows **two distinct handles**,
where only one line was ever examined previously:

```
Error 0x27: remote_handle64_invoke failed for handle 0x69204180, method 5 on domain 3 (sc 0x5000100)
Error 0x27: remote_handle64_invoke failed for handle 0x692040c0, method 2 on domain 3 (sc 0x2020000)
```

Decoding `sc` via QAIC's actual scalar bit layout (`REMOTE_SCALARS_METHOD`/`INBUFS`/`OUTBUFS`):
`0x5000100` = method 5, **0 input buffers, 1 output buffer**, 0 handles either way -- a
zero-argument "give me a small result" call, repeated dozens of times in a tight loop (once per
graph-execution step, matching the original finding). `0x692040c0`'s `method 2` matches
`hexagon_rpc_skel.c`'s own `send` case exactly -- almost certainly a secondary/cascade failure on
the *main* session handle while it tries to report the first error back over the wire, not an
independent bug.

Two hypotheses this new evidence lets rule out cleanly, both now confirmed *false*:
- **"A second `remote_handle64_open` gets called somewhere in TVM's own C++."** Grepping the
  entire TVM source tree (`src/`, `apps/`) for `remote_handle64_open` turns up exactly one call
  site -- the host-side stub in `hexagon_rpc_stub.c`. TVM's C++ never opens a second handle
  itself; if `0x69204180` is a second handle, something outside TVM's own explicit code path
  opens it.
- **"The spliced kernel's HVX use never gets 'powered on' because `te.extern` bypasses whatever
  bookkeeping TVM's own codegen normally inserts for HVX-using compute."** `HexagonPowerManager`
  (`hexagon_power_manager.cc`) calls `PowerOnHVX()` unconditionally in its own constructor, which
  runs during `Session.__enter__`'s `device_api.hexagon.acquire_resources` call -- **once per
  session, before any module is even loaded**, not per-kernel or per-module. HVX is already fully
  powered on by the time `load_module`/`set_input` (which already succeed) run.

What's left unexplained: what specifically calls `method 5` on that second handle during
`gm.run()`, dozens of times, only for the spliced module. The trail leads into either a
Qualcomm-proprietary system library (most plausibly something in the DCVS/power-request or
VTCM/DMA-allocation path triggered specifically by executing our externally-linked kernel's HVX
instructions, as opposed to TVM's own tensorized/codegen-inserted HVX use) or some other
TVM-runtime-internal per-op dispatch path not yet identified -- genuinely still open, at the same
kind of closed-source boundary both prior passes hit, just narrowed much further than before.

### The fallback: real value delivered a different way

Given a third serious attempt still didn't land the root cause, and `../native_transport/` (a
separate, since-completed effort to remove TVM as a transport dependency entirely) already proved
a working, TVM-free alternative for calling a fixed hand-written kernel, that fallback was
completed instead: a **real** `hex_gemm_kernel.py`-generated kernel (not a placeholder) now runs
end to end through `native_transport/`, at the exact flagship shape this splice work targets
(`cin=64,cout=256,m=54400`), verified **bit-exact correct on real hardware** against a numpy
reference. See `../README.md`'s "Removing TVM as a transport dependency" section for the result.
This sidesteps the bug above rather than fixing it -- the two are not the same outcome, and this
README keeps them distinct rather than conflating a bypass with a fix.

## Next steps if picked back up

- Root-causing "method 5": would need either (a) instrumenting/tracing the actual DSP-side
  syscalls or FastRPC calls made specifically during `gm.run()` for the spliced module (e.g. via
  the Hexagon simulator's tracing flags now known to exist, see `../README.md`'s hexagon-sim
  section, if the failure reproduces there too) to see exactly what triggers the second handle, or
  (b) finding a way to get the extra kernel object file linked into TVM's *own* internal Hexagon
  codegen link step (inside `codegen_hexagon.cc`) instead of re-linking separately via
  `export_library` -- untested in isolation, though now a weaker lead than before given the stock
  module (which *does* go through that same manual `export_library` path) runs fine.
- Given the fallback above already delivers the underlying goal for one shape, the more valuable
  next step is probably extending `native_transport/`'s real-kernel pattern to the other 5 covered
  shapes and to a genuine multi-op graph (not just one splice into TVM's compiled graph), rather
  than continuing to chase this specific TVM-internal bug.
