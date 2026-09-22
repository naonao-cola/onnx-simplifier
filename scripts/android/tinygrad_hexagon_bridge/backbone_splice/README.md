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

## Next steps if picked back up

- This looks like it needs either: (a) C++-level reading of `rpc_module.cc` /
  `runtime/hexagon/rpc/*` to find what "method 5" actually dispatches to and what state it
  expects the loaded module to be in, or (b) finding a way to get the extra kernel object file
  linked into TVM's *own* internal Hexagon codegen link step (inside `codegen_hexagon.cc`)
  instead of re-linking separately via `export_library` -- which would make the resulting
  module identical in structure to a normal `relay.build()` output, sidestepping the
  metadata-gap hypothesis entirely rather than needing to diagnose it.
- Once loading works: apply the same splice to all 4 instances of the `cin=64,cout=256` shape in
  the real backbone (`$S/rcnn/backbone_relay.json`), run the *whole* backbone on real hardware,
  and measure genuine end-to-end wall-time improvement against the known baseline (~3.6s).
- Extend to the other 5 covered shapes (`../README.md`'s coverage table) the same way.
