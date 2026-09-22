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
`tvm.contrib.graph_executor.create(...)` fails at runtime with an opaque `hexagon_rpc_send
failed: 78` RPC error. The well-tested `session.get_executor_from_factory()` path (used
successfully everywhere else in this project) isn't exercised by this manual
export_library+load_module+graph_executor.create sequence, so this is most likely a
module-metadata/imports-packing gap specific to using a custom `fcompile`, not a problem with the
splice itself -- the compute/build/link side is proven correct up to this final loading step.

## Next steps if picked back up

- Root-cause the `hexagon_rpc_send failed: 78` error -- likely needs comparing exactly what
  `get_executor_from_factory` does differently at the module-loading level (it may re-derive
  metadata `export_library`'s custom-fcompile path skips, or use a different `.save()`/packing
  route entirely for `ExecutorFactoryModule` objects specifically).
- Once loading works: apply the same splice to all 4 instances of the `cin=64,cout=256` shape in
  the real backbone (`$S/rcnn/backbone_relay.json`), run the *whole* backbone on real hardware,
  and measure genuine end-to-end wall-time improvement against the known baseline (~3.6s).
- Extend to the other 5 covered shapes (`../README.md`'s coverage table) the same way.
