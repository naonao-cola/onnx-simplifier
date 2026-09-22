"""SUPERSEDED by hex_gemm_kernel.py, which is the working, correct, and fast result of this
investigation (8.65x faster than stock TVM at the real Mask R-CNN shape -- see ../README.md's
"It works" section). This file is kept as the historical record of the intermediate, still-broken
step (v5) that got past the shape-broadcast and UOp-verification issues but not yet the
control-flow linearizer assertion described below -- useful context for the bug-by-bug history in
../README.md's "Modeling Hexagon as its own accelerator" section, not something to run as-is.

Attempt at modeling Hexagon vrmpy as its own hand-written custom_kernel path (see
tinygrad/llm/kernels/amd.py for the established pattern this follows), bypassing
Ops.WMMA/TensorCore/the generic devectorizer entirely -- motivated by three earlier, independent
dead ends trying to fix the WMMA accumulator's scalar-decomposition problem within the shared,
generic machinery (see ../README.md's "Three more attempts" section). Requires a tinygrad
checkout with vrmpy_tensorcore.patch applied, though this script itself never uses Ops.WMMA.

STATE AT THIS FILE (v5): progressed through five iterations, each fixing a real, understood bug:
  v1/v2: UOp shape-broadcast mismatch (32,)/(128,)/(1,) from calling .load() on differently-
         shaped array values before passing them to the CUSTOMI op.
  v3/v4: fixed by passing raw Ops.INDEX (shape ()) instead of .load()'d values -- but then a UOp
         spec-verification failure on `acc.load()` (LOAD of an AFTER-wrapped value).
  v5 (this file): fixed by dropping .load() entirely, matching amd.py's real idiom exactly
         (`acc.after(offset)[head]`, no .load()) -- now fails LATER, in the control-flow
         linearizer, on an assertion whose own comment says "TODO: this can happen! it causes
         infinite loop in shufflenet".

RESOLUTION (in hex_gemm_kernel.py, not this file): that assertion turned out to be triggered by
calling `.end(kc)` on the reduction range twice (once implicitly via the accumulate step, again
in the final store) -- not an unrelated tinygrad scheduler bug after all. Fixing that, plus three
more bugs (AxisType.GLOBAL needing renderer support ClangRenderer lacks; a stray `&` on an
already-pointer INDEX expression; and the real semantic fix -- representing the whole accumulate
step as one void-dtype Ops.CUSTOM statement addressed by pointer, never a (32,)-shaped VALUE the
generic elementwise devectorizer tries to decompose per-lane), got to a correct, fast kernel.
See ../README.md for the full bug-by-bug list."""
import os
os.environ["DEV"] = "DSP"
os.environ["MOCKDSP"] = "1"
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.renderer.cstyle import ClangRenderer

captured = {}
_orig = ClangRenderer.render
def _cap(self, uops):
    src = _orig(self, uops)
    captured["src"] = src
    return src
ClangRenderer.render = _cap

def _reg_i32(shape, slot, dep=None):
    ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
    return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(0)))

def _hex_vrmpy(acc, a_idx, b_idx):
    assert a_idx.op is Ops.INDEX and b_idx.op is Ops.INDEX
    return UOp(Ops.CUSTOMI, dtypes.int32, (acc, b_idx, a_idx), arg=
        "__builtin_HEXAGON_V6_vrmpyub_acc_128B({0}, *(unsigned char128*)&{1}, *(unsigned int*)&{2})")

def hex_gemm_kernel(C:UOp, A:UOp, Bp:UOp) -> UOp:
    acc = _reg_i32((32,), slot=0)
    kc = UOp.range(16, 0, AxisType.REDUCE)
    prev = acc.after(kc)          # no .load(): the AFTER-scoped placeholder IS the value
    a_idx = A[0, kc*4]
    b_idx = Bp[kc, 0]
    update = acc.store(_hex_vrmpy(prev, a_idx, b_idx)).end(kc)
    final = acc.after(update)     # again, no .load()
    return C[0, :].store(final).end(kc).sink(arg=KernelInfo(name="hex_gemm_min", opts_to_apply=()))

cin, cout = 64, 32
rng = np.random.default_rng(7)
a_np = rng.integers(0, 100, (1, cin)).astype(np.uint8)
b_np = rng.integers(0, 100, (cin, cout)).astype(np.uint8)
b_packed = b_np.reshape(cin // 4, 4, cout).transpose(0, 2, 1).reshape(cin // 4, cout * 4)

C = Tensor.empty(1, cout, dtype="int32", device="DSP")
A = Tensor(a_np, device="DSP")
Bp = Tensor(b_packed, device="DSP")
try:
    out = Tensor.custom_kernel(C, A, Bp, fxn=hex_gemm_kernel)[0]
    out.realize()
    ref = a_np.astype(np.int64) @ b_np.astype(np.int64)
    print("correct:", np.array_equal(out.numpy().astype(np.int64), ref[0]))
    print("got     ", out.numpy()[:8])
    print("expected", ref[0][:8])
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:800])
    raise
finally:
    if captured:
        open("min_custom_kernel5.c", "w").write(captured["src"])
        print("saved kernel source")
