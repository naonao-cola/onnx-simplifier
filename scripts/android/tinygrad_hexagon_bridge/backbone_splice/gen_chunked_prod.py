"""Generate the production chunked-addressing kernel for cin=64,cout=256 @200x272 (the flagship
backbone shape, 4 instances) -- reads TVM's own packed NCHWc int8 buffers directly, no repacking."""
import os
os.environ["DEV"] = "DSP"
os.environ["MOCKDSP"] = "1"
import functools
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

I32X32 = "int __attribute__((vector_size(128)))"
U8X128 = "unsigned char __attribute__((vector_size(128)))"
IC_BN = 32

def build_chunked_kernel(cin, cout, H, W, a, kernel_vec, kernel_name):
    ic_chunks, oc_chunks = cin // IC_BN, cout // 32
    kc_per_chunk = IC_BN // 4

    def kernel_fn(C: UOp, A: UOp, K: UOp) -> UOp:
        h = UOp.range(H, 0, AxisType.WEAK)
        w = UOp.range(W, 1, AxisType.WEAK)
        nt = UOp.range(oc_chunks, 2, AxisType.WEAK)
        acc = _reg_i32((32,), slot=0, dep=nt)
        kc = UOp.range(cin // 4, 3, AxisType.REDUCE)
        acc_addr = acc.after(kc)[0]
        ic_chunk = kc // kc_per_chunk
        ic_within = kc % kc_per_chunk
        a_flat_idx = ic_chunk * (H * W * IC_BN) + (h * W + w) * IC_BN + ic_within * 4
        k_flat_idx = nt * (ic_chunks * kc_per_chunk * 32 * 4) + ic_chunk * (kc_per_chunk * 32 * 4) + ic_within * (32 * 4)
        a_idx = A[a_flat_idx]
        k_idx = K[k_flat_idx]
        broadcast32 = ",".join(["*(unsigned int*){2}"] * 32)
        arg_str = (
            f"*({I32X32}*){{0}} = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(*({I32X32}*){{0}}, "
            f"({I32X32}){{{{{broadcast32}}}}}, *({U8X128}*){{1}});"
        )
        step = UOp(Ops.CUSTOM, dtypes.void, (acc_addr, k_idx, a_idx), arg=arg_str)
        update = step.end(kc)
        final_addr = acc.after(update)[0]
        c_flat_idx = nt * (H * W * 32) + (h * W + w) * 32
        out_stmt = UOp(Ops.CUSTOM, dtypes.void, (C[c_flat_idx], final_addr), arg=f"*({I32X32}*){{0}} = *({I32X32}*){{1}};")
        return out_stmt.end(nt, w, h).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(oc_chunks * H * W * 32, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a, kernel_vec, fxn=functools.partial(kernel_fn))[0]

def pack_b(b, n_tile=32, k_sub=4):
    K, N = b.shape
    return b.reshape(K // k_sub, k_sub, N // n_tile, n_tile).transpose(2, 0, 3, 1).reshape(N // n_tile, K // k_sub, n_tile * k_sub)

cin, cout, H, W = 64, 256, 200, 272
ic_chunks, oc_chunks = cin // 32, cout // 32
rng = np.random.default_rng(11)
data_img = rng.integers(0, 200, (cin, H, W)).astype(np.uint8)
weight = rng.integers(-40, 40, (cout, cin)).astype(np.int8)
data_vec_np = data_img.reshape(ic_chunks, 32, H, W).transpose(0, 2, 3, 1).reshape(-1)
kernel_vec_np = pack_b(weight.T.copy()).reshape(-1).astype(np.uint8)  # store as raw bytes (signed bits preserved)

A = Tensor(data_vec_np, device="DSP")
K = Tensor(kernel_vec_np, device="DSP")
out = build_chunked_kernel(cin, cout, H, W, A, K, "hex_gemm_chunked_64_256")
out.realize()
ref = np.einsum("oc,chw->ohw", weight.astype(np.int64), data_img.astype(np.int64))
out_np = out.numpy().reshape(oc_chunks, H, W, 32).transpose(0, 3, 1, 2).reshape(cout, H, W)
match = np.array_equal(out_np.astype(np.int64), ref)
print("correct at full scale (H=200,W=272):", match)
if not match:
    diff = out_np.astype(np.int64) - ref
    print("max abs diff", np.abs(diff).max(), "mismatched", np.count_nonzero(diff), "/", diff.size)

marker = captured["src"].find("/* DSP boilerplate */")
open("kernel_chunked_64_256.c", "w").write(captured["src"][:marker].rstrip() + "\n")
print("saved kernel source")
