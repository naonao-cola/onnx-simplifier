"""Render the tinygrad fp16 matmul (HMX TensorCore, tinygrad fork branch hvx-hmx) for one shape into tg_kernel.c.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 CC=clang-19 PYTHONPATH=<tinygrad hvx-hmx> python gen_kernel.py M K N out_dir [--i8]

--i8: a uint8 x int8 -> int32 matmul (the hexagon_hmx_i8 ":cm" TensorCore) instead of fp16.

Writes out_dir/tg_kernel.c (the kernel function plus its HMX tile op; everything the renderer puts after
"/* DSP boilerplate */" is dropped) and out_dir/tg_kernel.h with the entry name and shape for tg_hmx_impl.c.
"""
import os, sys, re, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (installs the MOCKDSP capture hooks)
from tinygrad import Tensor, dtypes  # noqa: E402

M, K, N = (int(x) for x in sys.argv[1:4]); out = pathlib.Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
i8 = "--i8" in sys.argv
if i8: Tensor(np.zeros((M, K), np.uint8)).matmul(Tensor(np.zeros((K, N), np.int8)), dtype=dtypes.int32).realize()
else: Tensor(np.zeros((M, K), np.float16)).matmul(Tensor(np.zeros((K, N), np.float16)), dtype=dtypes.half).realize()
assert len(hmxsim._calls) == 1, f"expected one WMMA kernel, got {len(hmxsim._calls)}"
body = hmxsim._calls[0][0].split("/* DSP boilerplate */")[0]
name = re.search(r"noinline\)\) void\s+(\w+)\(", body).group(1)
# tinygrad's int vector typedefs (int32 = 32 lanes of int, ...) collide with the Hexagon SDK's scalar int32 etc. in the skel
body = re.sub(r"\bint(\d+)\b", r"tgint\1", body)
(out / "tg_kernel.c").write_text(body)
ta, tb, tc_ = ("unsigned char", "signed char", "int") if i8 else ("__fp16", "__fp16", "__fp16")
(out / "tg_kernel.h").write_text(f"#define TG_M {M}\n#define TG_K {K}\n#define TG_N {N}\n#define TG_I8 {int(i8)}\n"
                                 f"typedef {ta} tg_a_t;\ntypedef {tb} tg_b_t;\ntypedef {tc_} tg_c_t;\n"
                                 f"void {name}(tg_c_t* out, tg_a_t* a, tg_b_t* b);\n#define TG_KERNEL {name}\n")
print(f"{name}: {len(body)} bytes -> {out}/tg_kernel.c")
