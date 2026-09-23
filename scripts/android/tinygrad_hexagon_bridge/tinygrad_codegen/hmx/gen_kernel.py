"""Render the tinygrad fp16 matmul (HMX TensorCore, tinygrad fork branch hvx-hmx) for one shape into tg_kernel.c.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 CC=clang-19 PYTHONPATH=<tinygrad hvx-hmx> python gen_kernel.py M K N out_dir

Writes out_dir/tg_kernel.c (the kernel function plus its HMX tile op; everything the renderer puts after
"/* DSP boilerplate */" is dropped) and out_dir/tg_kernel.h with the entry name and shape for tg_hmx_impl.c.
"""
import os, sys, re, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (installs the MOCKDSP capture hooks)
from tinygrad import Tensor, dtypes  # noqa: E402

M, K, N = (int(x) for x in sys.argv[1:4]); out = pathlib.Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
Tensor(np.zeros((M, K), np.float16)).matmul(Tensor(np.zeros((K, N), np.float16)), dtype=dtypes.half).realize()
assert len(hmxsim._calls) == 1, f"expected one WMMA kernel, got {len(hmxsim._calls)}"
body = hmxsim._calls[0][0].split("/* DSP boilerplate */")[0]
name = re.search(r"noinline\)\) void\s+(\w+)\(", body).group(1)
(out / "tg_kernel.c").write_text(body)
(out / "tg_kernel.h").write_text(f"#define TG_M {M}\n#define TG_K {K}\n#define TG_N {N}\n"
                                 f"void {name}(__fp16* out, __fp16* a, __fp16* b);\n#define TG_KERNEL {name}\n")
print(f"{name}: {len(body)} bytes -> {out}/tg_kernel.c")
