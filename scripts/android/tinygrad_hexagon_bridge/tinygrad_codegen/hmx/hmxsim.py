"""Run a tinygrad fp16 matmul that uses the HMX TensorCore (tinygrad fork branch hvx-hmx, HMX=1) on hexagon-sim, real data.

qemu (MOCKDSP) can't execute HMX, so the MOCKDSP program call is intercepted (MOCKDSP itself builds the TC's scalar
reference, -DHMX_REF): the rendered kernel and its real argument buffers are captured, then rebuilt with
hexagon-clang -mv69 -mhmx around a hosted main() that points __hmx_vtcm at the simulator's VTCM, sets SSR bit 26
(HMX enable; hmx_lock does this on the phone) and fread()s the buffers, and run on `hexagon-sim -mv69 --mhmx 1 --timing`.
The output is compared bit for bit against the TC's rounding model: an fp16 accumulator, each 32-wide K block added
exactly and rounded once (round to nearest even), which is exactly what one HMX tile op computes.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 CC=clang-19 PYTHONPATH=<tinygrad hvx-hmx> python hmxsim.py 128 576 1536 [--ref]

--ref also runs the MOCKDSP (qemu, scalar reference) build for the same shape.
"""
import os, sys, re, subprocess, tempfile, pathlib
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import to_mv
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime import ops_dsp

TOOLS = pathlib.Path(os.environ.get("HEXAGON_TOOLS", os.path.expanduser("~/.cache/hexagon-oa-19/Tools")))

_src: list[str] = []
_orig_render = ClangRenderer.render
def _cap_render(self, uops):
  s = _orig_render(self, uops); _src.append(s); return s
ClangRenderer.render = _cap_render
_calls: list[tuple[str, list[bytes]]] = []
_orig_init, _orig_call = ops_dsp.MockDSPProgram.__init__, ops_dsp.MockDSPProgram.__call__
def _cap_init(self, dev, obj): _orig_init(self, dev, obj); self._src = _src[-1]
def _cap_call(self, *bufs, vals=(), **kw):
  if "WMMA" in self._src and not os.environ.get("HMXSIM_RUN_REF"):
    _calls.append((self._src, [bytes(to_mv(b.va_addr, b.size)) for b in bufs])); return 0.0
  return _orig_call(self, *bufs, vals=vals, **kw)
ops_dsp.MockDSPProgram.__init__, ops_dsp.MockDSPProgram.__call__ = _cap_init, _cap_call

def model(A:np.ndarray, B:np.ndarray) -> np.ndarray:
  acc = np.zeros((A.shape[0], B.shape[1]), np.float16)
  for k0 in range(0, A.shape[1], 32):
    acc = (acc.astype(np.float64) + A[:, k0:k0+32].astype(np.float64) @ B[k0:k0+32].astype(np.float64)).astype(np.float16)
  return acc

def run_sim(src:str, bufs:list[bytes], repeat:int, work:pathlib.Path) -> tuple[bytes, int]:
  body = src.split("/* DSP boilerplate */")[0]
  name = re.search(r"void\s+(\w+)\(", body.split("#endif")[-1]).group(1)
  for i, b in enumerate(bufs): (work / f"buf{i}.bin").write_bytes(b)
  decl = "\n".join(f"static unsigned char b{i}[{len(b)}] __attribute__((aligned(128)));" for i, b in enumerate(bufs))
  load = "\n".join(f'  {{ FILE* f = fopen("buf{i}.bin", "rb"); fread(b{i}, 1, {len(b)}, f); fclose(f); }}' for i, b in enumerate(bufs))
  (work / "k.c").write_text(f"""#include <stdio.h>
unsigned char* __hmx_vtcm;
unsigned int __hmx_gen = 1;
{body}
{decl}
int main(void) {{
  unsigned base; __asm__ volatile("%0 = cfgbase" : "=r"(base));
  __hmx_vtcm = (unsigned char*)(*(volatile unsigned*)((base << 16) + 0x38) << 16);  /* VTCM base from the config table */
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" :: "r"(r));
{load}
  for (int i = 0; i < {repeat}; i++) {name}({", ".join(f"(void*)b{i}" for i in range(len(bufs)))});
  {{ FILE* f = fopen("out.bin", "wb"); fwrite(b0, 1, {len(bufs[0])}, f); fclose(f); }}
  return 0;
}}
""")
  if os.environ.get("HMXSIM_KEEP"): (pathlib.Path(os.environ["HMXSIM_KEEP"])).write_text((work / "k.c").read_text())
  subprocess.run([str(TOOLS/"bin/hexagon-clang"), "-mv69", "-mhmx", "-mhvx", "-mhvx-length=128B", "-O2", "k.c", "-o", "k.elf"],
                 cwd=work, check=True)
  env = ops_dsp._hexsim_env(TOOLS, work)  # libncurses.so.5 shim, as tinygrad's own HEXSIM path
  r = subprocess.run([str(TOOLS/"bin/hexagon-sim"), "-mv69", "--mhmx", "1", "--timing", "k.elf"], cwd=work, env=env,
                     capture_output=True, text=True, check=True)
  return (work/"out.bin").read_bytes(), int(re.search(r"Pcycles=(\d+)", r.stdout + r.stderr).group(1))

if __name__ == "__main__":
  M, K, N = (int(x) for x in sys.argv[1:4])
  rng = np.random.default_rng(1)
  A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16); B = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
  if "--prepack" in sys.argv:
    # operands stored in HMX tile layout (weights prepacked once on the host; A as a previous HMX op would leave it):
    # Ap[mt, kt, IDX(i, j)] with IDX = (i//2, j, i%2); tinygrad reads the logical A[m, k] through a permuted view
    Ap = A.reshape(M//32, 16, 2, K//32, 32).transpose(0, 3, 1, 4, 2).copy()
    Bp = B.reshape(K//32, 16, 2, N//32, 32).transpose(0, 3, 1, 4, 2).copy()
    a_t = Tensor(Ap).permute(0, 2, 4, 1, 3).reshape(M, K)
    b_t = Tensor(Bp).permute(0, 2, 4, 1, 3).reshape(K, N)
    a_t.matmul(b_t, dtype=dtypes.half).realize()
  else: Tensor(A).matmul(Tensor(B), dtype=dtypes.half).realize()
  assert len(_calls) == 1, f"expected one WMMA kernel, got {len(_calls)}"
  src, bufs = _calls[0]
  ref = model(A, B)
  with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/.cache")) as d:
    out1, c1 = run_sim(src, bufs, 1, pathlib.Path(d))
    _, c2 = run_sim(src, bufs, 2, pathlib.Path(d))
  out = np.frombuffer(out1, np.float16)[:M*N].reshape(M, N)
  bad = int((out.view(np.uint16) != ref.view(np.uint16)).sum())
  cyc = c2 - c1
  print(f"{M}x{K}x{N}: hexagon-sim HMX {bad}/{M*N} bit mismatches vs the TC rounding model; {cyc} Pcycles/call, "
        f"{M*K*N/max(cyc,1):.1f} MAC/cycle ({'PASS' if bad == 0 else 'FAIL'})")
  if "--ref" in sys.argv:
    os.environ["HMXSIM_RUN_REF"] = "1"
    o = Tensor(A).matmul(Tensor(B), dtype=dtypes.half).numpy()
    print(f"  MOCKDSP scalar reference: {int((o.view(np.uint16) != ref.view(np.uint16)).sum())} mismatches")
