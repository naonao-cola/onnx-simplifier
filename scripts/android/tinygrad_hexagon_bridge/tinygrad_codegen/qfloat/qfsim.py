"""Run a plain-tinygrad float kernel on hexagon-sim with real data, for qfloat (HVX_ARCH>=v68) accuracy and cycles.

qemu can't decode HVX float, so instead of executing, the MOCKDSP program call is intercepted: the rendered C source
and the real argument buffers (in kernel call order) are captured, then a hosted main() that fread()s them is built
with hexagon-clang and run on hexagon-sim. Output buffer 0 is read back and compared against numpy/ORT fp32.

  HVX_ARCH=v69 DEV=DSP MOCKDSP=1 python qfsim.py blend 4096
  HVX_ARCH=v69 DEV=DSP MOCKDSP=1 python qfsim.py sigmoid 163200
"""
import os, sys, re, subprocess, tempfile, pathlib, struct
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import to_mv
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime import ops_dsp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qf_ops import op_inputs, reference, ort_reference, errors, POLY  # noqa: E402

TOOLS = pathlib.Path(os.environ.get("HEXAGON_TOOLS", "/home/takecheeze/.cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools"))
SIM_ARCH = os.environ.get("SIM_ARCH", os.environ.get("HVX_ARCH", "v69"))
SHIM = pathlib.Path(os.environ.get("NCURSES5_SHIM", pathlib.Path(__file__).parent / "shim"))

_src: list[str] = []
_orig_render = ClangRenderer.render
def _cap_render(self, uops):
  s = _orig_render(self, uops); _src.append(s); return s
ClangRenderer.render = _cap_render

_calls: list[tuple[str, list[bytes]]] = []
def _cap_call(self, *bufs, vals=(), **kw):
  assert not vals, "symbolic vals not supported"
  _calls.append((self._src, [bytes(to_mv(b.va_addr, b.size)) for b in bufs]))
  return 0.0
_orig_init = ops_dsp.MockDSPProgram.__init__
def _cap_init(self, dev, obj):
  _orig_init(self, dev, obj); self._src = _src[-1]
ops_dsp.MockDSPProgram.__init__ = _cap_init
ops_dsp.MockDSPProgram.__call__ = _cap_call

def tinygrad_op(op:str, ins):
  if op == "blend":
    a, b, c, d, w = (Tensor(x) for x in ins)
    return a*w[:, 0:1] + b*w[:, 1:2] + c*w[:, 2:3] + d*w[:, 3:4]
  if op == "diffprod":
    a, b, c, d = (Tensor(x) for x in ins); return ((a - b) * (c - d)) * ((a + b) * (c + d))
  if op == "poly":
    x = Tensor(ins[0]); y = x * POLY[0]
    for c in POLY[1:-1]: y = (y + c) * x
    return y + POLY[-1]
  return Tensor(ins[0]).sigmoid()

def run_sim(src:str, bufs:list[bytes], repeat:int, work:pathlib.Path) -> tuple[bytes, int]:
  body = src.split("/* DSP boilerplate */")[0]
  name = re.search(r"void\s+(\w+)\(", body).group(1)
  for i, b in enumerate(bufs): (work / f"buf{i}.bin").write_bytes(b)
  decl = "\n".join(f"static unsigned char b{i}[{len(b)}] __attribute__((aligned(128)));" for i, b in enumerate(bufs))
  load = "\n".join(f'  {{ FILE* f = fopen("buf{i}.bin", "rb"); fread(b{i}, 1, {len(b)}, f); fclose(f); }}' for i, b in enumerate(bufs))
  main = f"""#include <stdio.h>
{body}
{decl}
int main(void) {{
{load}
  for (int r = 0; r < {repeat}; r++) {name}({", ".join(f"(void*)b{i}" for i in range(len(bufs)))});
  {{ FILE* f = fopen("out.bin", "wb"); fwrite(b0, 1, {len(bufs[0])}, f); fclose(f); }}
  return 0;
}}
"""
  (work / "k.c").write_text(main)
  subprocess.run([str(TOOLS / "bin/hexagon-clang"), f"-m{SIM_ARCH}", "-mhvx", "-mhvx-length=128B", "-O2", "k.c", "-o", "k.elf"],
                 cwd=work, check=True)
  env = dict(os.environ, LD_LIBRARY_PATH=f"{SHIM}:{os.environ.get('LD_LIBRARY_PATH', '')}")
  r = subprocess.run([str(TOOLS / "bin/hexagon-sim"), f"-m{SIM_ARCH}", "--timing", "k.elf"], cwd=work, env=env,
                     capture_output=True, text=True, check=True)
  cyc = int(re.search(r"Pcycles=(\d+)", r.stdout + r.stderr).group(1))
  return (work / "out.bin").read_bytes(), cyc

def dump(op:str, n:int, src:str, bufs:list[bytes], ins, out:pathlib.Path, tag:str):
  """Write the kernel C (renamed to `tag`) and its argument buffers, for the phone harness (build.sh)."""
  out.mkdir(parents=True, exist_ok=True)
  body = src.split("/* DSP boilerplate */")[0]
  name = re.search(r"void\s+(\w+)\(", body).group(1)
  (out / f"{tag}.c").write_text(re.sub(rf"\b{name}\b", tag, body))
  for i, b in enumerate(bufs): (out / f"{tag}_buf{i}.bin").write_bytes(b)
  (out / f"{tag}.meta").write_text(f"{tag} {len(bufs)} " + " ".join(str(len(b)) for b in bufs) + "\n")
  np.save(out / f"{tag}_ref64.npy", reference(op, ins, np.float64))

if __name__ == "__main__":
  op, n = sys.argv[1], int(sys.argv[2])
  ins = op_inputs(op, n)
  tinygrad_op(op, ins).realize()
  assert len(_calls) == 1, f"expected one kernel, got {len(_calls)}"
  src, bufs = _calls[0]
  if len(sys.argv) > 4 and sys.argv[3] == "--dump":
    dump(op, n, src, bufs, ins, pathlib.Path(sys.argv[4]), sys.argv[5]); print(f"dumped {sys.argv[5]}"); sys.exit(0)
  with tempfile.TemporaryDirectory() as d:
    work = pathlib.Path(d)
    out1, c1 = run_sim(src, bufs, 1, work)
    _, c2 = run_sim(src, bufs, 2, work)
  ref32, ref64 = reference(op, ins, np.float32), reference(op, ins, np.float64)
  got = np.frombuffer(out1, dtype=np.float32).reshape(ref32.shape)
  qf = "qfloat" if "__hvx_" in src else "scalar"
  print(f"{op} n={n} arch={SIM_ARCH} code={qf} kernel_pcycles={c2 - c1}")
  a, r, ex = errors(got, ref32); print(f"  vs numpy fp32: max_abs={a:.3g} max_rel={r:.3g} exact_frac={ex:.4f}")
  a, r, _ = errors(got, ref64); print(f"  vs float64:    max_abs={a:.3g} max_rel={r:.3g}")
  a, r, _ = errors(ref32, ref64); print(f"  (numpy fp32 vs float64: max_abs={a:.3g} max_rel={r:.3g})")
  if (o := ort_reference(op, ins)) is not None:
    a, r, ex = errors(got, o.reshape(ref32.shape)); print(f"  vs ORT fp32:   max_abs={a:.3g} max_rel={r:.3g} exact_frac={ex:.4f}")
