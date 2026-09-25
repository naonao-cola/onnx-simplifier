"""A static QDQ ONNX (onnxsim full_qdq + quantized_io, the hmx_gemm runner's input: Conv k x k s1/s2, Add, MaxPool) as a
tinygrad graph on the DSP, lowered by the fork's HMX int8 TensorCore rules -- the same model the hand runner
(hmx_gemm/runner) executes, through tinygrad UOps instead of hand kernels.

Layout: every activation is a padded flat grid -- NHWC, a ring of `pad` pixels holding its zero point, flattened to
(rows, C) at row stride Wp = W + 2*pad, plus a tail the consumers' windows may overrun into. A k x k / stride s conv is
tinygrad's ordinary (A * W).sum() over the windowed view A(p, dy, dx, c) = x[s*p + dy*Wp + dx, c] (hmxsim_conv.py), on an
output grid at the input's row stride; a copy crops that grid into the next padded tensor. The stride-2 stem runs as a
stride-1 4x4 conv on the input's 2x2 phase split (4 phases x 8 padded channels = one 32-byte K block per tap), written
straight into MaxPool's padded input. Adds are ops_dsp.hmx_qlinear_add over whole padded buffers (the pads of zp_a + zp_b come out as
zp_y: checked per Add). MaxPool is the windowed view's max.

The reference is qdq_graph.py's exact emulator (itself checked against ORT CPU).

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC=clang-19 PYTHONPATH=<tinygrad> python qdq_net.py model.onnx outdir
writes outdir/k*.c (one kernel each), outdir/graph.h (buffers, constants blob layout, call list), outdir/blob.bin.
"""
import sys, os, re, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "hmx_gemm" / "runner"))
import qdq_graph  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.helpers import getenv  # noqa: E402
from tinygrad.runtime.ops_dsp import hmx_qlinear_add  # noqa: E402

f32 = np.float32
r64 = lambda n: (n + 63) // 64 * 64

class Act:
  """a padded flat grid: t (L, C) uint8, H x W pixels inside a ring of `pad` (wider on the right when Wp > W + 2 pad), row
  stride Wp"""
  def __init__(self, t, H, W, C, pad, zp, Wp=None):
    self.t, self.H, self.W, self.C, self.pad, self.zp = t, H, W, C, pad, zp
    self.Wp = Wp or W + 2 * pad

def need_rows(H, W, pad, k, s, kx=None, Wp=None):
  """rows of a padded grid (H x W, ring pad, row stride Wp = W + 2 pad by default) a k x kx / s conv or pool reads: its output
  grid is Ho x Wp pixels padded to 64, pixel p reading s*p + base + dy*Wp + dx, base = the window's top-left in the ring"""
  Wp, Ho, kx = Wp or W + 2 * pad, (H - 1) // s + 1, kx or k
  base = (pad - k // 2) * (Wp + 1)
  return base + s * (r64(Ho * Wp) - 1) + (k - 1) * Wp + kx

def canon(g, Ho, Wo, Wg, C, zp, pad, L):
  """output grid (rows at stride Wg, Ho x Wo valid) -> a padded flat grid of L rows (ring `pad` and tail = zp)"""
  x = g[: Ho * Wg].reshape(Ho, Wg, C)[:, :Wo]
  x = x.pad(((pad, pad), (pad, pad), (0, 0)), value=zp).reshape(-1, C)
  return x.pad(((0, L - x.shape[0]), (0, 0)), value=zp).contiguous()

def into_grid(y:Tensor, H, W, C, zp, L, Wp=None) -> Tensor:
  """a stride-1 conv's output grid (row stride Wp = W + 2, pixel p = i*Wp + j) written straight into the next padded grid:
  pixel (i, j) belongs at (i+1)*Wp + (j+1) = p + Wp + 1, so the grid goes in at that offset and its 2 garbage columns per row
  land on the pad columns; then the ring (and the rows the grid overran into) is rewritten with zp. No copy of the tensor"""
  Wp, P64 = Wp or W + 2, y.shape[0]
  assert L >= Wp + 1 + P64
  g = Tensor.empty(L, C, dtype=dtypes.uint8, device=y.device)
  g[Wp + 1: Wp + 1 + P64].assign(y)
  v = g[: (H + 2) * Wp].reshape(H + 2, Wp, C)
  for sl in ((slice(0, 1), slice(0, Wp)), (slice(H + 1, H + 2), slice(0, Wp)), (slice(1, H + 1), slice(0, 1)), (slice(1, H + 1), slice(W + 1, Wp))):
    v[sl].assign(Tensor.full(v[sl].shape, zp, dtype=dtypes.uint8, device=y.device))
  return g

def window(x:Tensor, Wp, k, s, P64, base, kx=None):
  """(L, C) -> (P64, dy, dx, C): x[base + s*p + dy*Wp + dx], dy < k, dx < kx (default k)"""
  C, kx = x.shape[1], kx or k
  v = x[base:].permute(1, 0)._pool((kx,), 1, 1)                    # (C, L', kx): dx
  v = v.permute(0, 2, 1)._pool((k,), s, Wp)                        # (C, kx, P', k): dy, stride s over the grid
  return v.shrink(((0, C), (0, kx), (0, P64), (0, k))).permute(2, 3, 1, 0)

class Net:
  def __init__(self, model):
    self.tensors, self.ops, _, self.xin, self.yout = qdq_graph.lower(model)
    # rows each activation must hold: the max over its consumers' windows (and its own grid)
    self.L: dict[str, int] = {}
    def need(t, pad, k, s):
      self.L[t.name] = max(self.L.get(t.name, 0), need_rows(t.h, t.w, pad, k, s), (t.h + 2 * pad) * (t.w + 2 * pad))
    # the stem (k x k, stride 2, on the input) runs as a stride-1 (k+1)/2-square conv on the input's 2x2 phase split S: pixel
    # (u, v) of S holds x_pad[2u + by, 2v + bx, c] at channel (2 by + bx) * 8 + c -- 32 bytes, one K block per tap
    st = self.ops[0]
    assert st["op"] == "conv" and st["x"] is self.xin and st["s"] == 2 and self.xin.c <= 8
    self.sk = (st["k"] + 1) // 2
    self.Hs, self.Ws = (self.xin.h + 2 * (st["k"] // 2) + 1) // 2, (self.xin.w + 2 * (st["k"] // 2) + 1) // 2
    self.L["S"] = max(self.Hs * self.Ws, r64(st["y"].h * self.Ws) - 1 + (self.sk - 1) * self.Ws + self.sk)  # window from its top-left
    self.Wg: dict[str, int] = {st["y"].name: self.Ws}  # grids whose row stride isn't W + 2
    for o in self.ops:
      if o["op"] == "conv" and o["x"] is not self.xin: need(o["x"], 1, o["k"], o["s"])
      elif o["op"] == "maxpool":
        Wp = self.Wg.get(o["x"].name)
        self.L[o["x"].name] = max(self.L.get(o["x"].name, 0), need_rows(o["x"].h, o["x"].w, 1, 3, 2, Wp=Wp))
      need(o["y"], 1, 1, 1)
    y0 = st["y"]  # the stem writes its grid straight into its consumer's padded grid (row stride Ws)
    self.L[y0.name] = max(self.L[y0.name], self.Ws + 1 + r64(y0.h * self.Ws), (y0.h + 2) * self.Ws)
    for o in self.ops:  # a stride-1 conv writes its whole output grid into the next padded grid at offset Wp + 1
      if o["op"] == "conv" and o["s"] == 1 and o["x"] is not self.xin:
        y, Wp = o["y"], o["y"].w + 2
        self.L[y.name] = max(self.L[y.name], Wp + 1 + r64(y.h * Wp))
    for o in self.ops:  # an Add runs over its inputs' whole buffers: all three the same length
      if o["op"] == "add":
        n = max(self.L[o[k].name] for k in ("a", "b", "y"))
        n = (n * o["y"].c + 127) // 128 * 128 // o["y"].c if (n * o["y"].c) % 128 else n
        for k in ("a", "b", "y"): self.L[o[k].name] = max(self.L[o[k].name], n)
    for o in self.ops:
      if o["op"] == "add":
        n = max(self.L[o[k].name] for k in ("a", "b", "y"))
        for k in ("a", "b", "y"): self.L[o[k].name] = n

  def build(self, x_nhwc:Tensor, upto:int|None=None) -> Tensor:
    """x_nhwc: (H, W, 3) uint8 at the input's quantization -> the output (C, H, W) uint8 (as ORT's NCHW output)"""
    xin, st = self.xin, self.ops[0]
    self.consts: list[Tensor] = []
    # stem: the 2x2 phase split of the padded input (one copy), then a stride-1 sk x sk conv on it, K = sk*sk blocks of 32
    k0, p0, Hs, Ws, sk = st["k"], st["k"] // 2, self.Hs, self.Ws, self.sk
    x = x_nhwc.pad(((p0, 2 * Hs - xin.h - p0), (p0, 2 * Ws - xin.w - p0), (0, 8 - xin.c)), value=xin.zp)
    S = x.reshape(Hs, 2, Ws, 2, 8).permute(0, 2, 1, 3, 4).reshape(Hs * Ws, 32)
    S = S.pad(((0, self.L["S"] - Hs * Ws), (0, 0)), value=xin.zp).contiguous()
    wq, bq, N = st["wq"], st["bq"].astype(np.int64), st["wq"].shape[0]
    wk = np.zeros((sk, sk, 2, 2, 8, N), np.int8)                    # (ay, ax, by, bx, c, n): tap (2 ay + by, 2 ax + bx)
    for dy in range(k0):
      for dx in range(k0): wk[dy // 2, dx // 2, dy % 2, dx % 2, :xin.c] = wq[:, :, dy, dx].T
    yt = st["y"]
    bias = (bq - int(xin.zp) * wq.reshape(N, -1).astype(np.int64).sum(1)).astype(np.int32)
    m = (f32(xin.scale) * st["swa"].astype(f32) / f32(yt.scale)).astype(f32)
    W_, B_, M_ = Tensor(wk.reshape(sk, sk, 32, N)), Tensor(bias), Tensor(m)
    self.consts += [W_, B_, M_]
    P64 = r64(yt.h * Ws)
    v = window(S, Ws, sk, 1, P64, 0)
    acc = (v.reshape(P64, 1, sk, sk, 32).cast(dtypes.int32) *
           W_.permute(3, 0, 1, 2).reshape(1, N, sk, sk, 32).cast(dtypes.int32)).sum((2, 3, 4)) + B_
    y = ((acc.cast(dtypes.float32) * M_).round() + float(yt.zp)).clip(0, 255).cast(dtypes.uint8)
    if getenv("QDQ_STEM_COPY"): vals = {yt.name: Act(canon(y.contiguous(), yt.h, yt.w, Ws, N, yt.zp, 1, self.L[yt.name]), yt.h, yt.w, N, 1, yt.zp)}
    else: vals = {yt.name: Act(into_grid(y, yt.h, yt.w, N, yt.zp, self.L[yt.name], Wp=Ws), yt.h, yt.w, N, 1, yt.zp, Wp=Ws)}
    for o in self.ops[1:(upto + 1 if upto is not None else None)]:
      if o["op"] == "conv":
        a, yt, k, s = vals[o["x"].name], o["y"], o["k"], o["s"]
        wq, bq = o["wq"], o["bq"].astype(np.int64)
        N, Cw = wq.shape[0], wq.shape[1]
        kx = k
        wk = np.zeros((k, kx, a.C, N), np.int8)
        wk[:, :k, :Cw, :] = wq.transpose(2, 3, 1, 0)                  # (dy, dx, C, N)
        bias = (bq - int(a.zp) * wq.reshape(N, -1).astype(np.int64).sum(1)).astype(np.int32)  # zero point folded in
        m = (f32(o["x"].scale) * o["swa"].astype(f32) / f32(yt.scale)).astype(f32)             # ORT: fp32(fp32(sx sw) / sy)
        W_, B_, M_ = Tensor(wk), Tensor(bias), Tensor(m)
        self.consts += [W_, B_, M_]
        Ho, Wo = (a.H - 1) // s + 1, (a.W - 1) // s + 1
        P64 = r64(Ho * a.Wp)
        v = window(a.t, a.Wp, k, s, P64, (a.pad - k // 2) * (a.Wp + 1), kx)
        acc = (v.reshape(P64, 1, k, kx, a.C).cast(dtypes.int32) *
               W_.permute(3, 0, 1, 2).reshape(1, N, k, kx, a.C).cast(dtypes.int32)).sum((2, 3, 4)) + B_
        y = ((acc.cast(dtypes.float32) * M_).round() + float(yt.zp)).clip(0, 255).cast(dtypes.uint8)
        if s == 1 and a.pad == 1 and a.Wp == a.W + 2 and getenv("QDQ_INTO_GRID", 1):  # same row stride: straight into the next padded grid
          vals[yt.name] = Act(into_grid(y, Ho, Wo, N, yt.zp, self.L[yt.name]), Ho, Wo, N, 1, yt.zp)
        else:
          # materialized on its grid: fused with the crop that follows, tinygrad splits the pixel axis again (no TensorCore)
          vals[yt.name] = Act(canon(y.contiguous(), Ho, Wo, a.Wp, N, yt.zp, 1, self.L[yt.name]), Ho, Wo, N, 1, yt.zp)
      elif o["op"] == "add":
        a, b, yt = vals[o["a"].name], vals[o["b"].name], o["y"]
        ra, rb, fixed = qdq_graph.add_consts(o["a"], o["b"], yt)
        assert (a.H, a.W, a.C, a.pad) == (b.H, b.W, b.C, b.pad) and a.t.shape == b.t.shape
        # the pads hold za, zb: they must come out as zy (they are the next conv's padding)
        pz = qdq_graph.add(np.array([o["a"].zp], np.uint8), o["a"], np.array([o["b"].zp], np.uint8), o["b"], yt)[0]
        assert pz == yt.zp, f"Add pads: {o['a'].zp} + {o['b'].zp} -> {pz}, not zy {yt.zp}"
        vals[yt.name] = Act(hmx_qlinear_add(a.t, b.t, float(ra), float(rb), float(fixed)), a.H, a.W, a.C, 1, yt.zp)
      else:  # maxpool 3x3 s2 p1: the ring holds the input's zero point, 0 (post-Relu): the pooling minimum
        a, yt = vals[o["x"].name], o["y"]
        assert a.zp == 0 and a.pad == 1, "MaxPool needs a zero-point-0 input (its ring is the pooling minimum)"
        Ho, Wo = (a.H - 1) // 2 + 1, (a.W - 1) // 2 + 1
        P64 = r64(Ho * a.Wp)
        y = window(a.t, a.Wp, 3, 2, P64, 0).max(axis=(1, 2)).contiguous()
        vals[yt.name] = Act(canon(y, Ho, Wo, a.Wp, a.C, yt.zp, 1, self.L[yt.name]), Ho, Wo, a.C, 1, yt.zp)
    out = vals[self.yout.name if upto is None else self.ops[upto]["y"].name]
    y = out.t[: (out.H + 2) * out.Wp].reshape(out.H + 2, out.Wp, out.C)[1:out.H + 1, 1:out.W + 1]
    return y.permute(2, 0, 1).contiguous()

# ---------------------------------------------------------------- capture: every kernel of one realize, in order, not run

_CAP: dict = {"calls": [], "bufs": {}}  # mutated in place: pattern functions snapshot their globals (no rebinding)

def _cap_exec(ctx, call, ast):
  import tinygrad.engine.realize as R
  resolved = R.resolve_params(call, ctx.input_uops)
  ids = []
  for i in ast.arg.globals:
    b = resolved[i].buffer  # the Buffer (resolve_params gives UOps)
    if id(b) not in _CAP["bufs"]: _CAP["bufs"][id(b)] = [b, b.nbytes, None]
    ids.append(id(b))
  outs = {ast.arg.globals.index(i) for i in ast.arg.outs}
  src = ast.src[2].arg
  if "__hmx_qadd_chunk(" in src: outs.add(0)  # the custom Add writes its output in the CUSTOM call, not with a STORE
  _CAP["calls"].append((re.sub(r"\x1b\[[0-9;]*m", "", ast.arg.name), src, ids, outs))
  return [0.0]

def capture(out:Tensor, consts:list[Tensor], x:Tensor):
  """every kernel of out's realize, in order, recorded instead of run -> (calls, bufs): calls = [(name, src, [buffer ids],
  output param indices)], bufs = {id: [Buffer, nbytes, "read" if read before any kernel writes it]}"""
  import tinygrad.engine.realize as R
  from tinygrad.uop.ops import PatternMatcher
  Tensor.realize(*consts, x)
  _CAP["calls"].clear(); _CAP["bufs"].clear()
  pats = [(p, _cap_exec if f is R.exec_kernel else f) for p, f in R.pm_exec.patterns]
  old, R.pm_exec = R.pm_exec, PatternMatcher(pats)
  try: out.realize()
  finally: R.pm_exec = old
  calls, bufs = list(_CAP["calls"]), dict(_CAP["bufs"])
  written: set[int] = set()
  for _, _, ids, outs in calls:
    for j, b in enumerate(ids):
      if j in outs: written.add(b)
      elif b not in written and bufs[b][2] is None: bufs[b][2] = "read"
  return calls, bufs

def emit(outdir, calls, bufs, xid:int, yid:int) -> dict:
  """the captured graph as C: k<n>.c per distinct kernel (renamed, boilerplate dropped), graph.h (buffer sizes, the constants'
  offsets in blob.bin, the call sequence g_run), blob.bin. The input buffer is filled per run, the output read back."""
  out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
  order = list(dict.fromkeys(b for _, _, ids, _ in calls for b in ids))
  idx = {b: i for i, b in enumerate(order)}
  blob, off = bytearray(), {}
  for b in order:
    if bufs[b][2] == "read" and b != xid:
      off[b] = len(blob); blob += bytes(bufs[b][0].as_memoryview())
      blob += bytes(-len(blob) % 128)
  knames, lines = {}, []
  for name, src, ids, _ in calls:
    if src not in knames:
      kn = knames[src] = f"k{len(knames)}"
      body = src.split("/* DSP boilerplate */")[0]
      body = re.sub(rf"\bvoid\s+{re.escape(name)}\(", f"void {kn}(", body)
      (out / f"{kn}.c").write_text(body)
    lines.append(f"  {knames[src]}({', '.join(f'B[{idx[b]}]' for b in ids)}); G_PROF({len(lines)});  /* {name} */")
  rnd = lambda n: (n + 127) // 128 * 128 + 128  # + 128: slack for the A packer's unaligned row loads
  h = [f"/* generated by qdq_net.py: {len(calls)} kernel calls, {len(knames)} distinct kernels, {len(order)} buffers */",
       f"#define G_NBUF {len(order)}", f"#define G_INPUT {idx[xid]}", f"#define G_OUTPUT {idx[yid]}", f"#define G_BLOB_BYTES {len(blob)}",
       f"#define G_VTCM_KB {int(os.environ.get('HMX_VTCM_KB', 256))}",
       "static const unsigned G_BYTES[G_NBUF] = {" + ", ".join(str(rnd(bufs[b][1])) for b in order) + "};",
       "static const int G_OFF[G_NBUF] = {" + ", ".join(str(off.get(b, -1)) for b in order) + "};",
       *[f"void {k}();" for k in knames.values()],
       "#ifndef G_PROF\n#define G_PROF(i)\n#endif  /* per-call timing hook: G_PROF(call index) after each call */",
       "static void g_run(unsigned char** B) {", *lines, "}"]
  (out / "graph.h").write_text("\n".join(h) + "\n")
  (out / "blob.bin").write_bytes(bytes(blob))
  return {"kernels": list(knames.values()), "nbuf": len(order), "blob": len(blob), "calls": len(calls),
          "scratch": sum(rnd(bufs[b][1]) for b in order if b not in off)}

SIM_MAIN = r"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern unsigned long long hexagon_sim_read_pcycles(void);
static unsigned long long g_prof[256], g_last;
#define G_PROF(i) (g_prof[i] += hexagon_sim_read_pcycles() - g_last, g_last = hexagon_sim_read_pcycles())
#include "graph.h"
unsigned char* __hmx_vtcm;
unsigned int __hmx_gen = 1;
static void rd(const char* p, void* d, unsigned n) { FILE* f = fopen(p, "rb"); if (!f || fread(d, 1, n, f) != n) { printf("read %s\n", p); exit(1); } fclose(f); }
int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 1;
  unsigned base; __asm__ volatile("%0 = cfgbase" : "=r"(base));
  __hmx_vtcm = (unsigned char*)(*(volatile unsigned*)((base << 16) + 0x38) << 16);  /* VTCM base from the config table */
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" :: "r"(r));
  unsigned char* blob = aligned_alloc(128, G_BLOB_BYTES + 128);
  rd("blob.bin", blob, G_BLOB_BYTES);
  unsigned char* B[G_NBUF];
  for (int i = 0; i < G_NBUF; i++) {
    B[i] = aligned_alloc(128, G_BYTES[i]);
    if (G_OFF[i] >= 0) memcpy(B[i], blob + G_OFF[i], G_BYTES[i]); else memset(B[i], 0, G_BYTES[i]);
  }
  rd("input.bin", B[G_INPUT], INPUT_BYTES);
  unsigned long long t0 = hexagon_sim_read_pcycles();
  for (int i = 0; i < reps; i++) { g_last = hexagon_sim_read_pcycles(); g_run(B); }
  unsigned long long t1 = hexagon_sim_read_pcycles();
  FILE* f = fopen("out.bin", "wb"); fwrite(B[G_OUTPUT], 1, OUTPUT_BYTES, f); fclose(f);
  printf("graph pcycles %llu\n", (t1 - t0) / reps);
  for (int i = 0; i < 256 && g_prof[i]; i++) printf("call %d pcycles %llu\n", i, g_prof[i] / reps);
  return 0;
}
"""

def run_sim(outdir, x_bytes:bytes, y_nbytes:int, reps:int=1) -> tuple[bytes, int]:
  import subprocess
  out = pathlib.Path(outdir)
  tools = pathlib.Path(os.environ["HEXAGON_TOOLS"])
  (out / "input.bin").write_bytes(x_bytes)
  (out / "sim_main.c").write_text(f"#define INPUT_BYTES {len(x_bytes)}\n#define OUTPUT_BYTES {y_nbytes}\n" + SIM_MAIN)
  ks = sorted(p.name for p in out.glob("k*.c"))
  subprocess.run([str(tools / "bin/hexagon-clang"), "-mv69", "-mhmx", "-mhvx", "-mhvx-length=128B", "-O2", "-Wno-deprecated-non-prototype",
                  "sim_main.c", *ks, "-o", "g.elf", "-lhexagon"], cwd=out, check=True)
  from tinygrad.runtime import ops_dsp
  r = subprocess.run([str(tools / "bin/hexagon-sim"), "-mv69", "--mhmx", "1", "--timing", "g.elf", "--", str(reps)], cwd=out,
                     env=ops_dsp._hexsim_env(tools, out), capture_output=True, text=True, check=True)
  (out / "sim_profile.txt").write_text(r.stdout)
  return (out / "out.bin").read_bytes(), int(re.search(r"graph pcycles (\d+)", r.stdout).group(1))

if __name__ == "__main__":
  import onnx
  model = onnx.shape_inference.infer_shapes(onnx.load(sys.argv[1]))
  outdir = pathlib.Path(sys.argv[2])
  net = Net(model)
  x = Tensor.empty(net.xin.h, net.xin.w, net.xin.c, dtype=dtypes.uint8)
  upto = int(sys.argv[sys.argv.index("--dump") + 1]) if "--dump" in sys.argv else None  # stop after op N (debugging)
  y = net.build(x, upto)
  calls, bufs = capture(y, net.consts, x)
  kinds = {}
  for name, src, ids, outs in calls:
    kind = "hmx" if "__hmx_i8_mac" in src else "qadd" if "__hmx_qadd_chunk" in src else "other"
    kinds[kind] = kinds.get(kind, 0) + 1
  info = emit(outdir, calls, bufs, id(x.uop.buffer), id(y.uop.buffer))
  print(f"{len(calls)} kernel calls {kinds} ({len(info['kernels'])} distinct), {info['nbuf']} buffers: constants "
        f"{info['blob'] / 1e6:.2f} MB, scratch {info['scratch'] / 1e6:.2f} MB -> {outdir}")
  if "--sim" in sys.argv:  # the whole graph on hexagon-sim vs qdq_graph's exact emulator
    xin_path = sys.argv[sys.argv.index("--sim") + 1]
    xq = np.fromfile(xin_path, np.uint8).reshape(1, net.xin.h, net.xin.w, net.xin.c)
    ref = qdq_graph.emulate(net.tensors, net.ops, net.xin, xq)[net.yout.name if upto is None else net.ops[upto]["y"].name]
    got, cyc = run_sim(outdir, xq.tobytes(), ref.nbytes)
    got = np.frombuffer(got, np.uint8).reshape(ref.shape)
    bad = int((got != ref).sum())
    if bad and os.environ.get("QDQ_DIFF"):
      c, h, w = np.nonzero(got != ref)
      print("  mismatch positions (h, w):", sorted(set(zip(h.tolist(), w.tolist())))[:20], "channels", sorted(set(c.tolist()))[:10])
      d = got.astype(int) - ref
      print("  diffs:", {int(k): int(v) for k, v in zip(*np.unique(d[d != 0], return_counts=True))})
    print(f"hexagon-sim: {bad}/{ref.size} mismatches vs the exact emulator; {cyc} pcycles/inference "
          f"({'PASS' if bad == 0 else 'FAIL'})")
    if upto is None and len(sys.argv) > sys.argv.index("--sim") + 2 and not sys.argv[sys.argv.index("--sim") + 2].startswith("--"):
      ort = np.fromfile(sys.argv[sys.argv.index("--sim") + 2], np.uint8)
      print(f"  vs ORT CPU's output: {int((got.ravel() != ort).sum())}/{ort.size} mismatches")
