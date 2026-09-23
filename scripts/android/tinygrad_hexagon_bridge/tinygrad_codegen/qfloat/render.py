import sys
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.runtime.ops_dsp import MockDSPRenderer
class _NoCompile:
  def compile_cached(self, src): return b""
def render(t:Tensor) -> list[str]:
  out = []
  for k in t.schedule_linear().src:
    ast = k.src[0]
    if ast.op.name != "SINK": continue
    ren = MockDSPRenderer(Target(device="DSP")); ren.compiler = _NoCompile()
    out.append(to_program(ast, ren).src[2].arg.split("/* DSP boilerplate */")[0])
  return out
def blend(M, C=256):
  a,b,c,d = (Tensor.empty(M, C) for _ in range(4)); w = Tensor.empty(M, 4)
  return a*w[:,0:1] + b*w[:,1:2] + c*w[:,2:3] + d*w[:,3:4]
def sigmoid(n): return Tensor.empty(n).sigmoid()
if __name__ == "__main__":
  t = blend(4096) if sys.argv[1] == "blend" else sigmoid(int(sys.argv[2]))
  for s in render(t): print(s)
