"""Pure-numpy side of the qfloat checks: fixed-seed inputs, fp32/float64/ORT references, error metrics."""
import numpy as np

def op_inputs(op:str, n:int):
  rng = np.random.default_rng(0)
  if op == "blend":
    taps = [rng.standard_normal((n, 256)).astype(np.float32) for _ in range(4)]
    w = rng.random((n, 4)).astype(np.float32)
    w /= w.sum(1, keepdims=True)   # bilinear weights sum to 1
    return taps + [w]
  if op == "sigmoid": return [rng.normal(0, 3, n).astype(np.float32)]
  if op == "poly": return [rng.uniform(-1, 1, n).astype(np.float32)]
  if op == "diffprod": return [rng.uniform(0, 1000, n).astype(np.float32) for _ in range(4)]
  raise ValueError(op)

POLY = [0.0013333558, 0.0096181291, 0.0555041087, 0.2402265070, 0.6931471806, 1.0]  # 2**x Taylor-ish, Horner form

def reference(op:str, ins, dtype):
  x = [i.astype(dtype) for i in ins]
  if op == "blend":
    a, b, c, d, w = x
    return a*w[:, 0:1] + b*w[:, 1:2] + c*w[:, 2:3] + d*w[:, 3:4]
  if op == "diffprod":
    a, b, c, d = x; return ((a - b) * (c - d)) * ((a + b) * (c + d))
  if op == "poly":
    y = x[0] * dtype(POLY[0])
    for c in POLY[1:-1]: y = (y + dtype(c)) * x[0]
    return y + dtype(POLY[-1])
  return 1 / (1 + np.exp(-x[0]))

def ort_reference(op:str, ins):
  if op != "sigmoid": return None
  try:
    import onnx, onnxruntime as ort
    from onnx import helper, TensorProto
    g = helper.make_graph([helper.make_node("Sigmoid", ["x"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [ins[0].size])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [ins[0].size])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": ins[0]})[0]
  except Exception: return None

def errors(got, ref):
  got, ref = got.astype(np.float64), ref.astype(np.float64)
  ae = np.abs(got - ref)
  re_ = ae / np.maximum(np.abs(ref), 1e-30)
  return ae.max(), re_.max(), float(np.mean(got == ref))
