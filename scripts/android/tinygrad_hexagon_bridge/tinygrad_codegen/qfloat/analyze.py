"""Compare the phone's <tag>_out.bin (build.sh) against numpy fp32, float64 and (sigmoid) ORT fp32.
Tags are <op>_<variant> or sigmoid_<n>; inputs are regenerated with qfsim.op_inputs' fixed seeds."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qf_ops as Q  # noqa: E402

SIZES = {"blend": 4096, "diffprod": 1048576}
data = pathlib.Path(sys.argv[1])
print(f"{'kernel':16s} {'vs fp32 max_abs':>16s} {'max_rel':>10s} {'vs f64 max_abs':>15s} {'max_rel':>10s} {'ORT max_rel':>12s}")
for f in sorted(data.glob("*_out.bin")):
  tag = f.name[:-8]
  op = tag.split("_")[0] if not tag.startswith("hand_") else tag.split("_")[1]
  n = int(tag.split("_")[1]) if op == "sigmoid" else SIZES[op]
  ins = Q.op_inputs(op, n)
  r32, r64 = Q.reference(op, ins, np.float32), Q.reference(op, ins, np.float64)
  got = np.fromfile(f, dtype=np.float32).reshape(r32.shape)
  a32, rel32, _ = Q.errors(got, r32); a64, rel64, _ = Q.errors(got, r64)
  o = Q.ort_reference(op, ins); ort = f"{Q.errors(got, o.reshape(r32.shape))[1]:12.3g}" if o is not None else f"{'-':>12s}"
  print(f"{tag:16s} {a32:16.3g} {rel32:10.3g} {a64:15.3g} {rel64:10.3g} {ort}")
