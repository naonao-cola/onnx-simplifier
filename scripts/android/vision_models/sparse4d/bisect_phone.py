"""Where does the HTP diverge from fp32? Expose intermediates of a frame graph, run frame 0 on the
phone (strict HTP) and on ORT CPU, and print each tensor's cosine / max abs.

  PHONE_LOCK_OWNER=codex/android-sparse4d ~/.cache/android-phone/phone-run \\
    python bisect_phone.py --work <work> [--model frame_first.sim.onnx]

Probes, in graph order: the 4 FPN levels (GridSample value inputs), every DFA layer's first
sampling grid and first GridSample output, and every node output named in --extra.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import phone_chain as pc
from export import normalized_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="frame_first.sim.onnx")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    work = Path(a.work)
    m = onnx.load(str(work / a.model))
    gs = [n for n in m.graph.node if n.op_type == "GridSample"]
    probes = []
    for n in gs[:4]:
        probes.append(n.input[0])  # FPN levels
    for i in range(0, len(gs), 4):  # per DFA layer: the grid and the first sample
        probes += [gs[i].input[1], gs[i].output[0]]
    probes += a.extra
    have = {o.name for o in m.graph.output}
    for p in probes:
        if p not in have:
            m.graph.output.append(onnx.ValueInfoProto(name=p))
    out_names = [o.name for o in m.graph.output]
    name = a.model.replace(".onnx", ".bisect.onnx")
    onnx.save(m, str(work / name), save_as_external_data=False)
    fr = pickle.load(open(work / "frames" / "0.pkl", "rb"))
    feeds = {"rgb": fr["rgb"], "proj": fr["metas"]["projection_mat"].numpy(),
             "proj_n": normalized_proj(fr["metas"]["projection_mat"]).numpy()}
    feeds = {k: v for k, v in feeds.items() if k in {i.name for i in m.graph.input}}
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, feeds)
    pc.TMP.mkdir(parents=True, exist_ok=True)
    pc.setup(work, [name])
    import re
    import subprocess

    ins = [(k, "u8" if v.dtype == np.uint8 else "f32", v) for k, v in feeds.items()]
    # run() pulls 4 outputs; pull them all here
    lines = []
    for k, dt, v in ins:
        fn = f"b_{k}.bin"
        p = pc.TMP / fn
        np.ascontiguousarray(v).tofile(p)
        pc.adb("push", "-q", str(p), f"{pc.R}/{fn}")
        lines.append(f"{k} {dt} {fn} {','.join(map(str, v.shape))}")
    man = pc.TMP / "b_manifest.txt"
    man.write_text("\n".join(lines) + "\n")
    pc.adb("push", "-q", str(man), f"{pc.R}/b_manifest.txt")
    out = pc.adb("shell", f"cd {pc.R} && ORT_SPIN=0 QNN_PERF=burst LD_LIBRARY_PATH={pc.R} "
                 f"ADSP_LIBRARY_PATH='{pc.R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
                 f"./qnn_run_multi {name} b_manifest.txt htp 1 b_out 2>&1", capture=True)
    if "PASS" not in out:
        raise SystemExit(out[-3000:])
    shapes = {int(mm.group(1)): [int(x) for x in mm.group(4).split(",") if x]
              for mm in re.finditer(r"out (\d+) (\S+) (\S+) ([\d,]+)", out)}
    for i, nm in enumerate(out_names):
        loc = pc.TMP / f"b_o{i}.bin"
        subprocess.run(["adb", "-s", pc.SERIAL, "pull", "-q", f"{pc.R}/b_out_o{i}.bin", str(loc)], check=True)
        ph = np.fromfile(loc, dtype=np.float32).reshape(shapes[i])
        loc.unlink()
        c = cpu[i]
        print(f"{nm[:48]:48s} cos {pc.cos(ph, c):.6f}  max|cpu| {np.abs(c).max():9.3g}  max|diff| {np.abs(ph - c).max():9.3g}")


if __name__ == "__main__":
    main()
