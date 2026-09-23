"""Find where the HTP run of a decoder graph departs from host ORT: expose chosen intermediates.

  python bisect_llm.py make   --work W --graph dec_prefill.fp16.onnx [--prompt 0] [--pick SUBSTR ...]
      -> W/bisect/<graph>.dbg.onnx with the picked float tensors as extra outputs (default: every
         residual-stream Add, i.e. the output of each attention and MLP block), the phone manifest
         and host-ORT references of every exposed tensor
  ./bisect_llm.sh W <graph>   (phone, under the lock; qnn_run_multi writes out_htp_o<i>.bin)
  python bisect_llm.py report --work W --graph dec_prefill.fp16.onnx
      -> per-tensor cosine / max abs error vs host ORT, in graph order, first bad one flagged

Extra outputs can change what the HTP compiler fuses, so expose a few at a time.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import models
import numpy as np
import onnx

NP = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16}


def feeds(work: Path, graph: str, prompt: int):
    pr = np.fromfile(work / "dec_in" / f"prompt_{prompt}.bin", np.int32)
    if "prefill" in graph:
        ids = np.zeros((1, models.PREFILL), np.int32)
        ids[0, : len(pr)] = pr
        return {"input_ids": ids, "last_idx": np.array([len(pr) - 1], np.int32)}
    raise SystemExit(
        "decode-step bisect: pass a prefill graph (step inputs come from its KV output)"
    )


def make(a):
    work = Path(a.work)
    d = work / "bisect"
    d.mkdir(exist_ok=True)
    m = onnx.shape_inference.infer_shapes(onnx.load(work / a.graph))
    vi = {v.name: v for v in m.graph.value_info}
    have = {o.name for o in m.graph.output}
    picked = []
    done = False
    for n in m.graph.node:
        o = n.output[0]
        v = vi.get(o)
        if v is None or o in have or v.type.tensor_type.elem_type not in NP:
            continue
        if a.upto:
            ok = not done
            if o == a.upto:
                done = True
        elif a.pick:
            ok = any(p in o or p in n.name for p in a.pick)
        else:
            dims = [x.dim_value for x in v.type.tensor_type.shape.dim]
            ok = n.op_type == "Add" and dims[-1:] == [576]
        if ok:
            m.graph.output.append(v)
            have.add(o)
            picked.append(o)
    out = d / a.graph.replace(".onnx", ".dbg.onnx")
    onnx.save(m, out)
    f = feeds(work, a.graph, a.prompt)
    man = []
    for k, v in f.items():
        v.tofile(d / f"in_{k}.bin")
        man.append(f"{k} i32 {d / f'in_{k}.bin'} {','.join(map(str, v.shape))}")
    (d / "manifest.txt").write_text("\n".join(man) + "\n")
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    for i, v in enumerate(s.run(None, f)):
        np.asarray(v).astype(np.float32).tofile(d / f"ref_o{i}.bin")
    (d / "names.txt").write_text("\n".join(names) + "\n")
    print(f"{out.name}: {len(names)} outputs ({len(picked)} exposed)")


def report(a):
    d = Path(a.work) / "bisect"
    names = (d / "names.txt").read_text().split()
    m = onnx.load(d / a.graph.replace(".onnx", ".dbg.onnx"), load_external_data=False)
    et = {o.name: o.type.tensor_type.elem_type for o in m.graph.output}
    first = None
    for i, n in enumerate(names):
        ref = np.fromfile(d / f"ref_o{i}.bin", np.float32).astype(np.float64)
        # qnn_run_multi writes fp16 outputs with a 4-byte element size: the first half is the tensor
        got = np.fromfile(d / f"out_htp_o{i}.bin", NP[et[n]])[: ref.size].astype(
            np.float64
        )
        if ref.size != got.size:
            print(f"{i:3d} {n}: size mismatch {ref.size} vs {got.size}")
            continue
        c = ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-30)
        # the residual stream's outlier channels dominate a plain cosine: also compare the rest
        small = np.abs(ref) < 0.05 * np.abs(ref).max()
        cs = (
            ref[small]
            @ got[small]
            / (np.linalg.norm(ref[small]) * np.linalg.norm(got[small]) + 1e-30)
            if small.any()
            else 1.0
        )
        c = min(c, cs)
        err = np.abs(ref - got).max()
        flag = ""
        if first is None and (c < 0.999 or not np.isfinite(got).all()):
            first, flag = n, "  <-- first bad"
        short = re.sub(r"^/m/", "", n)
        print(
            f"{i:3d} {short:60s} cos {c:.6f} maxerr {err:.3g} |ref|max {np.abs(ref).max():.3g} finite {np.isfinite(got).all()}{flag}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "report"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--pick", nargs="*")
    ap.add_argument(
        "--upto",
        help="expose every float tensor produced before (and including) this one",
    )
    a = ap.parse_args()
    make(a) if a.cmd == "make" else report(a)
