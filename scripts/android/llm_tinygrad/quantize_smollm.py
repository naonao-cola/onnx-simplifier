"""SmolLM2-135M variants of the prefill/step graphs.

  python quantize_smollm.py --work W [--only int8dyn,w8a16]

(dec_*.fp16.onnx, fp16 weights/activations/graph I/O, comes from export_smollm.py --fp16-only.)

- dec_*.int8dyn.onnx : ORT dynamic int8 (MatMul/Gemm weights), the CPU int8 baseline
- dec_*.w8a16.onnx   : onnxsim whole-graph QDQ, int8 per-channel weights, uint16 activations
                       (RMSNorm/Softmax/SiLU and the rotary/attention glue stay fp16), fp16 graph I/O
"""

from __future__ import annotations

import argparse
from pathlib import Path

import models
import numpy as np
import onnx

GRAPHS = ("dec_prefill", "dec_step")


def calib(work: Path, n: int = 8):
    """Prefill inputs from 8 calibration prompts + decode inputs taken from real fp32 runs."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    texts = [
        "The weather today is",
        "My favourite recipe for dinner is",
        "In the year 2050, cities will",
        "The most important invention in history was",
        "A good friend is someone who",
        "import numpy as np\n",
        "The river flowed quietly through the",
        "Scientists have recently discovered that",
    ][:n]
    tok = AutoTokenizer.from_pretrained(models.fetch(models.SMOLLM))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    P = ort.InferenceSession(
        str(work / "dec_prefill.fp32.onnx"), so, providers=["CPUExecutionProvider"]
    )
    S = ort.InferenceSession(
        str(work / "dec_step.fp32.onnx"), so, providers=["CPUExecutionProvider"]
    )
    kshape = S.get_inputs()[2].shape
    pre, step = [], []
    for t in texts:
        ids_l = tok(t).input_ids
        n_ = len(ids_l)
        ids = np.zeros((1, models.PREFILL), np.int32)
        ids[0, :n_] = ids_l
        feed = {"input_ids": ids, "last_idx": np.array([n_ - 1], np.int32)}
        pre.append(feed)
        lg, k, v = P.run(None, feed)
        kc = np.zeros(kshape, np.float32)
        vc = np.zeros_like(kc)
        kc[:, :, :n_], vc[:, :, :n_] = k[:, :, :n_], v[:, :, :n_]
        tok_id = int(lg.argmax())
        for s in range(8):
            pos = n_ + s
            f = {
                "input_ids": np.array([[tok_id]], np.int32),
                "pos": np.array([pos], np.int32),
                "k_cache": kc.copy(),
                "v_cache": vc.copy(),
            }
            step.append(f)
            lg, kn, vn = S.run(None, f)
            kc[:, :, pos], vc[:, :, pos] = kn[:, :, 0], vn[:, :, 0]
            tok_id = int(lg.argmax())
    return {"dec_prefill": pre, "dec_step": step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="int8dyn,w8a16")
    a = ap.parse_args()
    work = Path(a.work)
    only = a.only.split(",")
    for g in GRAPHS:
        src = work / f"{g}.fp32.onnx"
        if "int8dyn" in only:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                str(src),
                str(work / f"{g}.int8dyn.onnx"),
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
    if "w8a16" in only:
        from onnxsim.full_qdq import quantize_full_qdq

        data = calib(work)
        for g in GRAPHS:
            m = onnx.load(work / f"{g}.fp32.onnx")
            q = quantize_full_qdq(
                m,
                data[g],
                activation_dtype="uint16",
                op_types=["MatMul", "Gemm"],
                method="minmax",
            )
            onnx.save(q, work / f"{g}.w8a16.fp32io.onnx")
            print(f"{g}.w8a16: {len(q.graph.node)} nodes")


if __name__ == "__main__":
    main()
