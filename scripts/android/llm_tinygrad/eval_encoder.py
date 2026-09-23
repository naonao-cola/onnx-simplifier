"""Encoder accuracy: embedding cosine vs the fp32 torch reference.

python eval_encoder.py --work W --phone OUTDIR     # emb_<i>.bin pulled from the phone
python eval_encoder.py --work W --host enc.a8.onnx # same model on host ORT (basic opts)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--phone")
    ap.add_argument("--host")
    a = ap.parse_args()
    work = Path(a.work)
    ref = np.load(work / "enc_ref.npy")
    n = len(ref)
    if a.phone:
        got = np.stack(
            [np.fromfile(Path(a.phone) / f"emb_{i}.bin", np.float32) for i in range(n)]
        )
    else:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        s = ort.InferenceSession(
            str(work / a.host), so, providers=["CPUExecutionProvider"]
        )
        got = np.concatenate(
            [
                s.run(
                    None,
                    {
                        "input_ids": np.fromfile(
                            work / "enc_in" / f"ids_{i}.bin", np.int32
                        ).reshape(1, -1),
                        "attention_mask": np.fromfile(
                            work / "enc_in" / f"mask_{i}.bin", np.int32
                        ).reshape(1, -1),
                    },
                )[0]
                for i in range(n)
            ]
        )
    got = got / np.linalg.norm(got, axis=-1, keepdims=True)
    cos = (got * ref).sum(-1)
    # retrieval sanity: does each sentence's nearest neighbour among the references stay itself?
    top1 = float(np.mean(np.argmax(got @ ref.T, -1) == np.arange(n)))
    print(
        f"cos mean {cos.mean():.6f} min {cos.min():.6f}  self-retrieval top1 {top1:.2f}"
    )


if __name__ == "__main__":
    main()
