"""Decoder accuracy vs the fp32 torch greedy reference (export_smollm.py's dec_ref/).

  python eval_decoder.py --work W --phone OUTDIR [--forced]   # gen_/logits_<i>.bin from llm_run
  python eval_decoder.py --work W --host PREFILL STEP [--forced] # the same loop on host ORT

Free-running (no --forced): how many leading tokens equal the fp32 greedy continuation.
Teacher-forced (--forced, the fp32 tokens fed back): per-step top-1 agreement with fp32, and the
cosine of the first logits vectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import models
import numpy as np


def host_run(work: Path, pre: str, step: str, forced: bool, ngen: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    P = ort.InferenceSession(str(work / pre), so, providers=["CPUExecutionProvider"])
    S = ort.InferenceSession(str(work / step), so, providers=["CPUExecutionProvider"])
    kshape = S.get_inputs()[2].shape
    cdt = np.float16 if "float16" in S.get_inputs()[2].type else np.float32
    out = []
    for i in range(len(models.PROMPTS)):
        pr = np.fromfile(work / "dec_in" / f"prompt_{i}.bin", np.int32)
        fo = np.fromfile(work / "dec_in" / f"force_{i}.bin", np.int32)
        n = len(pr)
        ids = np.zeros((1, models.PREFILL), np.int32)
        ids[0, :n] = pr
        lg, k, v = P.run(
            None, {"input_ids": ids, "last_idx": np.array([n - 1], np.int32)}
        )
        kc = np.zeros(kshape, cdt)
        vc = np.zeros_like(kc)
        kc[:, :, :n], vc[:, :, :n] = k[:, :, :n], v[:, :, :n]
        gen, logits = [int(lg.argmax())], [lg[0].astype(np.float32)]
        for s in range(1, ngen):
            tok = fo[s - 1] if forced else gen[-1]
            pos = n + s - 1
            lg, kn, vn = S.run(
                None,
                {
                    "input_ids": np.array([[tok]], np.int32),
                    "pos": np.array([pos], np.int32),
                    "k_cache": kc,
                    "v_cache": vc,
                },
            )
            kc[:, :, pos], vc[:, :, pos] = kn[:, :, 0], vn[:, :, 0]
            gen.append(int(lg.argmax()))
            if len(logits) < 9:
                logits.append(lg[0].astype(np.float32))
        out.append((np.array(gen), np.stack(logits)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--phone")
    ap.add_argument("--host", nargs=2)
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--logits-dtype", choices=["f32", "f16"], default="f32")
    a = ap.parse_args()
    work = Path(a.work)
    refs = [
        (
            np.load(work / "dec_ref" / f"gen_{i}.npy"),
            np.load(work / "dec_ref" / f"logits_{i}.npy"),
        )
        for i in range(len(models.PROMPTS))
    ]
    ngen, V = len(refs[0][0]), refs[0][1].shape[1]
    if a.host:
        got = host_run(work, a.host[0], a.host[1], a.forced, ngen)
    else:
        d = Path(a.phone)
        got = []
        for i in range(len(models.PROMPTS)):
            gen = np.fromfile(d / f"gen_{i}.bin", np.int32)
            dt = np.float32 if a.logits_dtype == "f32" else np.float16
            lg = (
                np.fromfile(d / f"logits_{i}.bin", dt).astype(np.float32).reshape(-1, V)
            )
            got.append((gen, lg))
    prefix, agree, coss = [], [], []
    for (rg, rl), (g, lg) in zip(refs, got):
        m = min(len(rg), len(g))
        same = rg[:m] == g[:m]
        prefix.append(int(np.argmin(same)) if not same.all() else m)
        agree.append(float(same.mean()))
        k = min(len(rl), len(lg))
        c = (rl[:k] * lg[:k]).sum(-1) / (
            np.linalg.norm(rl[:k], axis=-1) * np.linalg.norm(lg[:k], axis=-1)
        )
        coss.append(c.min())
    if a.forced:
        print(
            f"forced: top-1 agreement {np.mean(agree):.4f} (min {np.min(agree):.3f}); logits cos min {np.min(coss):.6f} mean {np.mean(coss):.6f}"
        )
    else:
        print(
            f"free: identical leading tokens per prompt {prefix} (of {ngen}); all {ngen} identical in {sum(p == ngen for p in prefix)}/{len(prefix)} prompts"
        )


if __name__ == "__main__":
    main()
