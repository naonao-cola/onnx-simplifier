"""MiniLM-L6 (all-MiniLM-L6-v2) sentence encoder -> static ONNX, fp32 reference embeddings, phone inputs.

  python export_minilm.py --work W

Writes W/enc.fp32.onnx (input_ids/attention_mask int32 [1,128] -> embedding [1,384], mean pooling +
L2 normalization inside the graph, like sentence-transformers), W/enc_in/{ids,mask}_<i>.bin for the
20 evaluation sentences, and W/enc_ref.npy (fp32 torch embeddings).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import models
import numpy as np
import onnx
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):
        m = attention_mask.to(torch.float32)
        h = self.bert(input_ids=input_ids.long(), attention_mask=m).last_hidden_state
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1e-9)
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def bounded_mask(model: onnx.ModelProto) -> onnx.ModelProto:
    """HF's additive attention mask is (1 - mask) * finfo(fp32).min = -3.4e38: -inf in fp16 and a
    range no quantizer can share with real scores. -1e4 masks the same (exp(-1e4) == 0)."""
    from onnx import numpy_helper

    for n in model.graph.node:
        if n.op_type == "Constant":
            a = numpy_helper.to_array(n.attribute[0].t)
            if (
                a.dtype.kind == "f"
                and a.size == 1
                and abs(float(a.reshape(-1)[0])) > 1e30
            ):
                n.attribute[0].t.CopyFrom(
                    numpy_helper.from_array(
                        np.full_like(a, -1e4), n.attribute[0].t.name
                    )
                )
    return model


def int32_ids(model: onnx.ModelProto) -> onnx.ModelProto:
    """Feed the int32 graph inputs to Gather directly: the exporter adds Cast(int64), which the
    HTP doesn't need (ONNX Gather takes int32 indices)."""
    g = model.graph
    for n in list(g.node):
        if n.op_type == "Cast" and n.input[0] == "input_ids":
            for m in g.node:
                for i, x in enumerate(m.input):
                    if x == n.output[0]:
                        m.input[i] = "input_ids"
            g.node.remove(n)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    (work / "enc_in").mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    snap = models.fetch(models.MINILM)
    tok = AutoTokenizer.from_pretrained(snap)
    bert = AutoModel.from_pretrained(snap, attn_implementation="eager").eval()
    enc = Encoder(bert).eval()

    t = tok(
        models.SENTENCES,
        padding="max_length",
        max_length=models.ENC_SEQ,
        truncation=True,
        return_tensors="pt",
    )
    ids, mask = t["input_ids"].to(torch.int32), t["attention_mask"].to(torch.int32)
    with torch.no_grad():
        ref = torch.cat(
            [enc(ids[i : i + 1], mask[i : i + 1]) for i in range(len(ids))]
        ).numpy()
    np.save(work / "enc_ref.npy", ref)
    for i in range(len(ids)):
        ids[i : i + 1].numpy().tofile(work / "enc_in" / f"ids_{i}.bin")
        mask[i : i + 1].numpy().tofile(work / "enc_in" / f"mask_{i}.bin")

    out = work / "enc.fp32.onnx"
    torch.onnx.export(
        enc,
        (ids[:1], mask[:1]),
        out,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,
    )
    m = bounded_mask(int32_ids(onnx.load(out)))
    import onnxsim

    m, ok = onnxsim.simplify(m)
    assert ok
    onnx.save(m, out)

    import onnxruntime as ort

    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = np.concatenate(
        [
            s.run(
                None,
                {
                    "input_ids": ids[i : i + 1].numpy(),
                    "attention_mask": mask[i : i + 1].numpy(),
                },
            )[0]
            for i in range(len(ids))
        ]
    )
    cos = (got * ref).sum(-1)  # both unit-norm
    print(f"enc.fp32.onnx vs torch: min cos {cos.min():.7f}, nodes {len(m.graph.node)}")


if __name__ == "__main__":
    main()
