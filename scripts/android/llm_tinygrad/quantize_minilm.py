"""Quantized MiniLM-L6 variants.

  python quantize_minilm.py --work W

- enc.int8dyn.onnx : ORT dynamic int8 (MatMul/Gemm weights), the usual CPU int8 baseline
- enc.a8.onnx      : onnxsim whole-graph QDQ (uint8 activations, int8 per-channel weights) for the
                     HTP; LayerNorm, Softmax and the GELU/pooling tail stay fp16
- enc.a16.onnx     : the same with uint16 activations (W8A16)
- enc.lin8.onnx    : only Gemm/MatMul quantized (uint8), everything else fp16

Calibration uses 32 sentences disjoint from the 20 evaluation sentences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import models
import numpy as np
import onnx

CALIB = [
    "The weather forecast predicts sunshine for the weekend.",
    "Scientists discovered a new species of frog in the rainforest.",
    "The concert was cancelled due to a power outage.",
    "Please remember to bring your passport to the airport.",
    "The software update fixed several security vulnerabilities.",
    "A group of volunteers cleaned up the river bank on Saturday.",
    "The price of coffee has risen steadily over the past year.",
    "My grandmother taught me how to bake bread when I was young.",
    "The telescope captured detailed images of a distant galaxy.",
    "Commuters were delayed by a signal failure on the subway.",
    "The novel explores themes of memory, loss and identity.",
    "Farmers are worried about the lack of rain this summer.",
    "The city council approved funding for a new bike lane.",
    "He fixed the leaking faucet with a wrench and some tape.",
    "Online shopping has changed how people buy groceries.",
    "The orchestra rehearsed the symphony for three hours.",
    "Doctors recommend drinking plenty of water every day.",
    "The startup raised twenty million dollars in its latest round.",
    "Snow covered the mountains and closed several ski roads.",
    "The detective examined the fingerprints on the glass.",
    "Students protested against the increase in tuition fees.",
    "A cat was sleeping on the warm windowsill.",
    "The airline introduced a new route between Paris and Seoul.",
    "Solar panels were installed on the roof of the school.",
    "The documentary follows a family of elephants across the savanna.",
    "Her presentation on climate policy impressed the audience.",
    "The bakery sells out of croissants every morning by nine.",
    "Engineers tested the bridge for earthquake resistance.",
    "The toddler laughed as the puppy chased its own tail.",
    "Voters lined up early outside the polling station.",
    "The recipe calls for two cups of flour and three eggs.",
    "A strong earthquake shook the coastal town overnight.",
]
FLOAT_OPS = ["LayerNormalization", "Softmax", "Erf", "ReduceSum", "ReduceL2", "Clip"]


def calib_data(snap):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    t = tok(
        CALIB,
        padding="max_length",
        max_length=models.ENC_SEQ,
        truncation=True,
        return_tensors="np",
    )
    return [
        {
            "input_ids": t["input_ids"][i : i + 1].astype(np.int32),
            "attention_mask": t["attention_mask"][i : i + 1].astype(np.int32),
        }
        for i in range(len(CALIB))
    ]


def mask_nodes(m: onnx.ModelProto):
    """The attention-mask path (Cast/Unsqueeze/Sub/Mul -> the additive -1e4 mask) and the score Adds
    that consume it: a -1e4 range would crush the scores' quantization step, so they stay fp16."""
    cons = {}
    for n in m.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    names, todo = set(), ["attention_mask"]
    while todo:
        for n in cons.get(todo.pop(), []):
            if n.name in names:
                continue
            names.add(n.name)
            if n.op_type != "Add":
                todo.append(n.output[0])
    return names


def gelu_and_tail_nodes(m: onnx.ModelProto):
    """The GELU (x * 0.5 * (1 + erf(x / sqrt2))) Div/Add/Mul around each Erf, and the pooling tail
    after the last LayerNorm: keep them float with the rest of FLOAT_OPS."""
    prod = {o: n for n in m.graph.node for o in n.output}
    cons = {}
    for n in m.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    names = set()
    for n in m.graph.node:
        if n.op_type == "Erf":
            d = prod[n.input[0]]
            names.add(d.name)
            a = cons[n.output[0]][0]
            names.add(a.name)
            for c in cons[a.output[0]]:
                names.add(c.name)
                for c2 in cons.get(c.output[0], []):
                    if c2.op_type == "Mul":
                        names.add(c2.name)
    last_ln = [n for n in m.graph.node if n.op_type == "LayerNormalization"][-1]
    after = False
    for n in m.graph.node:
        if after:
            names.add(n.name)
        if n is last_ln:
            after = True
    return sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="int8dyn,a8,a16,lin8")
    a = ap.parse_args()
    work = Path(a.work)
    src = work / "enc.fp32.onnx"
    snap = models.fetch(models.MINILM)
    data = calib_data(snap)
    only = a.only.split(",")
    if "int8dyn" in only:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(src), str(work / "enc.int8dyn.onnx"), weight_type=QuantType.QInt8
        )
    from onnxsim.full_qdq import quantize_full_qdq

    m = onnx.load(src)
    keep = sorted(set(gelu_and_tail_nodes(m)) | mask_nodes(m))
    for tag, kw in [
        ("a8", dict(exclude_op_types=FLOAT_OPS, exclude_nodes=keep)),
        (
            "a16",
            dict(
                exclude_op_types=FLOAT_OPS,
                exclude_nodes=keep,
                activation_dtype="uint16",
            ),
        ),
        ("lin8", dict(op_types=["Gemm", "MatMul"])),
    ]:
        if tag not in only:
            continue
        q = quantize_full_qdq(m, data, method="minmax", **kw)
        onnx.save(q, work / f"enc.{tag}.onnx")
        print(f"enc.{tag}.onnx: {len(q.graph.node)} nodes")


if __name__ == "__main__":
    main()
