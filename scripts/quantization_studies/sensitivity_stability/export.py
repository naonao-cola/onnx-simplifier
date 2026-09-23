"""Export every pinned model to static fp32 ONNX (opset 17): python export.py [name ...]"""

import sys

import torch
from common import MODELS, WORK, onnx_path, torch_model

names = sys.argv[1:] or list(MODELS)
(WORK / "models").mkdir(parents=True, exist_ok=True)
for name in names:
    p = onnx_path(name)
    if p.exists():
        print("have", p)
        continue
    m = torch_model(name)
    torch.onnx.export(
        m,
        torch.zeros(1, 3, 224, 224),
        str(p),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    print("exported", name, p)
