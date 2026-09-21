"""Shim `onnx.mapping` (removed in onnx>=1.16) for TVM 0.17's Relax ONNX frontend."""
import sys
import types

import numpy as np
import onnx
from onnx import TensorProto, helper

if not hasattr(onnx, "mapping"):
    m = types.ModuleType("onnx.mapping")
    tt2np = {}
    for k in TensorProto.DataType.values():
        try:
            tt2np[k] = np.dtype(helper.tensor_dtype_to_np_dtype(k))
        except Exception:
            pass
    m.TENSOR_TYPE_TO_NP_TYPE = tt2np
    m.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in tt2np.items()}
    sys.modules["onnx.mapping"] = m
    onnx.mapping = m
