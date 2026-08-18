#!/usr/bin/env python3
"""Numeric helpers shared by the vendor execution-provider compatibility harnesses.

Deterministic input synthesis and output comparison, factored out of the
Qualcomm QNN harness so the Apple CoreML and Intel OpenVINO harnesses (and any
future one) do not each carry their own copy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import onnx

_ELEM_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.BOOL: np.bool_,
}


def random_feeds(
    model: onnx.ModelProto, seed: int = 0, default_dim: int = 1
) -> Dict[str, np.ndarray]:
    """Deterministic random inputs for every graph input that is not an initializer.

    Unknown / dynamic dimensions (``dim_param`` or missing) are filled with
    ``default_dim``. Float inputs use a small range so backend emulation/kernels
    do not overflow; integer inputs use small non-negative values.
    """
    rng = np.random.RandomState(seed)
    initializers = {init.name for init in model.graph.initializer}
    feeds: Dict[str, np.ndarray] = {}
    for inp in model.graph.input:
        if inp.name in initializers:
            continue
        ttype = inp.type.tensor_type
        shape = []
        for dim in ttype.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(default_dim)
        np_dtype = _ELEM_TO_NP.get(ttype.elem_type, np.float32)
        if np.issubdtype(np_dtype, np.floating):
            feeds[inp.name] = (rng.rand(*shape).astype(np_dtype) - 0.5) * 2.0
        elif np_dtype == np.bool_:
            feeds[inp.name] = rng.rand(*shape) > 0.5
        else:
            feeds[inp.name] = rng.randint(0, 4, size=shape).astype(np_dtype)
    return feeds


def compare(
    reference: List[np.ndarray],
    candidate: List[np.ndarray],
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Tuple[bool, float]:
    """Return (all_close, max_abs_diff) over matching output tensors."""
    max_diff = 0.0
    ok = True
    for ref, cand in zip(reference, candidate):
        ref_a = np.asarray(ref, dtype=np.float64)
        cand_a = np.asarray(cand, dtype=np.float64)
        if ref_a.shape != cand_a.shape:
            return False, float("inf")
        diff = float(np.max(np.abs(ref_a - cand_a))) if ref_a.size else 0.0
        max_diff = max(max_diff, diff)
        if not np.allclose(ref_a, cand_a, rtol=rtol, atol=atol):
            ok = False
    return ok, max_diff
