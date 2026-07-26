import os
import tempfile
from typing import Any, Callable, Dict, Optional

import onnx
import torch

import onnxsim


def _always_valid(_: Any) -> bool:
    return True


def export_simplify_and_check_by_python_api(
    m: torch.nn.Module,
    input: Any,
    *,
    is_model_valid: Optional[Callable[[Any], bool]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    simplify_kwargs: Optional[Dict[str, Any]] = None,
) -> onnx.ModelProto:
    if is_model_valid is None:
        is_model_valid = _always_valid
    if export_kwargs is None:
        export_kwargs = {}
    if simplify_kwargs is None:
        simplify_kwargs = {}
    # Use legacy TorchScript-based exporter (dynamo=False) for ScriptModule support
    # PyTorch 2.9+ defaults to dynamo=True which doesn't support ScriptModule
    if "dynamo" not in export_kwargs:
        export_kwargs["dynamo"] = False
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_fn = os.path.join(tmpdirname, "tmp.onnx")
        torch.onnx.export(m, input, model_fn, **export_kwargs)
        model = onnx.load(model_fn)
        if not is_model_valid(model):
            raise AssertionError(f"model is invalid:\n{model}")
        # read the model from filesystem to support >2GB large model
        sim_model, check_ok = onnxsim.simplify(model_fn, check_n=3, **simplify_kwargs)
        assert check_ok
        return sim_model
