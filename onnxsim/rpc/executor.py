"""A remote ``ModelExecutor`` for onnxsim's constant folder (see ``docs/dlpack-executor.md``)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import onnx

from onnxsim.onnx_simplifier import PyModelExecutor

from .client import Session


class RemoteModelExecutor(PyModelExecutor):
    """Runs each throwaway fold-group sub-model on the server instead of locally.

    Constant folding then uses the *target's* numerics and kernels, and a device-only operator
    that the host's onnxruntime lacks can still be folded.
    """

    def __init__(self, session: Session, providers: Optional[Sequence[str]] = None):
        super().__init__(providers)
        self._session = session

    # The base class annotates these as str/List[str] although the C++ core passes serialized bytes.
    def Run(self, model_str: str, inputs_str: List[str]) -> List[bytes]:  # noqa: N802
        model = onnx.ModelProto()
        model.ParseFromString(model_str)
        arrays = []
        for serialized in inputs_str:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(serialized)
            arrays.append(onnx.numpy_helper.to_array(tensor))
        inputs = dict(zip((i.name for i in model.graph.input), arrays))
        # Only provider names cross the wire (per-provider option dicts are not transferred).
        names = (
            [p if isinstance(p, str) else p[0] for p in self.providers]
            if self.providers
            else None
        )
        outputs = self._session.run(model, inputs, providers=names)
        return [
            onnx.numpy_helper.from_array(np.asarray(value)).SerializeToString()
            for value in outputs.values()
        ]
