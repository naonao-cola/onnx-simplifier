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

    def Run(self, model_str: bytes, inputs_str: List[bytes]) -> List[bytes]:  # noqa: N802
        model = onnx.ModelProto()
        model.ParseFromString(model_str)
        arrays = []
        for serialized in inputs_str:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(serialized)
            arrays.append(onnx.numpy_helper.to_array(tensor))
        inputs = dict(zip((i.name for i in model.graph.input), arrays))
        outputs = self._session.run(model, inputs, providers=self.providers)
        return [
            onnx.numpy_helper.from_array(np.asarray(value)).SerializeToString()
            for value in outputs.values()
        ]
