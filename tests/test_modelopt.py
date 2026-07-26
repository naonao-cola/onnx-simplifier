"""Run onnxsim on the models that NVIDIA Model Optimizer's ONNX PTQ example
exports.

NVIDIA Model Optimizer (https://github.com/NVIDIA/Model-Optimizer) optimizes
ONNX models. Its ``examples/onnx_ptq`` flow produces the ONNX to quantize with
``download_example_onnx.py``, which exports a ``timm`` model to ONNX:

    model = timm.create_model(name, pretrained=True, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    input_shape = (batch_size,) + data_config["input_size"]
    # export (model, randn(input_shape)) to ONNX

The two models named in that script's help text are ``vit_base_patch16_224``
and ``swin_tiny_patch4_window7_224``. These tests reproduce the same export
(resolving the input size from timm's data config exactly as Model Optimizer
does) and confirm onnxsim can simplify the result. ``pretrained=False`` is used
so the tests don't need to download pretrained weights over the network -- the
weight values don't affect the graph structure that onnxsim simplifies.
"""

import onnxruntime
import timm
import torch

from onnxsim.test_utils import export_simplify_and_check_by_python_api


def _export_and_simplify(model_name: str):
    """Export a timm model the way Model Optimizer's onnx_ptq example does and
    simplify it with onnxsim."""
    model = timm.create_model(model_name, pretrained=False, num_classes=1000).eval()
    # Model Optimizer resolves the input shape from timm's data config.
    data_config = timm.data.resolve_model_data_config(model)
    input_shape = (1,) + tuple(data_config["input_size"])
    dummy_input = torch.randn(*input_shape, device="cpu")

    opt = export_simplify_and_check_by_python_api(
        model,
        (dummy_input,),
        export_kwargs={
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["output"],
        },
    )

    ort_sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"input": dummy_input.numpy().astype("float32")})
    assert outputs[0].shape == (1, 1000)


# The headline model used throughout Model Optimizer's onnx_ptq README.
def test_vit_base_patch16_224():
    _export_and_simplify("vit_base_patch16_224")


# The second model named in download_example_onnx.py's help text.
def test_swin_tiny_patch4_window7_224():
    _export_and_simplify("swin_tiny_patch4_window7_224")
