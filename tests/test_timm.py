import onnxruntime
import timm
import torch

from onnxsim.test_utils import export_simplify_and_check_by_python_api


# From https://github.com/onnxsim/onnxsim/issues/307
#
# swin_tiny_patch4_window7_224 is also one of the two models named in NVIDIA
# Model Optimizer's onnx_ptq example (download_example_onnx.py); see
# test_vit_base_patch16_224 below for the other one.
def test_swin():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=False, num_classes=1000
    ).eval()
    dummy_input = torch.randn(*(1, 3, 224, 224), device="cpu")

    opt = export_simplify_and_check_by_python_api(
        model,
        (dummy_input,),
        export_kwargs={
            "verbose": False,
            "opset_version": 17,
            "do_constant_folding": True,
            "keep_initializers_as_inputs": True,
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        },
    )

    ort_sess = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = ort_sess.run(None, {"input": dummy_input.numpy().astype("float32")})
    assert outputs[0].shape == (1, 1000)


# vit_base_patch16_224 is the headline model used throughout NVIDIA Model
# Optimizer's examples/onnx_ptq README. Its download_example_onnx.py exports a
# timm model to ONNX, resolving the input shape from timm's data config:
#
#     model = timm.create_model(name, pretrained=True, num_classes=1000)
#     data_config = timm.data.resolve_model_data_config(model)
#     input_shape = (batch_size,) + data_config["input_size"]
#
# This test reproduces that export (with pretrained=False so no weights need to
# be downloaded -- the weight values don't affect the graph onnxsim simplifies)
# and confirms onnxsim can simplify the result.
def test_vit_base_patch16_224():
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=1000
    ).eval()
    # Resolve the input shape from timm's data config, as Model Optimizer does.
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

    ort_sess = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = ort_sess.run(None, {"input": dummy_input.numpy().astype("float32")})
    assert outputs[0].shape == (1, 1000)
