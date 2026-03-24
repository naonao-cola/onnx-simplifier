import torch
import onnx
import onnxruntime
import onnxsim
import timm

from onnxsim.test_utils import export_simplify_and_check_by_python_api


# From https://github.com/onnxsim/onnxsim/issues/307
def test_swin():
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=1000).eval()
    dummy_input = torch.randn(*(1, 3, 224, 224), device='cpu')

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
            "dynamic_axes": {"input":{0:"batch_size"},"output":{0:"batch_size"}},
        }
    )

    ort_sess = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = ort_sess.run(None, {'input': dummy_input.numpy().astype('float32')})
    assert outputs[0].shape == (1, 1000)
