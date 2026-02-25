import torch
import onnx
import onnxsim


def test_onnx_simplifier():
    def _create_dummy_model():
        class MockModel(torch.nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = MockModel()
        dummy_input = torch.randn(1, 10)
        onnx_file = "dummy_model.onnx"
        torch.onnx.export(model, dummy_input, onnx_file)
        return onnx_file

    onnx_model_path = _create_dummy_model()
    onnx_model = onnx.load(onnx_model_path)
    simple_model, _ = onnxsim.simplify(onnx_model, perform_optimization=False, skip_shape_inference=True)
    assert simple_model is not None
