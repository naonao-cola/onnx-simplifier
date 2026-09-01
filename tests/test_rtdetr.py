# Integration/regression test: RT-DETR and RT-DETRv2
# (https://huggingface.co/docs/transformers/model_doc/rt_detr,
# https://huggingface.co/docs/transformers/model_doc/rt_detr_v2) exports
# simplified by onnxsim.
#
# Unlike RT-DETRv4 (see tests/test_rtdetrv4.py, which has no pip package and
# needs a hand-built onnx.parser graph), RT-DETR and RT-DETRv2 are shipped
# directly in Hugging Face transformers as ``RTDetrForObjectDetection`` /
# ``RTDetrV2ForObjectDetection``, so a real, tiny, offline instance can be
# built the same way transformers' own fast unit tests do (see
# ``RTDetrModelTester``/``RTDetrV2ModelTester`` in
# ``tests/models/rt_detr{,_v2}/test_modeling_rt_detr{,_v2}.py`` upstream): a
# small ``RTDetrResNetConfig`` backbone plus a handful of tiny encoder/decoder
# dimensions, no checkpoint download. ``disable_custom_kernels=True`` forces
# the deformable-attention decoder's pure-PyTorch fallback path (the fused
# CUDA kernel isn't traceable), which is exactly what a CPU ONNX export uses.
#
# Between the two variants and RT-DETRv4's synthetic graph, this covers the
# op patterns across the whole RT-DETR family that onnxsim must get right:
#
#   * a ResNet-D backbone (Conv/BatchNormalization folding, AveragePool
#     downsample-shortcut -- a HGNetV2 backbone, as RT-DETRv4 uses, does not
#     exercise this),
#   * AIFI: a transformer encoder layer with a 2D sin/cos position embedding
#     that is a pure function of the (static) feature-map size and must
#     collapse away during simplification,
#   * the hybrid encoder's CCFM feature-pyramid fusion (Resize + Concat),
#   * a multi-scale deformable-attention decoder (GridSample), with
#     RT-DETRv2 additionally varying the sampling offsets per feature level
#     (``decoder_n_levels``), and
#   * the NMS-free postprocessor's top-k label/box decode (TopK), matching
#     RT-DETRv4's own postprocessor shape.
#
# torch and transformers are not normal test dependencies (transformers pulls
# a large stack), so this skips unless they are already importable. To run it
# locally::
#
#     pip install torch transformers onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_rtdetr.py -v

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# onnxsim.test_utils imports torch at module load, so it must follow the
# importorskip guard above (hence the E402 exemption).
from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402

IMAGE_SIZE = 64  # divisible by the backbone's stride-32 stem, kept tiny for speed


def _tiny_backbone_config():
    # Mirrors upstream transformers' own RTDetr{,V2}ModelTester.get_config():
    # a 4-stage ResNet-D with tiny channel widths, no pretrained weights.
    from transformers import RTDetrResNetConfig

    hidden_sizes = [10, 20, 30, 40]
    backbone_config = RTDetrResNetConfig(
        embeddings_size=10,
        hidden_sizes=hidden_sizes,
        depths=[1, 1, 2, 1],
        out_features=["stage2", "stage3", "stage4"],
        out_indices=[2, 3, 4],
    )
    return backbone_config, hidden_sizes


def _tiny_config_kwargs(hidden_sizes):
    return dict(
        encoder_hidden_dim=32,
        encoder_in_channels=hidden_sizes[1:],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=2,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        d_model=32,
        num_queries=30,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        image_size=IMAGE_SIZE,
        disable_custom_kernels=True,
        with_box_refine=True,
        num_labels=10,
    )


def _build_tiny_rtdetr():
    from transformers import RTDetrConfig, RTDetrForObjectDetection

    backbone_config, hidden_sizes = _tiny_backbone_config()
    config = RTDetrConfig(
        backbone_config=backbone_config, **_tiny_config_kwargs(hidden_sizes)
    )
    torch.manual_seed(0)
    return RTDetrForObjectDetection(config).eval()


def _build_tiny_rtdetr_v2():
    from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection

    backbone_config, hidden_sizes = _tiny_backbone_config()
    kwargs = _tiny_config_kwargs(hidden_sizes)
    kwargs["decoder_n_levels"] = 3  # v2-only: per-level deformable offsets
    config = RTDetrV2Config(backbone_config=backbone_config, **kwargs)
    torch.manual_seed(0)
    return RTDetrV2ForObjectDetection(config).eval()


class _LogitsAndBoxes(torch.nn.Module):
    """Unwraps the ``RTDetrObjectDetectionOutput`` dataclass into a plain
    tuple by field name -- ``return_dict=False``'s tuple order depends on
    which optional fields happen to be non-None, so this is the robust way
    to export a fixed (logits, pred_boxes) signature."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values)
        return out.logits, out.pred_boxes


@pytest.mark.parametrize(
    "build_model",
    [
        pytest.param(_build_tiny_rtdetr, id="rtdetr"),
        pytest.param(_build_tiny_rtdetr_v2, id="rtdetr_v2"),
    ],
)
def test_rtdetr_export_simplify(build_model):
    model = build_model()
    num_queries = model.config.num_queries
    num_labels = model.config.num_labels
    wrapped = _LogitsAndBoxes(model)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    opt = export_simplify_and_check_by_python_api(
        wrapped,
        (dummy_input,),
        export_kwargs={
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["pixel_values"],
            "output_names": ["logits", "pred_boxes"],
        },
    )

    # The ops that make RT-DETR(v2) RT-DETR(v2) must survive simplification.
    op_types = {node.op_type for node in opt.graph.node}
    assert "LayerNormalization" in op_types  # AIFI transformer encoder
    assert "GridSample" in op_types  # multi-scale deformable attention
    assert "Resize" in op_types  # CCFM feature-pyramid fusion
    assert "TopK" in op_types  # NMS-free postprocessor

    # The simplified model must still run and match the export's signature.
    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(opt.SerializeToString())
    assert [o.name for o in session.get_outputs()] == ["logits", "pred_boxes"]
    logits, pred_boxes = session.run(
        None, {"pixel_values": dummy_input.numpy().astype("float32")}
    )
    assert logits.shape == (1, num_queries, num_labels)
    assert pred_boxes.shape == (1, num_queries, 4)
