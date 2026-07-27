# Integration/regression test: RF-DETR (https://github.com/roboflow/rf-detr)
# exports simplified by onnxsim.
#
# RF-DETR's own ONNX export path calls ``onnxsim.simplify(..., check_n=3)``
# (rfdetr/export/_onnx/exporter.py), so onnxsim is part of RF-DETR's
# deployment pipeline. This test guards against onnxsim regressions on
# RF-DETR-style graphs.
#
# A *single small model* is used, deliberately built to cover the ops shared
# by most RF-DETR variants:
#
#   * the windowed DINOv2 ViT backbone (LayerNormalization + the Mul/Add
#     residual stream that onnxsim's constant folding rewrites),
#   * the deformable-attention decoder (GridSample / GatherElements / Gather),
#   * the DETR detection heads (MLP box/class heads, box decode), and
#   * -- in the segmentation parametrization -- the mask head (Resize/Conv).
#
# The detection model matches the RFDETRNano/Small/Medium/Base/Large family
# (5 variants); the segmentation model matches the RFDETRSeg* family
# (7 variants). Between them they exercise 12 of RF-DETR's 13 shipped
# variants. The model is built with random weights via
# ``build_model_from_config`` (no checkpoint download) and at a small
# resolution -- only tensor sizes shrink, the graph structure and op set are
# identical to the full models, so regression coverage is unchanged while the
# test stays fast and offline.
#
# ``rfdetr`` is NOT a normal test dependency: it pins ``onnxsim<0.6.0``, so
# installing it into the environment that tests the built wheel would
# downgrade onnxsim. The test therefore skips unless ``rfdetr`` is already
# importable. To run it locally::
#
#     pip install torch rfdetr onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_rfdetr.py -v

import os

import onnxruntime
import pytest

# Must be set before importing rfdetr / transformers so the DINOv2 backbone is
# constructed with random weights instead of reaching Hugging Face Hub. The
# patch_size=16 configs below also force ``load_dinov2_weights=False``; this is
# belt-and-suspenders so the test never touches the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

torch = pytest.importorskip("torch")
pytest.importorskip("rfdetr")

# onnxsim.test_utils imports torch at module load, so it must follow the
# importorskip guard above (hence the E402 exemption).
from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402

# Small square input; must be divisible by patch_size * num_windows = 16 * 2.
RESOLUTION = 96
NUM_QUERIES = 50


def _build_small_rfdetr(segmentation_head: bool):
    """Build a tiny, randomly-initialised RF-DETR model in ONNX-export mode.

    Uses the RFDETRNano config (encoder ``dinov2_windowed_small``, patch_size
    16 -> no backbone download) at a reduced resolution and query count.
    """
    from rfdetr.config import RFDETRNanoConfig
    from rfdetr.models import build_model_from_config

    torch.manual_seed(0)
    config = RFDETRNanoConfig(
        resolution=RESOLUTION,
        num_queries=NUM_QUERIES,
        segmentation_head=segmentation_head,
    )
    model = build_model_from_config(config).eval()
    # Switch LWDETR into export mode: forward then returns a tuple of tensors
    # (dets, labels[, masks]) instead of the training-time dict.
    model.export()
    return model


@pytest.mark.parametrize(
    "segmentation_head, output_names",
    [
        pytest.param(False, ["dets", "labels"], id="detection"),
        pytest.param(True, ["dets", "labels", "masks"], id="segmentation"),
    ],
)
def test_rfdetr_export(segmentation_head, output_names):
    model = _build_small_rfdetr(segmentation_head)
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)

    opt = export_simplify_and_check_by_python_api(
        model,
        (dummy_input,),
        export_kwargs={
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": output_names,
        },
    )

    # The simplified graph must still carry the ops that make RF-DETR
    # RF-DETR: windowed-ViT LayerNorm and deformable-attention sampling.
    # If a future onnxsim change silently drops/miscompiles these, the
    # check above would already fail, but assert their presence too so the
    # test documents what it is protecting.
    op_types = {node.op_type for node in opt.graph.node}
    assert "LayerNormalization" in op_types
    assert "GridSample" in op_types or "GatherElements" in op_types
    if segmentation_head:
        assert "Resize" in op_types  # mask head upsampling

    # The simplified model must run and produce the expected output signature.
    session = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = session.run(None, {"input": dummy_input.numpy().astype("float32")})
    assert [o.name for o in session.get_outputs()] == output_names
    dets, labels = outputs[0], outputs[1]
    assert dets.shape == (1, NUM_QUERIES, 4)
    assert labels.shape[:2] == (1, NUM_QUERIES)
    if segmentation_head:
        assert outputs[2].shape[:2] == (1, NUM_QUERIES)  # per-query masks
