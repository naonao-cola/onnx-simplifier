# Integration/regression test: the latest Ultralytics YOLO models
# (https://github.com/ultralytics/ultralytics) simplified by onnxsim.
#
# Ultralytics YOLO is the most widely deployed real-time detector family, and
# its ONNX export path has long offered a "simplify" step backed by onnxsim
# (the ``simplify=True`` export flag). onnxsim is therefore part of the YOLO
# deployment pipeline, and this test guards against onnxsim regressions on
# YOLO-style graphs.
#
# Coverage targets the *newest* generations and multiple task heads:
#
#   * ``yolo26n``      -- YOLO26 detect (the current flagship; NMS-free head),
#   * ``yolo26n-seg``  -- YOLO26 instance segmentation (mask proto + Resize),
#   * ``yolo12n``      -- YOLO12 detect (area-attention "A2" backbone),
#   * ``yolo11n-pose`` -- YOLO11 keypoint head.
#
# Between them these exercise the Conv/BatchNorm backbones onnxsim fuses, the
# detection/segmentation/pose heads, and the anchor-grid constant folding that
# dominates YOLO's node-count reduction. The standalone compatibility harness
# in ``scripts/yolo/simplify_yolo.py`` runs the full generation x task-head
# matrix (see ``scripts/yolo/RESULTS.md``).
#
# Each model is built from its Ultralytics YAML config with **random weights**
# (no checkpoint download): the graph structure onnxsim rewrites is identical
# to the pretrained ``.pt`` model, so regression coverage is unchanged while
# the test stays fast and offline.
#
# ``ultralytics`` is not a normal test dependency (it is a large install and
# pulls torch). The test therefore ``importorskip``s it and skips unless it is
# already available. To run it locally::
#
#     pip install ultralytics onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_yolo.py -v

import os

import onnx
import onnxruntime
import pytest

# Keep Ultralytics from reaching the network (e.g. font/asset fetches) during
# export. Must be set before ultralytics is imported.
os.environ.setdefault("YOLO_OFFLINE", "True")

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import onnxsim  # noqa: E402


@pytest.mark.parametrize(
    "spec, task, imgsz, expected_op",
    [
        pytest.param("yolo26n", "detect", 640, "Conv", id="yolo26-detect"),
        pytest.param("yolo26n-seg", "segment", 640, "Resize", id="yolo26-segment"),
        pytest.param("yolo12n", "detect", 640, "Softmax", id="yolo12-detect"),
        pytest.param("yolo11n-pose", "pose", 640, "Conv", id="yolo11-pose"),
    ],
)
def test_yolo_export_simplify(tmp_path, spec, task, imgsz, expected_op):
    from ultralytics import YOLO

    # Build from the YAML config -> random weights, no checkpoint download.
    model = YOLO(f"{spec}.yaml")
    assert model.task == task

    onnx_path = str(
        model.export(
            format="onnx",
            opset=17,
            imgsz=imgsz,
            simplify=False,  # we run onnxsim ourselves, below
            dynamic=False,
            verbose=False,
        )
    )
    # Ultralytics writes next to the (nonexistent) weights; move into tmp_path.
    dst = os.path.join(str(tmp_path), os.path.basename(onnx_path))
    if os.path.abspath(dst) != os.path.abspath(onnx_path):
        os.replace(onnx_path, dst)
        onnx_path = dst

    original = onnx.load(onnx_path)
    nodes_before = len(original.graph.node)

    # onnxsim's numerical equivalence check must pass (check_n=3) -- this is the
    # same call Ultralytics' export makes with simplify=True.
    sim_model, check_ok = onnxsim.simplify(onnx_path, check_n=3)
    assert check_ok, f"{spec}: onnxsim numerical check failed"

    nodes_after = len(sim_model.graph.node)
    # Simplification must actually reduce the graph (YOLO exports carry a lot of
    # foldable anchor-grid / shape arithmetic).
    assert nodes_after < nodes_before, f"{spec}: {nodes_before} -> {nodes_after}"

    op_types = {node.op_type for node in sim_model.graph.node}
    assert expected_op in op_types, (
        f"{spec}: expected {expected_op} in simplified graph"
    )

    # The simplified model must load and run in onnxruntime and preserve the
    # export's input/output signature.
    orig_out_names = [o.name for o in original.graph.output]
    session = onnxruntime.InferenceSession(sim_model.SerializeToString())
    assert [o.name for o in session.get_outputs()] == orig_out_names

    inp = session.get_inputs()[0]
    import numpy as np

    dummy = np.random.RandomState(0).rand(1, 3, imgsz, imgsz).astype(np.float32)
    outputs = session.run(None, {inp.name: dummy})
    assert len(outputs) == len(orig_out_names)
    # Batch dimension is preserved through simplification.
    assert outputs[0].shape[0] == 1
