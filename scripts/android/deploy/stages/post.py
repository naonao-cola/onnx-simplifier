"""Post-processing graphs that run on the CPU after the HTP part (`postprocess:` in a spec).

Built from standard ONNX ops so the same graph runs in host ORT (for the accuracy reference) and
in the phone's ORT CPU session (for the benchmark); nothing model-specific lives in the runtime.

  kind: yolo_detect   Ultralytics YOLOv8/11 head output (1, 4+nc, N): cx,cy,w,h + class scores.
                      -> det_boxes (K,4) xyxy in model-input pixels, det_scores (K), det_classes (K)
                      via ONNX NonMaxSuppression (class-wise, like Ultralytics' default).
      conf: 0.25, iou: 0.7, max_det: 300
  kind: yolo_end2end  Ultralytics YOLO26 (NMS-free, one-to-one head). Either the head output
                      (1, 4+nc, N): x1,y1,x2,y2 + class scores -> the same two-stage top-k as
                      Ultralytics' Detect.get_topk_index (top max_det anchors by best class
                      score, then top max_det (anchor, class) pairs among them), no NMS; or
                      Ultralytics' own end2end output (1, max_det, 6) when the top-k already ran
                      on the HTP -> just split into the same three outputs.
      max_det: 300
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _c(name, arr, dt=None):
    a = np.asarray(arr, dtype=dt)
    return numpy_helper.from_array(a, name)


def yolo_detect(model_path: Path, out_dir: Path, spec: dict) -> dict:
    m = onnx.load(str(model_path), load_external_data=False)
    head_name = spec.get("head", m.graph.output[0].name)
    rw_meta_p = model_path.parent / "rewrite_meta.json"
    rw = json.loads(rw_meta_p.read_text()) if rw_meta_p.exists() else {}
    q = (rw.get("uint8_outputs") or {}).get(head_name)  # head handed out as uint8?
    src = q["name"] if q else head_name
    go = next(o for o in m.graph.output if o.name == src)
    _, ch, n = [d.dim_value for d in go.type.tensor_type.shape.dim]
    nc = ch - 4
    nodes, inits = [], []
    x = src
    if q:
        inits += [_c("dq_s", q["scale"], np.float32), _c("dq_z", q["zero_point"], np.uint8)]
        nodes.append(helper.make_node("DequantizeLinear", [src, "dq_s", "dq_z"], ["head_f"]))
        x = "head_f"
    inits += [_c("s0", [0]), _c("s2", [2]), _c("s4", [4]), _c("sN", [ch]), _c("ax1", [1]),
              _c("half", 0.5, np.float32), _c("max_out", [spec.get("max_det", 300)], np.int64),
              _c("iou", [spec.get("iou", 0.7)], np.float32), _c("conf", [spec.get("conf", 0.25)], np.float32),
              _c("i0", [0]), _c("i1", [1]), _c("i2", [2]), _c("i3", [3]), _c("i12", [1, 2])]
    nodes += [
        helper.make_node("Slice", [x, "s0", "s2", "ax1"], ["cxcy"]),
        helper.make_node("Slice", [x, "s2", "s4", "ax1"], ["wh"]),
        helper.make_node("Slice", [x, "s4", "sN", "ax1"], ["scores"]),  # (1, nc, N)
        helper.make_node("Mul", ["wh", "half"], ["hwh"]),
        helper.make_node("Sub", ["cxcy", "hwh"], ["xy1"]),
        helper.make_node("Add", ["cxcy", "hwh"], ["xy2"]),
        helper.make_node("Concat", ["xy1", "xy2"], ["xyxy_t"], axis=1),  # (1, 4, N)
        helper.make_node("Transpose", ["xyxy_t"], ["boxes"], perm=[0, 2, 1]),  # (1, N, 4)
        helper.make_node("NonMaxSuppression", ["boxes", "scores", "max_out", "iou", "conf"], ["sel"],
                         center_point_box=0),  # (K, 3): batch, class, box
        helper.make_node("Gather", ["sel", "i1"], ["cls_k"], axis=1),
        helper.make_node("Gather", ["sel", "i2"], ["box_k"], axis=1),
        helper.make_node("Squeeze", ["cls_k", "i1"], ["det_classes"]),
        helper.make_node("Squeeze", ["box_k", "i1"], ["box_i"]),
        helper.make_node("Squeeze", ["boxes", "i0"], ["boxes0"]),
        helper.make_node("Gather", ["boxes0", "box_i"], ["det_boxes"], axis=0),
        helper.make_node("Squeeze", ["scores", "i0"], ["scores0"]),  # (nc, N)
        helper.make_node("Slice", ["sel", "i1", "i3", "i1"], ["cb"]),  # (K, 2): class, box
        helper.make_node("GatherND", ["scores0", "cb"], ["det_scores"]),
    ]
    in_t = TensorProto.UINT8 if q else TensorProto.FLOAT
    g = helper.make_graph(
        nodes, "yolo_detect_post", [helper.make_tensor_value_info(src, in_t, [1, ch, n])],
        [helper.make_tensor_value_info("det_boxes", TensorProto.FLOAT, ["K", 4]),
         helper.make_tensor_value_info("det_scores", TensorProto.FLOAT, ["K"]),
         helper.make_tensor_value_info("det_classes", TensorProto.INT64, ["K"])], inits)
    pm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    pm.ir_version = 8
    onnx.checker.check_model(pm)
    onnx.save(pm, str(out_dir / "post.onnx"))
    return {"kind": "yolo_detect", "input": src, "input_uint8": bool(q), "num_classes": nc,
            "outputs": ["det_boxes", "det_scores", "det_classes"]}


def _head_input(m, spec, model_path):
    """(graph input name, uint8 qparams or None, dims) of the HTP model's head output."""
    head_name = spec.get("head", m.graph.output[0].name)
    rw_meta_p = model_path.parent / "rewrite_meta.json"
    rw = json.loads(rw_meta_p.read_text()) if rw_meta_p.exists() else {}
    q = (rw.get("uint8_outputs") or {}).get(head_name)
    src = q["name"] if q else head_name
    go = next(o for o in m.graph.output if o.name == src)
    return src, q, [d.dim_value for d in go.type.tensor_type.shape.dim]


def yolo_end2end(model_path: Path, out_dir: Path, spec: dict) -> dict:
    m = onnx.load(str(model_path), load_external_data=False)
    src, q, dims = _head_input(m, spec, model_path)
    nodes, inits = [], []
    x = src
    if q:
        inits += [_c("dq_s", q["scale"], np.float32), _c("dq_z", q["zero_point"], np.uint8)]
        nodes.append(helper.make_node("DequantizeLinear", [src, "dq_s", "dq_z"], ["head_f"]))
        x = "head_f"
    inits.append(_c("i0", [0]))
    if dims[-1] == 6:  # (1, K, 6): top-k already in the model
        k = dims[1]
        inits += [_c("s0", [0]), _c("s4", [4]), _c("s5", [5]), _c("s6", [6])]
        nodes += [
            helper.make_node("Squeeze", [x, "i0"], ["dets"]),  # (K, 6)
            helper.make_node("Slice", ["dets", "s0", "s4", "i1c"], ["det_boxes"]),
            helper.make_node("Slice", ["dets", "s4", "s5", "i1c"], ["sc"]),
            helper.make_node("Slice", ["dets", "s5", "s6", "i1c"], ["cl"]),
            helper.make_node("Squeeze", ["sc", "i1c"], ["det_scores"]),
            helper.make_node("Squeeze", ["cl", "i1c"], ["clf"]),
            helper.make_node("Cast", ["clf"], ["det_classes"], to=TensorProto.INT64),
        ]
        inits.append(_c("i1c", [1]))
        nc = None
    else:
        _, ch, n = dims
        nc = ch - 4
        k = min(spec.get("max_det", 300), n)
        inits += [_c("s0", [0]), _c("s4", [4]), _c("sN", [ch]), _c("ax2", [2]), _c("k", [k]),
                  _c("nc", nc, np.int64), _c("flat", [1, k * nc])]
        nodes += [
            helper.make_node("Transpose", [x], ["xt"], perm=[0, 2, 1]),  # (1, N, 4+nc)
            helper.make_node("Slice", ["xt", "s0", "s4", "ax2"], ["boxes"]),  # (1, N, 4)
            helper.make_node("Slice", ["xt", "s4", "sN", "ax2"], ["scores"]),  # (1, N, nc)
            helper.make_node("ReduceMax", ["scores"], ["best"], axes=[-1], keepdims=0),  # (1, N)
            helper.make_node("TopK", ["best", "k"], ["_v0", "anchor_k"], axis=1),  # (1, k)
            helper.make_node("Squeeze", ["anchor_k", "i0"], ["anchor0"]),  # (k,)
            helper.make_node("Squeeze", ["scores", "i0"], ["scores0"]),  # (N, nc)
            helper.make_node("Gather", ["scores0", "anchor0"], ["cand"], axis=0),  # (k, nc)
            helper.make_node("Reshape", ["cand", "flat"], ["cand_f"]),  # (1, k*nc)
            helper.make_node("TopK", ["cand_f", "k"], ["top_s", "top_i"], axis=1),  # (1, k)
            helper.make_node("Squeeze", ["top_s", "i0"], ["det_scores"]),
            helper.make_node("Squeeze", ["top_i", "i0"], ["top_i0"]),
            helper.make_node("Mod", ["top_i0", "nc"], ["det_classes"]),
            helper.make_node("Div", ["top_i0", "nc"], ["slot"]),
            helper.make_node("Gather", ["anchor0", "slot"], ["anchor_sel"], axis=0),
            helper.make_node("Squeeze", ["boxes", "i0"], ["boxes0"]),
            helper.make_node("Gather", ["boxes0", "anchor_sel"], ["det_boxes"], axis=0),
        ]
    in_t = TensorProto.UINT8 if q else TensorProto.FLOAT
    g = helper.make_graph(
        nodes, "yolo_end2end_post", [helper.make_tensor_value_info(src, in_t, dims)],
        [helper.make_tensor_value_info("det_boxes", TensorProto.FLOAT, [k, 4]),
         helper.make_tensor_value_info("det_scores", TensorProto.FLOAT, [k]),
         helper.make_tensor_value_info("det_classes", TensorProto.INT64, [k])], inits)
    pm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    pm.ir_version = 8
    onnx.checker.check_model(pm)
    onnx.save(pm, str(out_dir / "post.onnx"))
    return {"kind": "yolo_end2end", "input": src, "input_uint8": bool(q), "num_classes": nc, "max_det": k,
            "topk_on_htp": dims[-1] == 6, "outputs": ["det_boxes", "det_scores", "det_classes"]}


KINDS = {"yolo_detect": yolo_detect, "yolo_end2end": yolo_end2end}


def build(spec: dict, model_path: Path, out_dir: Path) -> dict:
    info = KINDS[spec["kind"]](model_path, out_dir, spec)
    print(f"  post.onnx: {info}")
    return info
