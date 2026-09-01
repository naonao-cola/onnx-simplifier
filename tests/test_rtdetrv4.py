# Integration/regression test: RT-DETRv4
# (https://github.com/RT-DETRs/RT-DETRv4, arXiv:2510.25257) exports
# simplified by onnxsim.
#
# RT-DETRv4's own deployment tool (``tools/deployment/export_onnx.py``) calls
# ``onnxsim.simplify(...)`` right after ``torch.onnx.export`` -- same as
# RF-DETR (see ``tests/test_rfdetr.py``) and Ultralytics YOLO (see
# ``tests/test_yolo.py``) -- so onnxsim is part of RT-DETRv4's deployment
# pipeline. Its export signature is ``images``/``orig_target_sizes`` in,
# ``labels``/``boxes``/``scores`` out (an NMS-free, top-k decode -- no
# separate NMS op), confirmed from that script.
#
# Unlike RF-DETR/YOLO, RT-DETRv4 has no pip package (it is a config/checkpoint
# research repo, not a library) and is not yet available through Hugging Face
# transformers, so there is no small, offline, importable reference
# implementation to build a tiny instance from the way the RF-DETR/YOLO tests
# do. Per CLAUDE.md, this instead builds a small representative graph
# directly via ``onnx.parser`` that exercises the op patterns specific to the
# RT-DETR family (confirmed from the paper and the public RT-DETR/RT-DETRv2/
# D-FINE reference implementations RT-DETRv4 is built on):
#
#   * an HGNetV2-style Conv+BatchNormalization stem (routine Conv/BN folding),
#   * AIFI: a transformer encoder layer over the flattened last feature map,
#     with a 2D sin/cos position embedding that is a pure function of the
#     (static) feature-map size -- built here from Range/Tile/Sin/Cos, it
#     must collapse entirely into one Constant during simplification,
#   * anchor/reference-point generation -- likewise a pure function of the
#     feature-map size (Range/Tile/Log inverse-sigmoid), must also collapse
#     into one Constant,
#   * a deformable-attention decoder layer, standing in for real deformable
#     sampling with a single GridSample (matching test_rfdetr.py's own
#     stand-in for the same op), which onnxsim must preserve, and
#   * the NMS-free postprocessor's top-k label/box decode (TopK + Mod/Div to
#     split the flattened class*query index, then Gather), including scaling
#     boxes by the runtime ``orig_target_sizes`` input -- which must survive
#     simplification unfolded since it depends on a real graph input.

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime
from onnx import parser

import onnxsim

H = W = 8  # AIFI/decoder feature-map size, after the 2-stage stride-2 stem
C = 16  # hidden dim
FFN = 32  # feed-forward width
NUM_QUERIES = 8
NUM_CLASSES = 4
NUM_OUT = 5  # postprocessor top-k


def _model(body, initializer, opset=18, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _linear(rng, in_dim, out_dim, w_name, b_name, scale=0.1):
    w = _f32(rng.randn(in_dim, out_dim) * scale, w_name)
    b = _f32(rng.randn(out_dim) * scale, b_name)
    return [w, b]


def _ln(dim, scale_name, bias_name):
    # Identity-ish init (scale=1, bias=0) keeps every LayerNormalization
    # numerically well-behaved regardless of what feeds it.
    return [_f32(np.ones(dim), scale_name), _f32(np.zeros(dim), bias_name)]


def _self_attn_sublayer(rng, prefix, x, dim):
    """Post-LN self-attention: LN(x + MHA(x)). Returns (text, out_name, inits)."""
    inits = []
    inits += _linear(rng, dim, dim, f"{prefix}_Wq", f"{prefix}_bq")
    inits += _linear(rng, dim, dim, f"{prefix}_Wk", f"{prefix}_bk")
    inits += _linear(rng, dim, dim, f"{prefix}_Wv", f"{prefix}_bv")
    inits += _linear(rng, dim, dim, f"{prefix}_Wo", f"{prefix}_bo")
    inits += _ln(dim, f"{prefix}_ln_scale", f"{prefix}_ln_bias")
    text = f"""
      {prefix}_q = MatMul({x}, {prefix}_Wq)
      {prefix}_qb = Add({prefix}_q, {prefix}_bq)
      {prefix}_k = MatMul({x}, {prefix}_Wk)
      {prefix}_kb = Add({prefix}_k, {prefix}_bk)
      {prefix}_v = MatMul({x}, {prefix}_Wv)
      {prefix}_vb = Add({prefix}_v, {prefix}_bv)
      {prefix}_kt = Transpose<perm=[0,2,1]>({prefix}_kb)
      {prefix}_scores = MatMul({prefix}_qb, {prefix}_kt)
      {prefix}_scaled = Mul({prefix}_scores, attn_scale)
      {prefix}_attn = Softmax<axis=-1>({prefix}_scaled)
      {prefix}_ctx = MatMul({prefix}_attn, {prefix}_vb)
      {prefix}_proj = MatMul({prefix}_ctx, {prefix}_Wo)
      {prefix}_projb = Add({prefix}_proj, {prefix}_bo)
      {prefix}_res = Add({x}, {prefix}_projb)
      {prefix}_out = LayerNormalization<axis=-1>({prefix}_res, {prefix}_ln_scale, {prefix}_ln_bias)
    """
    return text, f"{prefix}_out", inits


def _ffn_sublayer(rng, prefix, x, dim, ffn_dim):
    """Post-LN feed-forward: LN(x + FFN(x)). Returns (text, out_name, inits)."""
    inits = []
    inits += _linear(rng, dim, ffn_dim, f"{prefix}_W1", f"{prefix}_b1")
    inits += _linear(rng, ffn_dim, dim, f"{prefix}_W2", f"{prefix}_b2")
    inits += _ln(dim, f"{prefix}_ln_scale", f"{prefix}_ln_bias")
    text = f"""
      {prefix}_h = MatMul({x}, {prefix}_W1)
      {prefix}_hb = Add({prefix}_h, {prefix}_b1)
      {prefix}_act = Relu({prefix}_hb)
      {prefix}_o = MatMul({prefix}_act, {prefix}_W2)
      {prefix}_ob = Add({prefix}_o, {prefix}_b2)
      {prefix}_res = Add({x}, {prefix}_ob)
      {prefix}_out = LayerNormalization<axis=-1>({prefix}_res, {prefix}_ln_scale, {prefix}_ln_bias)
    """
    return text, f"{prefix}_out", inits


def _cross_attn_sublayer(prefix, x, ref_unact, memory_map, dim, num_queries):
    """Deformable-attention stand-in: sample ``memory_map`` at each query's
    reference point via GridSample, then LN(x + sampled). Returns (text,
    out_name, inits)."""
    inits = _ln(dim, f"{prefix}_ln_scale", f"{prefix}_ln_bias")
    inits += [
        numpy_helper.from_array(
            np.array([1, num_queries, 1, 2], dtype=np.int64), f"{prefix}_grid_shape"
        ),
        numpy_helper.from_array(
            np.array([1, dim, num_queries], dtype=np.int64), f"{prefix}_sampled_shape"
        ),
    ]
    text = f"""
      {prefix}_ref_sig = Sigmoid({ref_unact})
      {prefix}_xy = Slice({prefix}_ref_sig, c0, c2, ax2)
      {prefix}_xy2 = Mul({prefix}_xy, two_const)
      {prefix}_grid_unit = Sub({prefix}_xy2, one_const)
      {prefix}_grid4d = Reshape({prefix}_grid_unit, {prefix}_grid_shape)
      {prefix}_sampled = GridSample<mode="bilinear",padding_mode="zeros",align_corners=0>({memory_map}, {prefix}_grid4d)
      {prefix}_sampled_r = Reshape({prefix}_sampled, {prefix}_sampled_shape)
      {prefix}_sampled_t = Transpose<perm=[0,2,1]>({prefix}_sampled_r)
      {prefix}_res = Add({x}, {prefix}_sampled_t)
      {prefix}_out = LayerNormalization<axis=-1>({prefix}_res, {prefix}_ln_scale, {prefix}_ln_bias)
    """
    return text, f"{prefix}_out", inits


def _build_rtdetrv4_like_model():
    rng = np.random.RandomState(0)
    inits = []

    # HGNetV2-style stem: two Conv+BatchNormalization+Relu stages, stride 2
    # each, taking the 32x32 input down to an 8x8 feature map.
    inits.append(_f32(rng.randn(8, 3, 3, 3) * 0.2, "W0"))
    inits += [
        _f32(rng.uniform(0.8, 1.2, 8), "bn0_scale"),
        _f32(rng.uniform(-0.1, 0.1, 8), "bn0_bias"),
        _f32(np.zeros(8), "bn0_mean"),
        _f32(np.ones(8), "bn0_var"),
    ]
    inits.append(_f32(rng.randn(16, 8, 3, 3) * 0.2, "W1"))
    inits += [
        _f32(rng.uniform(0.8, 1.2, 16), "bn1_scale"),
        _f32(rng.uniform(-0.1, 0.1, 16), "bn1_bias"),
        _f32(np.zeros(16), "bn1_mean"),
        _f32(np.ones(16), "bn1_var"),
    ]
    backbone_text = """
      c0n = Conv<kernel_shape=[3,3],strides=[2,2],pads=[1,1,1,1]>(images, W0)
      b0n = BatchNormalization<epsilon=1e-05>(c0n, bn0_scale, bn0_bias, bn0_mean, bn0_var)
      r0n = Relu(b0n)
      c1n = Conv<kernel_shape=[3,3],strides=[2,2],pads=[1,1,1,1]>(r0n, W1)
      b1n = BatchNormalization<epsilon=1e-05>(c1n, bn1_scale, bn1_bias, bn1_mean, bn1_var)
      feat_map = Relu(b1n)
    """

    # Constants shared across the position-embedding, anchor-generation and
    # postprocessor blocks below.
    # onnxsim's constant folder requires the canonical `value` tensor
    # attribute on Constant nodes (not the value_int/value_float/value_ints
    # scalar/1-D shorthand attributes -- onnxruntime normalizes those, but
    # onnxsim's own IR does not), so every Constant below spells it out.
    consts_text = """
      r0 = Constant<value = int64 {0}>()
      r1 = Constant<value = int64 {1}>()
      r4 = Constant<value = int64 {4}>()
      r8 = Constant<value = int64 {8}>()

      half_const = Constant<value = float {0.5}>()
      one_const = Constant<value = float {1.0}>()
      two_const = Constant<value = float {2.0}>()
      four_f = Constant<value = float {4.0}>()
      eight_f = Constant<value = float {8.0}>()
      temp_const = Constant<value = float {10000.0}>()
      attn_scale = Constant<value = float {0.25}>()

      c0 = Constant<value = int64[1] {0}>()
      c1 = Constant<value = int64[1] {1}>()
      c2 = Constant<value = int64[1] {2}>()
      c3 = Constant<value = int64[1] {3}>()
      c4 = Constant<value = int64[1] {4}>()
      c8 = Constant<value = int64[1] {8}>()
      cneg1 = Constant<value = int64[1] {-1}>()
      ax1 = Constant<value = int64[1] {1}>()
      ax2 = Constant<value = int64[1] {2}>()
      k5 = Constant<value = int64[1] {5}>()

      repeats_8_1 = Constant<value = int64[2] {8,1}>()
      repeats_1_8 = Constant<value = int64[2] {1,8}>()
      shape_64 = Constant<value = int64[1] {64}>()
      shape_64_1 = Constant<value = int64[2] {64,1}>()
      shape_1_4 = Constant<value = int64[2] {1,4}>()
      shape_1_16_64 = Constant<value = int64[3] {1,16,64}>()
      shape_1_16_8_8 = Constant<value = int64[4] {1,16,8,8}>()
      shape_1_32 = Constant<value = int64[2] {1,32}>()
      shape_8_4 = Constant<value = int64[2] {8,4}>()
      shape_5 = Constant<value = int64[1] {5}>()
    """

    # AIFI: flatten the backbone's last feature map into tokens.
    tokens_text = """
      feat_flat = Reshape(feat_map, shape_1_16_64)
      tokens = Transpose<perm=[0,2,1]>(feat_flat)
    """

    # 2D sin/cos position embedding: a pure function of H, W and C, built
    # from Range/Tile/Sin/Cos exactly the way RT-DETR's AIFI computes it at
    # export time -- onnxsim must constant-fold this whole block away.
    pos_embed_text = """
      base_i = Range(r0, r8, r1)
      base_f = Cast<to=1>(base_i)

      grid_col_row = Unsqueeze(base_f, c0)
      grid_col_tiled = Tile(grid_col_row, repeats_8_1)
      gx_flat = Reshape(grid_col_tiled, shape_64)

      grid_row_col = Unsqueeze(base_f, c1)
      grid_row_tiled = Tile(grid_row_col, repeats_1_8)
      gy_flat = Reshape(grid_row_tiled, shape_64)

      idx4_i = Range(r0, r4, r1)
      idx4_f = Cast<to=1>(idx4_i)
      idx4_norm = Div(idx4_f, four_f)
      pow_out = Pow(temp_const, idx4_norm)
      omega = Reciprocal(pow_out)

      gx_col = Reshape(gx_flat, shape_64_1)
      gy_col = Reshape(gy_flat, shape_64_1)
      omega_row = Reshape(omega, shape_1_4)
      out_w = Mul(gx_col, omega_row)
      out_h = Mul(gy_col, omega_row)
      sin_w = Sin(out_w)
      cos_w = Cos(out_w)
      sin_h = Sin(out_h)
      cos_h = Cos(out_h)
      pos_flat = Concat<axis=-1>(sin_w, cos_w, sin_h, cos_h)
      pos_embed = Unsqueeze(pos_flat, c0)

      tokens_pe = Add(tokens, pos_embed)
    """

    enc_sa_text, enc_sa_out, enc_sa_inits = _self_attn_sublayer(
        rng, "enc_sa", "tokens_pe", C
    )
    enc_ffn_text, enc_ffn_out, enc_ffn_inits = _ffn_sublayer(
        rng, "enc_ffn", enc_sa_out, C, FFN
    )
    inits += enc_sa_inits + enc_ffn_inits

    memory_text = f"""
      enc_t = Transpose<perm=[0,2,1]>({enc_ffn_out})
      memory_map = Reshape(enc_t, shape_1_16_8_8)
    """

    # Anchor / reference-point generation: also a pure function of H and W
    # (Range/Tile/Log inverse-sigmoid), so it must collapse to one Constant
    # too, just like RT-DETR/D-FINE's own `_generate_anchors`.
    anchors_text = """
      gx_shift = Add(gx_flat, half_const)
      gy_shift = Add(gy_flat, half_const)
      cx = Div(gx_shift, eight_f)
      cy = Div(gy_shift, eight_f)
      cx2 = Unsqueeze(cx, cneg1)
      cy2 = Unsqueeze(cy, cneg1)
      wh = ConstantOfShape<value = float[1] {0.125}>(shape_64_1)
      anchors4 = Concat<axis=-1>(cx2, cy2, wh, wh)
      anchors_b = Unsqueeze(anchors4, c0)
      one_minus = Sub(one_const, anchors_b)
      ratio = Div(anchors_b, one_minus)
      anchors_logit = Log(ratio)
      ref_unact = Slice(anchors_logit, c0, c8, ax1)
    """

    # Decoder: learnable query embeddings, self-attention, one deformable
    # cross-attention layer (GridSample), then feed-forward.
    inits.append(_f32(rng.randn(1, NUM_QUERIES, C) * 0.1, "tgt"))

    dec_sa_text, dec_sa_out, dec_sa_inits = _self_attn_sublayer(rng, "dec_sa", "tgt", C)
    dec_ca_text, dec_ca_out, dec_ca_inits = _cross_attn_sublayer(
        "dec_ca", dec_sa_out, "ref_unact", "memory_map", C, NUM_QUERIES
    )
    dec_ffn_text, dec_ffn_out, dec_ffn_inits = _ffn_sublayer(
        rng, "dec_ffn", dec_ca_out, C, FFN
    )
    inits += dec_sa_inits + dec_ca_inits + dec_ffn_inits

    # Class/bbox heads and the NMS-free postprocessor: decode boxes to
    # xyxy, scale by the *runtime* `orig_target_sizes` input (must NOT be
    # folded away), then pick the top-`NUM_OUT` (query, class) pairs by
    # score -- the same flattened-TopK + divmod decode RT-DETR/D-FINE use
    # instead of NMS.
    inits += _linear(rng, C, NUM_CLASSES, "cls_W", "cls_b")
    inits += _linear(rng, C, FFN, "bbox_W1", "bbox_b1")
    inits += _linear(rng, FFN, NUM_CLASSES, "bbox_W2", "bbox_b2")

    postproc_text = f"""
      cls_logits = MatMul({dec_ffn_out}, cls_W)
      cls_logits_b = Add(cls_logits, cls_b)

      bbox_h = MatMul({dec_ffn_out}, bbox_W1)
      bbox_hb = Add(bbox_h, bbox_b1)
      bbox_act = Relu(bbox_hb)
      bbox_delta = MatMul(bbox_act, bbox_W2)
      bbox_delta_b = Add(bbox_delta, bbox_b2)
      new_ref_unact = Add(ref_unact, bbox_delta_b)
      boxes_cxcywh = Sigmoid(new_ref_unact)

      bx_cx = Slice(boxes_cxcywh, c0, c1, ax2)
      bx_cy = Slice(boxes_cxcywh, c1, c2, ax2)
      bx_w = Slice(boxes_cxcywh, c2, c3, ax2)
      bx_h = Slice(boxes_cxcywh, c3, c4, ax2)
      halfw = Mul(bx_w, half_const)
      halfh = Mul(bx_h, half_const)
      bx_x1 = Sub(bx_cx, halfw)
      bx_y1 = Sub(bx_cy, halfh)
      bx_x2 = Add(bx_cx, halfw)
      bx_y2 = Add(bx_cy, halfh)
      boxes_xyxy = Concat<axis=-1>(bx_x1, bx_y1, bx_x2, bx_y2)

      sizes_f = Cast<to=1>(orig_target_sizes)
      ow = Slice(sizes_f, c0, c1, ax1)
      oh = Slice(sizes_f, c1, c2, ax1)
      scale4 = Concat<axis=-1>(ow, oh, ow, oh)
      scale4u = Unsqueeze(scale4, c1)
      boxes_scaled = Mul(boxes_xyxy, scale4u)

      scores_all = Sigmoid(cls_logits_b)
      scores_flat = Reshape(scores_all, shape_1_32)
      scores, topk_idx = TopK<axis=-1,largest=1,sorted=1>(scores_flat, k5)
      labels = Mod(topk_idx, c4)
      box_idx = Div(topk_idx, c4)

      boxes_sq = Reshape(boxes_scaled, shape_8_4)
      box_idx_sq = Reshape(box_idx, shape_5)
      gathered = Gather<axis=0>(boxes_sq, box_idx_sq)
      boxes = Unsqueeze(gathered, c0)
    """

    body = f"""
    rtdetrv4_like (float[1,3,32,32] images, int64[1,2] orig_target_sizes)
      => (float[1,{NUM_OUT},4] boxes, int64[1,{NUM_OUT}] labels, float[1,{NUM_OUT}] scores)
    {{
      {consts_text}
      {backbone_text}
      {tokens_text}
      {pos_embed_text}
      {enc_sa_text}
      {enc_ffn_text}
      {memory_text}
      {anchors_text}
      {dec_sa_text}
      {dec_ca_text}
      {dec_ffn_text}
      {postproc_text}
    }}
    """

    model = _model(body, inits)
    onnx.checker.check_model(model)
    return model


def test_rtdetrv4_export_simplify():
    model = _build_rtdetrv4_like_model()
    nodes_before = len(model.graph.node)

    rng = np.random.RandomState(1)
    images = rng.randn(1, 3, 32, 32).astype(np.float32)
    orig_target_sizes = np.array([[320, 240]], dtype=np.int64)

    # Mirrors RT-DETRv4's own export_onnx.py: real input_data, check_n>0.
    opt, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        input_data={"images": images, "orig_target_sizes": orig_target_sizes},
    )
    assert check_ok

    nodes_after = len(opt.graph.node)
    assert nodes_after < nodes_before, f"{nodes_before} -> {nodes_after}"

    op_types = {node.op_type for node in opt.graph.node}
    # The AIFI position-embedding and anchor-generation blocks are pure
    # functions of the static feature-map size, so every op used only to
    # build them must be folded away entirely.
    for folded_op in ("Range", "Tile", "Sin", "Cos", "Reciprocal", "ConstantOfShape"):
        assert folded_op not in op_types, folded_op

    # The ops that make RT-DETRv4 RT-DETRv4 -- and the ones a future onnxsim
    # regression could plausibly corrupt or drop -- must still be present.
    assert "LayerNormalization" in op_types
    assert "GridSample" in op_types
    assert "TopK" in op_types
    assert {"Mod", "Div", "Gather"} <= op_types

    session = onnxruntime.InferenceSession(opt.SerializeToString())
    assert [o.name for o in session.get_outputs()] == ["boxes", "labels", "scores"]

    boxes, labels, scores = session.run(
        None, {"images": images, "orig_target_sizes": orig_target_sizes}
    )
    assert boxes.shape == (1, NUM_OUT, 4)
    assert labels.shape == (1, NUM_OUT)
    assert scores.shape == (1, NUM_OUT)
    assert labels.dtype == np.int64
    assert (labels >= 0).all() and (labels < NUM_CLASSES).all()

    # Boxes must scale with `orig_target_sizes`: it is a real (dynamic)
    # graph input, not something the simplifier is allowed to bake in.
    doubled_sizes = orig_target_sizes * 2
    boxes_doubled, _, _ = session.run(
        None, {"images": images, "orig_target_sizes": doubled_sizes}
    )
    np.testing.assert_allclose(boxes_doubled, boxes * 2, rtol=1e-4, atol=1e-4)
