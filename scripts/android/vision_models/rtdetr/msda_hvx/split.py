#!/usr/bin/env python3
"""RT-DETR split around the HVX MSDA kernel (../../../msda_hvx/): 4 HTP pieces and 3 kernel calls.

    pre  -> [msda0] -> mid0 -> [msda1] -> mid1 -> [msda2] -> post

  pre   image -> backbone, hybrid encoder, query selection; the 3 decoder layers' value maps
        value{0,1,2} (8400, 256) = value_proj_i(memory); decoder layer 0 up to its cross-attention:
        h0 (300, 256) (self-attn + norm), off0 (300, 192) raw sampling offsets, w0 (300, 96)
        softmaxed attention weights, ref0 (300, 4) sigmoid reference boxes
  mid_i  msda_i (300, 256) (kernel output) + h_i + ref_i -> rest of layer i (output_proj, residual,
        norm, FFN, box refinement) -> ref_{i+1} and layer i+1 up to its cross-attention
  post  msda2 + h2 + ref2 -> logits (300, 80), boxes (300, 4)

The kernel runs mmcv's contract in mode MSDA_REF_BOX (loc = ref.xy + off / P * ref.wh * 0.5), which
is RT-DETR's own formula, so the grid math never runs on the HTP.

usage: split.py check [--n N]            chained pieces + msda_reference vs HF, on the eval images
       split.py export --work <dir>      pieces -> <work>/split/{pre,mid0,mid1,post}.sim.onnx
       split.py quant --work <dir> [--policy front8]  pre in int8 (../quantize.py regions), uint8 NHWC input
       split.py dump --work <dir>        per-image phone inputs + fp32 refs -> <work>/split/img*/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "msda_hvx"))

import common as C  # noqa: E402
import model as M  # noqa: E402
from msda_ref import msda_reference  # noqa: E402

Q, CH, H, L, P = 300, 256, 8, 3, 4
LEVELS = M.SHAPES


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class Parts:
    def __init__(self, m):
        self.inner = m.model
        self.dec = self.inner.decoder

    def front(self, pixel_values):
        """HF RTDetrModel.forward up to the decoder (inference path)."""
        inner = self.inner
        features = inner.backbone(
            pixel_values, torch.ones(pixel_values.shape[0], *pixel_values.shape[2:])
        )
        proj = [
            inner.encoder_input_proj[lv](src) for lv, (src, _) in enumerate(features)
        ]
        enc = inner.encoder(proj)
        sources = [
            inner.decoder_input_proj[lv](s)
            for lv, s in enumerate(enc.last_hidden_state)
        ]
        src = torch.cat(
            [s.flatten(2).transpose(1, 2) for s in sources], 1
        )  # (1, 8400, 256)
        anchors, valid = inner.generate_anchors(tuple(LEVELS), dtype=src.dtype)
        memory = valid.to(src.dtype) * src
        om = inner.enc_output(memory)
        cls = inner.enc_score_head(om)
        coord = inner.enc_bbox_head(om) + anchors
        _, idx = torch.topk(cls.max(-1).values, Q, dim=1)
        ref_unact = coord.gather(1, idx.unsqueeze(-1).repeat(1, 1, 4))
        target = om.gather(1, idx.unsqueeze(-1).repeat(1, 1, CH))
        return src, target, torch.sigmoid(ref_unact)

    def head(self, i, hidden, ref):
        """Decoder layer i up to its cross-attention."""
        lay = self.dec.layers[i]
        pos = self.dec.query_pos_head(ref)
        res = hidden
        h, _ = lay.self_attn(hidden_states=hidden, position_embeddings=pos)
        h = lay.self_attn_layer_norm(res + h)
        x = h + pos
        att = lay.encoder_attn
        off = att.sampling_offsets(
            x
        )  # (1, Q, H*L*P*2), (H, L, P, 2) order = the kernel's
        w = F.softmax(att.attention_weights(x).view(1, Q, H, L * P), -1).reshape(
            1, Q, H * L * P
        )
        return h, off, w

    def tail(self, i, msda_out, h, ref):
        """Rest of decoder layer i after its MSDA; returns the new hidden, new ref, logits."""
        lay = self.dec.layers[i]
        x = lay.encoder_attn_layer_norm(h + lay.encoder_attn.output_proj(msda_out))
        x = lay.final_layer_norm(x + lay.mlp(x))
        new_ref = torch.sigmoid(self.dec.bbox_embed[i](x) + inverse_sigmoid(ref))
        return x, new_ref, self.dec.class_embed[i](x)

    def value(self, i, src):
        return self.dec.layers[i].encoder_attn.value_proj(src)


class Pre(torch.nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.p = parts
        self.inner = (
            parts.inner
        )  # registers the weights as parameters (ONNX initializers)

    def forward(self, pixel_values):
        src, hidden, ref = self.p.front(pixel_values)
        vals = [self.p.value(i, src)[0] for i in range(3)]
        h, off, w = self.p.head(0, hidden, ref)
        return (*vals, h[0], off[0], w[0], ref[0])


class Mid(torch.nn.Module):
    def __init__(self, parts, i):
        super().__init__()
        self.p, self.i = parts, i
        self.inner = parts.inner

    def forward(self, msda, h, ref):
        x, ref2, _ = self.p.tail(self.i, msda[None], h[None], ref[None])
        h2, off, w = self.p.head(self.i + 1, x, ref2)
        return h2[0], off[0], w[0], ref2[0]


class Post(torch.nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.p = parts
        self.inner = parts.inner

    def forward(self, msda, h, ref):
        _, ref2, logits = self.p.tail(2, msda[None], h[None], ref[None])
        return logits[0], ref2[0]


PIECES = {
    "pre": (["pixel_values"], ["value0", "value1", "value2", "h", "off", "w", "ref"]),
    "mid0": (["msda", "h", "ref"], ["h_out", "off", "w", "ref_out"]),
    "mid1": (["msda", "h", "ref"], ["h_out", "off", "w", "ref_out"]),
    "post": (["msda", "h", "ref"], ["logits", "boxes"]),
}


def kernel(value, off, w, ref):
    """The HVX kernel's contract (msda_ref.msda_reference), mode box."""
    return msda_reference(
        value[None],
        LEVELS,
        off.reshape(Q, H, 1, L, P, 2),
        w.reshape(Q, H, 1, L, P),
        mode="box",
        ref=ref.reshape(1, Q, 1, 1, 4),
    )


def chain(mods, x, msda=kernel):
    v0, v1, v2, h, off, w, ref = mods["pre"](x)
    vals = [v0, v1, v2]
    for i in range(2):
        h, off, w, ref = mods[f"mid{i}"](msda(vals[i], off, w, ref), h, ref)
    return mods["post"](msda(vals[2], off, w, ref), h, ref)


def build():
    m = M.load(patched=True)
    m.requires_grad_(False)
    parts = Parts(m)
    mods = {
        "pre": Pre(parts).eval(),
        "mid0": Mid(parts, 0).eval(),
        "mid1": Mid(parts, 1).eval(),
        "post": Post(parts).eval(),
    }
    return m, mods


def check(n):
    torch.set_grad_enabled(False)
    ref_m = M.load(patched=False)
    _, mods = build()
    worst, tot = 0.0, {"ref": 0, "det": 0, "matched": 0}
    for p in C.image_paths("eval")[:n]:
        x = torch.from_numpy(C.to_pixels(C.load_rgb_u8(p)))
        a = ref_m(pixel_values=x)
        lg, bx = chain(mods, x)
        worst = max(
            worst,
            float((a.logits[0] - lg).abs().max()),
            float((a.pred_boxes[0] - bx).abs().max()),
        )
        r = C.match(
            (a.logits.numpy(), a.pred_boxes.numpy()),
            (lg[None].numpy(), bx[None].numpy()),
        )
        for k in tot:
            tot[k] += r[k]
    print(
        f"split chain + kernel contract vs HF over {n} images: max abs {worst:.2e}, "
        f"matched {tot['matched']}/{tot['ref']} (det {tot['det']})"
    )


def sample_inputs(mods, x):
    """Real inputs of every piece for image x (fp32 torch)."""
    out = {"pre": (x,)}
    v0, v1, v2, h, off, w, ref = mods["pre"](x)
    vals = [v0, v1, v2]
    for i in range(2):
        mo = kernel(vals[i], off, w, ref)
        out[f"mid{i}"] = (mo, h, ref)
        h, off, w, ref = mods[f"mid{i}"](mo, h, ref)
    out["post"] = (kernel(vals[2], off, w, ref), h, ref)
    return out


def export(work: Path):
    import onnx
    import onnxruntime as ort

    import onnxsim

    torch.set_grad_enabled(False)
    _, mods = build()
    d = work / "split"
    d.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(C.to_pixels(C.load_rgb_u8(C.image_paths("eval")[0])))
    ins = sample_inputs(mods, x)
    for name, (in_names, out_names) in PIECES.items():
        raw = d / f"{name}.onnx"
        torch.onnx.export(
            mods[name],
            ins[name],
            str(raw),
            input_names=in_names,
            output_names=out_names,
            opset_version=17,
            dynamo=False,
        )
        sim, ok = onnxsim.simplify(onnx.load(str(raw)))
        assert ok
        onnx.save(sim, str(d / f"{name}.sim.onnx"))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        got = ort.InferenceSession(
            str(d / f"{name}.sim.onnx"), so, providers=["CPUExecutionProvider"]
        ).run(None, {k: v.numpy() for k, v in zip(in_names, ins[name])})
        ref = mods[name](*ins[name])
        md = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
        print(f"{name}: {len(sim.graph.node)} nodes, ORT vs torch max abs {md:.2e}")


def quant(work: Path, policy: str, u8_values: bool = False):
    import onnx
    import quantize as QZ

    from onnxsim import full_qdq as FQ

    d = work / "split"
    m = onnx.load(str(d / "pre.sim.onnx"))
    kw = QZ.quant_kwargs(m, policy)
    data = [
        {"pixel_values": C.to_pixels(C.load_rgb_u8(p))}
        for p in C.image_paths("calibration")[:32]
    ]
    q = FQ.quantize_full_qdq(m, data, ranges={"pixel_values": (0.0, 1.0)}, **kw)
    q, _ = FQ.quantized_io(
        q, inputs=["pixel_values"], outputs=[], nhwc_inputs=["pixel_values"]
    )
    q = QZ.transpose_as_qdq_unit(q, "pixel_values")
    if u8_values:
        q = u8_value_outputs(q, m, data)
    name = f"pre.{policy}{'.v8' if u8_values else ''}.onnx"
    onnx.save(q, str(d / name))
    print(f"{name}: {len(q.graph.node)} nodes")


def u8_value_outputs(q, float_pre, data):
    """value{0,1,2} as uint8 graph outputs: a per-tensor QuantizeLinear (min/max over the calibration
    images of the float pre piece) at the end of the fp16 value path, so the HTP writes a quarter of
    the bytes. The (scale, zero point) pairs are stored as `value{i}_scale` / `value{i}_zero_point`
    in the model's metadata_props for the runner (the MSDA kernel's uint8 value input)."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(
        float_pre.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in s.get_outputs()]
    lo, hi = {}, {}
    for feed in data:
        for n, v in zip(names, s.run(None, feed)):
            if n.startswith("value"):
                lo[n] = min(lo.get(n, 0.0), float(v.min()))
                hi[n] = max(hi.get(n, 0.0), float(v.max()))
    g = q.graph
    for o in g.output:
        if o.name not in lo:
            continue
        scale = (hi[o.name] - lo[o.name]) / 255
        zp = int(round(-lo[o.name] / scale))
        for n in g.node:
            for k, t in enumerate(n.output):
                if t == o.name:
                    n.output[k] = o.name + "_f"
        g.initializer.extend(
            [
                numpy_helper.from_array(np.array(scale, np.float32), o.name + "_scale"),
                numpy_helper.from_array(np.array(zp, np.uint8), o.name + "_zero_point"),
            ]
        )
        g.node.append(
            helper.make_node(
                "QuantizeLinear",
                [o.name + "_f", o.name + "_scale", o.name + "_zero_point"],
                [o.name],
                name=o.name + "/q_out",
            )
        )
        o.type.tensor_type.elem_type = TensorProto.UINT8
        for k, v in (("scale", repr(scale)), ("zero_point", str(zp))):
            q.metadata_props.add(key=f"{o.name}_{k}", value=v)
        print(f"  {o.name}: uint8 scale {scale:.6g} zero point {zp}")
    return q


def dump(work: Path, n: int):
    """<work>/split/img{i}/: image.u8 (640,640,3 RGB), pixels.f32, ref logits/boxes (fp32 HF)."""
    torch.set_grad_enabled(False)
    ref_m = M.load(patched=False)
    for i, p in enumerate(C.image_paths("eval")[:n]):
        d = work / "split" / f"img{i}"
        d.mkdir(parents=True, exist_ok=True)
        rgb = C.load_rgb_u8(p)
        rgb.tofile(d / "image.u8")
        px = C.to_pixels(rgb)
        px.tofile(d / "pixels.f32")
        a = ref_m(pixel_values=torch.from_numpy(px))
        np.savez(d / "ref.npz", logits=a.logits.numpy(), boxes=a.pred_boxes.numpy())
    print(f"dumped {n} images to {work / 'split'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "export", "quant", "dump"])
    ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-rtdetr/work"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--policy", default="front8")
    ap.add_argument(
        "--u8-values", action="store_true", help="value maps as uint8 graph outputs"
    )
    a = ap.parse_args()
    work = Path(a.work)
    if a.what == "check":
        check(a.n)
    elif a.what == "export":
        export(work)
    elif a.what == "quant":
        quant(work, a.policy, a.u8_values)
    else:
        dump(work, a.n)


if __name__ == "__main__":
    sys.exit(main())
