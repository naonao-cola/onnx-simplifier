"""SmolLM2-135M (Llama) -> two static ONNX graphs with an explicit KV cache, fp32 references.

  python export_smollm.py --work W [--ngen 32]

A plain-PyTorch Llama (checked against transformers' LlamaForCausalLM) exported as

- dec_prefill.fp32.onnx: input_ids int32 [1,P], last_idx int32 [1]
                         -> logits [1,V] (at last_idx), k/v [L,Hkv,P,D]
- dec_step.fp32.onnx:    input_ids int32 [1,1], pos int32 [1], k_cache/v_cache [L,Hkv,T,D]
                         -> logits [1,V], k_new/v_new [L,Hkv,1,D]

Every tensor has rank <= 4 (GQA repeats K/V by reshape/expand, heads live in the batch dim). The
decode step attends over the cache (positions < pos, the host writes row `pos` afterwards) plus the
new token, so the graph never scatters into the cache. RMSNorm pre-scales x by 1/32 (exact, with
eps/1024) so x^2 can't overflow fp16 on the HTP.

Also writes W/dec_in/prompt_<i>.bin and force_<i>.bin (the fp32 greedy tokens, for teacher-forced
runs) for models.PROMPTS, and W/dec_ref/gen_<i>.npy + logits_<i>.npy (fp32 greedy decoding).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import models
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn


class Llama(nn.Module):
    def __init__(self, cfg: dict, sd: dict):
        super().__init__()
        self.L = cfg["num_hidden_layers"]
        self.H = cfg["num_attention_heads"]
        self.Hkv = cfg["num_key_value_heads"]
        self.C = cfg["hidden_size"]
        self.D = self.C // self.H
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        g = lambda k: nn.Parameter(sd[k].float(), requires_grad=False)  # noqa: E731
        self.embed = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.layers = nn.ModuleList()
        for i in range(self.L):
            p = f"model.layers.{i}."
            m = nn.Module()
            for n, k in [
                ("ln1", "input_layernorm.weight"),
                ("ln2", "post_attention_layernorm.weight"),
                ("wq", "self_attn.q_proj.weight"),
                ("wk", "self_attn.k_proj.weight"),
                ("wv", "self_attn.v_proj.weight"),
                ("wo", "self_attn.o_proj.weight"),
                ("wg", "mlp.gate_proj.weight"),
                ("wu", "mlp.up_proj.weight"),
                ("wd", "mlp.down_proj.weight"),
            ]:
                setattr(m, n, g(p + k))
            self.layers.append(m)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.D, 2).float() / self.D))
        ang = torch.arange(models.MAXLEN).float()[:, None] * inv[None]
        ang = torch.cat([ang, ang], -1)  # [T, D], non-interleaved (rotate_half)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def rms(self, x, w):
        x = x * (1.0 / 32.0)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps / 1024.0) * w

    def rope(self, x, cos, sin):  # x [h, S, D], cos/sin [S, D]
        x1, x2 = x[..., : self.D // 2], x[..., self.D // 2 :]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def heads(self, x, n):  # [S, n*D] -> [n, S, D]
        return x.reshape(x.shape[0], n, self.D).transpose(0, 1)

    def repeat_kv(self, x):  # [Hkv, S, D] -> [H, S, D]
        r = self.H // self.Hkv
        S = x.shape[1]
        return x[:, None].expand(self.Hkv, r, S, self.D).reshape(self.H, S, self.D)

    def mlp(self, m, x):
        return F.linear(F.silu(F.linear(x, m.wg)) * F.linear(x, m.wu), m.wd)

    def logits(self, h):
        return F.linear(self.rms(h, self.norm), self.embed)

    def prefill(self, input_ids, last_idx):
        S = input_ids.shape[1]
        x = self.embed[input_ids[0].long()]  # [S, C]
        cos, sin = self.cos[:S], self.sin[:S]
        causal = torch.triu(torch.full((S, S), -1e4, dtype=x.dtype), 1)
        ks, vs = [], []
        for m in self.layers:
            h = self.rms(x, m.ln1)
            q = self.rope(self.heads(F.linear(h, m.wq), self.H), cos, sin)
            k = self.rope(self.heads(F.linear(h, m.wk), self.Hkv), cos, sin)
            v = self.heads(F.linear(h, m.wv), self.Hkv)
            ks.append(k[None])
            vs.append(v[None])
            a = (q @ self.repeat_kv(k).transpose(1, 2)) * (self.D**-0.5) + causal
            o = (a.softmax(-1) @ self.repeat_kv(v)).transpose(0, 1).reshape(S, self.C)
            x = x + F.linear(o, m.wo)
            x = x + self.mlp(m, self.rms(x, m.ln2))
        hl = x.index_select(0, last_idx.long())  # [1, C]
        return self.logits(hl), torch.cat(ks), torch.cat(vs)

    def step(self, input_ids, pos, k_cache, v_cache):
        T = k_cache.shape[2]
        x = self.embed[input_ids[0].long()]  # [1, C]
        p = pos.long()
        cos, sin = self.cos.index_select(0, p), self.sin.index_select(0, p)
        # cache rows < pos are valid; the new token is attended separately (no scatter)
        # float compare: int64 compares/arange don't run on the HTP
        valid = torch.arange(T).to(x.dtype)[None] < pos.to(x.dtype)[:, None]
        mask = torch.where(
            valid, torch.zeros((), dtype=x.dtype), torch.full((), -1e4, dtype=x.dtype)
        )  # [1, T]
        ks, vs = [], []
        for li, m in enumerate(self.layers):
            h = self.rms(x, m.ln1)
            q = self.rope(self.heads(F.linear(h, m.wq), self.H), cos, sin)  # [H,1,D]
            k = self.rope(
                self.heads(F.linear(h, m.wk), self.Hkv), cos, sin
            )  # [Hkv,1,D]
            v = self.heads(F.linear(h, m.wv), self.Hkv)
            ks.append(k[None])
            vs.append(v[None])
            kc, vc = self.repeat_kv(k_cache[li]), self.repeat_kv(v_cache[li])  # [H,T,D]
            sc = (q @ kc.transpose(1, 2)) * (self.D**-0.5) + mask  # [H,1,T]
            sn = (q * self.repeat_kv(k)).sum(-1, keepdim=True) * (
                self.D**-0.5
            )  # [H,1,1]
            w = torch.cat([sc, sn], -1).softmax(-1)
            o = w[..., :T] @ vc + w[..., T:] * self.repeat_kv(v)  # [H,1,D]
            x = x + F.linear(o.transpose(0, 1).reshape(1, self.C), m.wo)
            x = x + self.mlp(m, self.rms(x, m.ln2))
        return self.logits(x), torch.cat(ks), torch.cat(vs)


class Prefill(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, last_idx):
        return self.m.prefill(input_ids, last_idx)


class Step(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, pos, k_cache, v_cache):
        return self.m.step(input_ids, pos, k_cache, v_cache)


def int32_indices(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop the exporter's Cast(int32 -> int64) where every consumer is a Gather index: ONNX
    Gather takes int32 indices, and int64 tensors don't run on the HTP."""
    g = model.graph
    vi = onnx.shape_inference.infer_shapes(model).graph
    et = {
        v.name: v.type.tensor_type.elem_type
        for v in list(vi.value_info) + list(vi.input)
    }
    for n in list(g.node):
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        if (
            n.op_type != "Cast"
            or to != onnx.TensorProto.INT64
            or et.get(n.input[0]) != onnx.TensorProto.INT32
        ):
            continue
        users = [m for m in g.node if n.output[0] in m.input]
        if users and all(
            u.op_type == "Gather" and list(u.input).index(n.output[0]) == 1
            for u in users
        ):
            for u in users:
                u.input[1] = n.input[0]
            g.node.remove(n)
    return model


def greedy(m: Llama, prompt: list, ngen: int, keep: int = 9):
    P = models.PREFILL
    ids = torch.zeros(1, P, dtype=torch.int32)
    ids[0, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
    lg, k, v = m.prefill(ids, torch.tensor([len(prompt) - 1], dtype=torch.int32))
    kc = torch.zeros(m.L, m.Hkv, models.MAXLEN, m.D)
    vc = torch.zeros_like(kc)
    n = len(prompt)
    kc[:, :, :n], vc[:, :, :n] = k[:, :, :n], v[:, :, :n]
    gen, logits = [int(lg.argmax())], [lg[0].numpy()]
    for s in range(1, ngen):
        pos = n + s - 1
        lg, kn, vn = m.step(
            torch.tensor([[gen[-1]]], dtype=torch.int32),
            torch.tensor([pos], dtype=torch.int32),
            kc,
            vc,
        )
        kc[:, :, pos], vc[:, :, pos] = kn[:, :, 0], vn[:, :, 0]
        gen.append(int(lg.argmax()))
        if len(logits) < keep:
            logits.append(lg[0].numpy())
    return gen, np.stack(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ngen", type=int, default=32)
    ap.add_argument(
        "--fp16-only",
        action="store_true",
        help="only export dec_*.fp16.onnx: fp16 weights, activations and graph I/O (KV cache, logits)",
    )
    a = ap.parse_args()
    work = Path(a.work)
    for d in ("dec_in", "dec_ref"):
        (work / d).mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = models.fetch(models.SMOLLM)
    cfg = json.loads((snap / "config.json").read_text())
    m = Llama(cfg, load_file(snap / "model.safetensors")).eval()
    tok = AutoTokenizer.from_pretrained(snap)

    torch.set_grad_enabled(False)
    if a.fp16_only:
        export(m.half(), work, "fp16", torch.float16)
        return
    # check the plain model against transformers on one prompt (prefill logits at every position)
    hf = AutoModelForCausalLM.from_pretrained(
        snap, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    p0 = tok(models.PROMPTS[0]).input_ids
    ref = hf(torch.tensor([p0])).logits[0, -1]
    ids = torch.zeros(1, models.PREFILL, dtype=torch.int32)
    ids[0, : len(p0)] = torch.tensor(p0)
    mine = m.prefill(ids, torch.tensor([len(p0) - 1], dtype=torch.int32))[0][0]
    print(
        f"plain Llama vs transformers, prefill logits max abs diff {(mine - ref).abs().max():.2e}"
    )
    hf_gen = hf.generate(torch.tensor([p0]), max_new_tokens=a.ngen, do_sample=False)[
        0, len(p0) :
    ].tolist()
    del hf

    for i, prompt in enumerate(models.PROMPTS):
        t = tok(prompt).input_ids
        gen, lg = greedy(m, t, a.ngen)
        if i == 0:
            same = sum(x == y for x, y in zip(gen, hf_gen))
            print(f"greedy vs transformers.generate: {same}/{a.ngen} tokens identical")
        np.array(t, np.int32).tofile(work / "dec_in" / f"prompt_{i}.bin")
        np.array(gen[:-1], np.int32).tofile(work / "dec_in" / f"force_{i}.bin")
        np.save(work / "dec_ref" / f"gen_{i}.npy", np.array(gen, np.int32))
        np.save(work / "dec_ref" / f"logits_{i}.npy", lg)
        print(f"prompt {i}: {prompt!r} -> {tok.decode(gen)!r}")

    export(m, work, "fp32", torch.float32)


def export(m: Llama, work: Path, tag: str, dt) -> None:
    P, T = models.PREFILL, models.MAXLEN
    ids = torch.zeros(1, P, dtype=torch.int32)
    torch.onnx.export(
        Prefill(m),
        (ids, torch.tensor([3], dtype=torch.int32)),
        work / f"dec_prefill.{tag}.onnx",
        input_names=["input_ids", "last_idx"],
        output_names=["logits", "k", "v"],
        opset_version=17,
        dynamo=False,
    )
    kc = torch.zeros(m.L, m.Hkv, T, m.D, dtype=dt)
    torch.onnx.export(
        Step(m),
        (
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            kc,
            kc.clone(),
        ),
        work / f"dec_step.{tag}.onnx",
        input_names=["input_ids", "pos", "k_cache", "v_cache"],
        output_names=["logits", "k_new", "v_new"],
        opset_version=17,
        dynamo=False,
    )
    import onnxsim

    for f in (f"dec_prefill.{tag}.onnx", f"dec_step.{tag}.onnx"):
        g, ok = onnxsim.simplify(onnx.load(work / f))
        assert ok
        g = int32_indices(g)
        onnx.save(g, work / f)
        print(f"{f}: {len(g.graph.node)} nodes")


if __name__ == "__main__":
    main()
