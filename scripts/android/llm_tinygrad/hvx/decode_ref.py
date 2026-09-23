#!/usr/bin/env python3
"""SmolLM2-135M one-token decode step exactly as the DSP skel computes it (decode_impl.c), in numpy.

Every Linear is a W8A8 GEMV on the vrmpy layout: int8 weights, symmetric per output channel; the input
activation quantized per token to uint8 (asymmetric, zero point) or split into two uint8 halves ("a16": hi/lo
bytes of a 16-bit code, two vrmpy passes over the same weights). Everything else (RMSNorm, RoPE, attention
over the valid KV positions, SwiGLU, residuals) is fp32. The prompt is fed one token at a time through the
same step (no separate prefill).

  python decode_ref.py --work W --eval MODE [MODE...]   # MODE: fp32 | a8 | a16; accuracy vs dec_ref/
  python decode_ref.py --work W --export MODE --out D   # weight/scale/table blobs for the phone
  python decode_ref.py --work W --phone D               # accuracy of the phone's gen_/logits_ outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402

H, L, NH, NKV, HD, FF, V = 576, 30, 9, 3, 64, 1536, 49152
THETA, EPS, MAXLEN = 100000.0, 1e-5, 256
QKV = H + 2 * NKV * HD  # 960: q | k | v rows of one fused GEMV
NGEN = 32


def load_weights():
    from safetensors.torch import load_file  # bf16 checkpoint: numpy can't read it

    snap = models.fetch(models.SMOLLM)
    w = {
        k: v.float().numpy()
        for k, v in load_file(str(snap / "model.safetensors")).items()
    }
    p = "model.layers.{}."
    layers = []
    for i in range(L):
        g = lambda n: w[p.format(i) + n]  # noqa: E731
        layers.append(
            dict(
                ln1=g("input_layernorm.weight"),
                ln2=g("post_attention_layernorm.weight"),
                # [K, N] (x @ W) with the fused projections concatenated along N
                qkv=np.concatenate(
                    [
                        g("self_attn.q_proj.weight"),
                        g("self_attn.k_proj.weight"),
                        g("self_attn.v_proj.weight"),
                    ]
                ).T,
                o=g("self_attn.o_proj.weight").T,
                gu=np.concatenate(
                    [g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")]
                ).T,
                down=g("mlp.down_proj.weight").T,
            )
        )
    emb = w["model.embed_tokens.weight"]
    return dict(layers=layers, norm=w["model.norm.weight"], emb=emb, head=emb.T.copy())


def qweight(W):
    """[K,N] fp32 -> int8 [K,N], per-column scale [N], per-column sum of the int8 weights [N] (for the zero point)."""
    s = np.maximum(np.abs(W).max(0), 1e-12) / 127.0
    q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return q, s.astype(np.float32), q.astype(np.int32).sum(0)


class Lin:
    def __init__(self, W, mode, q=None):
        self.mode, self.W, self.rec = mode, W, None
        if mode != "fp32":
            self.q, self.s, self.cs = q if q is not None else qweight(W)
            self.qf = self.q.astype(np.float64)

    def __call__(self, x):
        if self.rec is not None:
            self.rec.append(x.astype(np.float32))
        if self.mode == "fp32":
            return x @ self.W
        if self.mode == "w8":  # weight-only: the int8 weights' own error bound
            return ((x.astype(np.float64) @ self.qf) * self.s).astype(np.float32)
        lo, hi = float(min(x.min(), 0.0)), float(max(x.max(), 0.0))
        levels = 255.0 if self.mode == "a8" else 65535.0
        s = max(hi - lo, 1e-12) / levels
        zp = float(np.clip(np.round(-lo / s), 0, levels))
        xq = np.clip(np.round(x / s) + zp, 0, levels)
        # a16: xq = 256*hi + lo, two exact u8 x s8 passes; numerically identical to one wide integer dot
        acc = xq.astype(np.float64) @ self.qf
        return (s * self.s * (acc - zp * self.cs)).astype(np.float32)


def rms(x, g):
    return (x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)).astype(
        np.float32
    ) * g


def rope_tab():
    inv = 1.0 / THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD)
    ang = np.arange(MAXLEN)[:, None] * inv[None]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, c, s):  # x [n, 64], HF rotate_half (non-interleaved)
    a, b = x[:, : HD // 2], x[:, HD // 2 :]
    return np.concatenate([a * c - b * s, b * c + a * s], 1)


class Model:
    def __init__(self, w, mode, gq=None):
        self.w = w
        # "MODE:skip=a,b": those Linears stay fp32 (sensitivity sweeps)
        mode, _, skip = mode.partition(":skip=")
        skip = set(skip.split(",")) if skip else set()
        mode, gptq = (mode[5:], True) if mode.startswith("gptq-") else (mode, False)
        pre = (lambda i, k: gq[f"{i}.{k}"]) if gptq else (lambda i, k: None)
        self.lin = [
            {
                k: Lin(lw[k], "fp32" if k in skip else mode, pre(i, k))
                for k in ("qkv", "o", "gu", "down")
            }
            for i, lw in enumerate(w["layers"])
        ]
        self.head = Lin(w["head"], "fp32" if "head" in skip else mode, pre("h", "head"))
        self.cos, self.sin = rope_tab()

    def reset(self):
        self.kc = np.zeros((L, NKV, MAXLEN, HD), np.float32)
        self.vc = np.zeros_like(self.kc)

    def step(self, tok, pos):
        h = self.w["emb"][tok].copy()
        c, s = self.cos[pos], self.sin[pos]
        for i, lw in enumerate(self.w["layers"]):
            ln = self.lin[i]
            y = ln["qkv"](rms(h, lw["ln1"]))
            q = rope(y[:H].reshape(NH, HD), c, s)
            k = rope(y[H : H + NKV * HD].reshape(NKV, HD), c, s)
            self.kc[i, :, pos], self.vc[i, :, pos] = (
                k,
                y[H + NKV * HD :].reshape(NKV, HD),
            )
            att = np.empty((NH, HD), np.float32)
            for hh in range(NH):
                kv = hh // (NH // NKV)
                sc = self.kc[i, kv, : pos + 1] @ q[hh] / np.sqrt(HD)
                p = np.exp(sc - sc.max())
                att[hh] = (p / p.sum()) @ self.vc[i, kv, : pos + 1]
            h = h + ln["o"](att.reshape(-1))
            gu = ln["gu"](rms(h, lw["ln2"]))
            g, u = gu[:FF], gu[FF:]
            h = h + ln["down"](g / (1 + np.exp(-g)) * u)
        return self.head(rms(h, self.w["norm"]))


def calib_tokens():
    """Calibration text disjoint from the eval prompts: the encoder's 20 sentences, each followed by 24 tokens of the
    fp32 model's own greedy continuation (so decode-time activations, not just sentence starts, are covered)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(models.fetch(models.SMOLLM) / "tokenizer.json"))
    return [tok.encode(t).ids for t in models.SENTENCES]


def gptq_one(W, X, damp=0.01, blk=128):
    """GPTQ (Frantar et al.) for x @ W, W [K,N]; per-output-channel symmetric int8 (the vrmpy kernel's format).
    X [T,K] calibration inputs. Returns (q int8 [K,N], scale [N], colsum [N])."""
    K, N = W.shape
    Wd = W.astype(np.float64).copy()
    s = np.maximum(np.abs(W).max(0), 1e-12).astype(np.float64) / 127.0
    Hm = X.astype(np.float64).T @ X.astype(np.float64)
    dead = np.diag(Hm) == 0
    Hm[dead, dead] = 1
    Wd[dead] = 0
    Hm += damp * np.mean(np.diag(Hm)) * np.eye(K)
    Hinv = np.linalg.cholesky(np.linalg.inv(Hm)).T  # upper Cholesky of H^-1
    Q = np.zeros((K, N))
    for b0 in range(0, K, blk):
        b1 = min(b0 + blk, K)
        Wb, Err = Wd[b0:b1].copy(), np.zeros((b1 - b0, N))
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            q = np.clip(np.round(Wb[j] / s), -127, 127) * s
            e = (Wb[j] - q) / Hb[j, j]
            Wb[j + 1 :] -= np.outer(Hb[j, j + 1 :], e)
            Q[b0 + j], Err[j] = q, e
        Wd[b1:] -= Hinv[b0:b1, b1:].T @ Err
    q = np.clip(np.round(Q / s), -127, 127).astype(np.int8)
    return q, s.astype(np.float32), q.astype(np.int32).sum(0)


def gptq_weights(w, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return {
            k[:-2]: (z[k[:-2] + ".q"], z[k[:-2] + ".s"], z[k[:-2] + ".c"])
            for k in z.files
            if k.endswith(".q")
        }
    m = Model(w, "fp32")
    lins = [
        (f"{i}.{k}", m.lin[i][k]) for i in range(L) for k in ("qkv", "o", "gu", "down")
    ] + [("h.head", m.head)]
    for _, ln in lins:
        ln.rec = []
    for ids in calib_tokens():
        m.reset()
        toks = list(ids)
        for p in range(len(ids) + 24):
            lg = m.step(toks[p], p)
            if p + 1 >= len(toks):
                toks.append(int(lg.argmax()))
    out = {}
    for name, ln in lins:
        out[name] = gptq_one(ln.W, np.stack(ln.rec))
        ln.rec = None
        print(
            f"gptq {name}: {len(out[name][0])}x{out[name][0].shape[1]}", flush=True
        ) if name.endswith("down") and name.startswith("29") else None
    np.savez(
        cache, **{f"{k}.{t}": v[j] for k, v in out.items() for j, t in enumerate("qsc")}
    )
    return out


def run(model, work: Path, forced: bool):
    out = []
    for i in range(len(models.PROMPTS)):
        pr = np.fromfile(work / "dec_in" / f"prompt_{i}.bin", np.int32)
        fo = np.fromfile(work / "dec_in" / f"force_{i}.bin", np.int32)
        model.reset()
        for p, t in enumerate(pr):
            lg = model.step(int(t), p)
        gen, logits = [int(lg.argmax())], [lg]
        for s in range(1, NGEN):
            tok = int(fo[s - 1]) if forced else gen[-1]
            lg = model.step(tok, len(pr) + s - 1)
            gen.append(int(lg.argmax()))
            if len(logits) < 9:
                logits.append(lg)
        out.append((np.array(gen), np.stack(logits)))
    return out


def score(work: Path, free, forced):
    refs = [
        (
            np.load(work / "dec_ref" / f"gen_{i}.npy"),
            np.load(work / "dec_ref" / f"logits_{i}.npy"),
        )
        for i in range(len(models.PROMPTS))
    ]
    same, lead, agree, cmin = 0, [], [], 1.0
    for (g, _), (fg, flg), (rg, rl) in zip(free, forced, refs):
        n = int(np.argmax(g != rg)) if (g != rg).any() else len(rg)
        lead.append(n)
        same += n == len(rg)
        agree.append(float((fg == rg).mean()))
        for a, b in zip(flg, rl):
            cmin = min(cmin, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return f"identical {same}/10, leading tokens {lead}, forced agreement {np.mean(agree):.3f}, worst logits cos {cmin:.5f}"


def pack(q):
    """int8 [K,N] -> the vrmpy layout Wp[N/32][K/4][32][4] (llm_kernels.pack)."""
    K, N = q.shape
    return np.ascontiguousarray(q.reshape(K // 4, 4, N // 32, 32).transpose(2, 0, 3, 1))


def export(w, gq, out: Path, work: Path):
    """decode_impl.c's blobs: W.bin (packed int8, per layer qkv|o|gu|down, then the head), F.bin (fp32: per layer
    ln1, ln2, (scale, colsum) x 4; final norm; head scale, colsum; RoPE cos, sin [256][32]), emb.f16.bin, and the
    eval prompts."""
    out.mkdir(parents=True, exist_ok=True)
    m = Model(w, "gptq-a8" if gq is not None else "a8", gq)
    with open(out / "W.bin", "wb") as fw, open(out / "F.bin", "wb") as ff:
        for i, lw in enumerate(w["layers"]):
            ff.write(
                np.concatenate([lw["ln1"], lw["ln2"]]).astype(np.float32).tobytes()
            )
            for k in ("qkv", "o", "gu", "down"):
                ln = m.lin[i][k]
                fw.write(pack(ln.q).tobytes())
                ff.write(np.concatenate([ln.s, ln.cs.astype(np.float32)]).tobytes())
        fw.write(pack(m.head.q).tobytes())
        ff.write(w["norm"].astype(np.float32).tobytes())
        ff.write(np.concatenate([m.head.s, m.head.cs.astype(np.float32)]).tobytes())
        ff.write(m.cos[:, :].tobytes() + m.sin[:, :].tobytes())
    w["emb"].astype(np.float16).tofile(out / "emb.f16.bin")
    for i in range(len(models.PROMPTS)):
        for n in ("prompt", "force"):
            (out / f"{n}_{i}.bin").write_bytes(
                (work / "dec_in" / f"{n}_{i}.bin").read_bytes()
            )


def phone_score(work: Path, d: Path):
    def load(sub):
        r = []
        for i in range(len(models.PROMPTS)):
            g = np.fromfile(d / sub / f"gen_{i}.bin", np.int32)
            lg = np.fromfile(d / sub / f"logits_{i}.bin", np.float32).reshape(-1, V)
            r.append((g, lg))
        return r

    return score(work, load("free"), load("forced"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--eval", nargs="*")
    ap.add_argument("--forced-only", action="store_true")
    ap.add_argument("--export", choices=["a8", "gptq"])
    ap.add_argument("--out")
    ap.add_argument("--phone")
    a = ap.parse_args()
    work = Path(a.work)
    if a.phone:
        print(phone_score(work, Path(a.phone)))
        return
    w = load_weights()
    if a.export:
        export(
            w,
            gptq_weights(w, work / "gptq_int8.npz") if a.export == "gptq" else None,
            Path(a.out),
            work,
        )
        return
    gq = None
    for mode in a.eval or []:
        if mode.startswith("gptq-") and gq is None:
            gq = gptq_weights(w, work / "gptq_int8.npz")
        m = Model(w, mode, gq)
        if a.forced_only:
            fo = run(m, work, True)
            print(f"{mode:5s}: {score(work, fo, fo)}", flush=True)
        else:
            print(
                f"{mode:5s}: {score(work, run(m, work, False), run(m, work, True))}",
                flush=True,
            )


if __name__ == "__main__":
    main()
