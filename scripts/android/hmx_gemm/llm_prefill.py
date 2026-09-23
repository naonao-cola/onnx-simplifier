"""SmolLM2-135M's 128-token prefill projections on the phone's HMX (hmx_gemm_llm_client).

prep:    capture every layer's real projection inputs (fp16) for a 128-token prompt with torch hooks,
         fuse q|k|v and gate|up, prepack the fp16 weights into HMX tiles (hmx_pack_w_f16's layout), and
         write the batch manifest + fp32 torch reference outputs.
compare: phone outputs vs the torch reference (per GEMM cos / max rel err) and the timing summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = (
    "The Hexagon DSP in a Snapdragon phone has a vector unit (HVX) and, since the HTP generation, a "
    "matrix unit (HMX) that the NPU uses for convolutions and matrix multiplies. This note measures how "
    "fast our own code can drive that matrix unit for a small language model's prompt processing, and "
    "compares it with the NPU running the same model through QNN. The model reads a long prompt first, "
    "so every projection becomes a real matrix multiply over all of the prompt's tokens at once."
)


def idx(i, j):
    return 64 * (i // 2) + 2 * j + i % 2


def pack_w(w):
    """K x N fp16 -> hmx_pack_w_f16 layout (per 32-column block, K/32 tiles of W(k, c))."""
    K, N = w.shape
    perm = np.empty(1024, np.int64)
    for k in range(32):
        for c in range(32):
            perm[idx(k, c)] = k * 32 + c
    t = (
        w.reshape(K // 32, 32, N // 32, 32)
        .transpose(2, 0, 1, 3)
        .reshape(N // 32, K // 32, 1024)
    )
    return np.ascontiguousarray(t[:, :, perm]).astype(np.float16)


def prep(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32
    ).eval()
    ids = tok(PROMPT + " " + PROMPT, return_tensors="pt").input_ids[
        :, : a.tokens
    ]  # repeated to reach 128 tokens
    assert ids.shape[1] == a.tokens, f"prompt has only {ids.shape[1]} tokens"
    cap = {}

    def hook(name):
        def f(mod, inp, outp):
            cap[name] = inp[0][0].detach().to(torch.float16)

        return f

    hs = []
    for i, layer in enumerate(model.model.layers):
        hs += [
            layer.self_attn.q_proj.register_forward_hook(hook(f"{i}.qkv")),
            layer.self_attn.o_proj.register_forward_hook(hook(f"{i}.o")),
            layer.mlp.gate_proj.register_forward_hook(hook(f"{i}.gateup")),
            layer.mlp.down_proj.register_forward_hook(hook(f"{i}.down")),
        ]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    man = []
    for i, layer in enumerate(model.model.layers):
        at, ml = layer.self_attn, layer.mlp
        mats = {
            "qkv": torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0),
            "o": at.o_proj.weight,
            "gateup": torch.cat([ml.gate_proj.weight, ml.up_proj.weight], 0),
            "down": ml.down_proj.weight,
        }
        for n, wt in mats.items():
            name = f"{i}.{n}"
            A = cap[name].numpy()  # M x K fp16
            W = wt.detach().to(torch.float16).numpy().T.copy()  # K x N fp16
            M, K = A.shape
            N = W.shape[1]
            A.tofile(out / f"{name}.a")
            pack_w(W).tofile(out / f"{name}.wp")
            ref = A.astype(np.float64) @ W.astype(np.float64)
            ref.astype(np.float32).tofile(out / f"{name}.ref")
            man.append((name, M, K, N))
    with open(out / "manifest.txt", "w") as f:
        for name, M, K, N in man:
            f.write(f"{name} {M} {K} {N}\n")
    print(f"wrote {len(man)} GEMMs to {out}")


def compare(a):
    w = Path(a.work)
    rows = [line.split() for line in open(w / "manifest.txt")]
    tim = {
        line.split()[0]: line.split()[1:] for line in open(w / a.phone / "times.txt")
    }
    worst_cos, worst_rel, tot_dsp, tot_wall, macs = 1.0, 0.0, 0.0, 0.0, 0.0
    per = {}
    for name, M, K, N in rows:
        M, K, N = int(M), int(K), int(N)
        ref = (
            np.fromfile(w / f"{name}.ref", np.float32).reshape(M, N).astype(np.float64)
        )
        got = (
            np.fromfile(w / a.phone / f"{name}.c", np.float16)
            .reshape(M, N)
            .astype(np.float64)
        )
        cos = float((ref * got).sum() / np.sqrt((ref * ref).sum() * (got * got).sum()))
        rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
        worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
        dsp, wall = (float(x) for x in tim[name][:2])
        tot_dsp, tot_wall, macs = tot_dsp + dsp, tot_wall + wall, macs + M * K * N
        kind = name.split(".")[1]
        p = per.setdefault(kind, [0.0, 0])
        p[0] += dsp
        p[1] += 1
    print(
        f"{len(rows)} GEMMs vs torch (fp16 inputs, float64 reference): worst cos {worst_cos:.7f}, worst max-rel err {worst_rel:.2e}"
    )
    for k, (us, n) in per.items():
        print(
            f"  {k:7s} {n} calls, {us / 1e3:.2f} ms on the DSP ({us / n:.0f} us each)"
        )
    print(
        f"total: {tot_dsp / 1e3:.2f} ms on the DSP, {tot_wall / 1e3:.2f} ms wall incl. FastRPC, {macs / tot_dsp / 1e6:.2f} TMAC/s"
    )
    json.dump(
        {
            "worst_cos": worst_cos,
            "worst_rel": worst_rel,
            "dsp_ms": tot_dsp / 1e3,
            "wall_ms": tot_wall / 1e3,
        },
        open(w / a.phone / "summary.json", "w"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "compare"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--phone", default="phone")
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else compare(a)
