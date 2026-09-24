"""Compare block / op-type sensitivity rankings between checkpoints -> markdown tables on stdout."""

import json

import numpy as np
from common import WORK

R = WORK / "results"


def load(name, split="A"):
    d = json.loads((R / f"{name}_{split}.json").read_text())
    for rep in d["reports"].values():
        wins = [g for g in rep["groups"] if g["group"].startswith("window:")]
        # stem = the window before the first block, head = the one after; names differ per exporter
        order = sorted(wins, key=lambda g: g["group"])
        for g in rep["groups"]:
            if g["group"].startswith("window:"):
                low = g["group"].lower()
                g["group"] = (
                    "stem" if ("conv1" in low or "patch_embed" in low) else "head"
                )
        del order
    return d


def sens(d, label):
    """Per-group harm on a linear scale. 'only': the logit noise power one quantized group adds
    (10^(-SQNR/10), additive across groups); 'all_but': the fraction of the all-uint8 noise that
    keeping this group float removes."""
    rep = d["reports"][label]
    if label.endswith("_only"):
        return {g["group"]: 10 ** (-g["score"] / 10) for g in rep["groups"]}
    n_all = 10 ** (-rep["all_quantized_score"]["uint8"] / 10)
    return {g["group"]: 1 - 10 ** (-g["score"] / 10) / n_all for g in rep["groups"]}


def ranks(v):
    order = np.argsort(np.argsort(-np.asarray(v)))
    return order.astype(float)


def spearman(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    ra, rb = ranks([a[k] for k in keys]), ranks([b[k] for k in keys])
    return float(np.corrcoef(ra, rb)[0, 1]), len(keys)


def topk(a: dict, b: dict, k):
    ta = [g for g, _ in sorted(a.items(), key=lambda x: -x[1])[:k]]
    tb = [g for g, _ in sorted(b.items(), key=lambda x: -x[1])[:k]]
    return len(set(ta) & set(tb))


def concentration(a: dict, k=3):
    v = np.sort(np.clip(np.array(list(a.values())), 0, None))[::-1]
    return float(v[:k].sum() / max(v.sum(), 1e-12))


def pair_rows(pairs, label):
    rows = []
    for x, y, kind in pairs:
        try:
            dx, dy = load(*x), load(*y)
        except FileNotFoundError:
            continue
        a, b = sens(dx, label), sens(dy, label)
        rho, n = spearman(a, b)
        rows.append(
            f"| {x[0]}/{x[1]} vs {y[0]}/{y[1]} | {kind} | {n} | {rho:+.2f} | {topk(a, b, 3)}/3 | {topk(a, b, 5)}/5 |"
            f" {concentration(a):.2f} / {concentration(b):.2f} |"
        )
    return rows


PAIRS = [
    (("tv_r50_v1", "A"), ("tv_r50_v1", "B"), "noise floor (other data split)"),
    (("tv_r50_v1", "A"), ("tv_r50_v2", "A"), "same arch, new recipe"),
    (("timm_r50_a1", "A"), ("timm_r50_a1", "B"), "noise floor (other data split)"),
    (("timm_r50_a1", "A"), ("timm_r50_a2", "A"), "same arch, recipe A1 vs A2"),
    (("timm_r50_a1", "A"), ("timm_r50_a3", "A"), "same arch, recipe A1 vs A3"),
    (("timm_r50_a2", "A"), ("timm_r50_a3", "A"), "same arch, recipe A2 vs A3"),
    (("tv_r50_v1", "A"), ("timm_r50_a1", "A"), "same arch, torchvision vs timm"),
    (("tv_r50_v2", "A"), ("timm_r50_a1", "A"), "same arch, torchvision V2 vs timm A1"),
    (("deit_s", "A"), ("deit_s", "B"), "noise floor (other data split)"),
    (("deit_s", "A"), ("vit_s_augreg", "A"), "same arch, DeiT vs AugReg (IN-21k)"),
]

if __name__ == "__main__":
    for label in ["block_only", "block_all_but"]:
        print(
            f"\n### {label}: Spearman of block sensitivity, top-k overlap, top-3 share of total\n"
        )
        print(
            "| pair | kind | groups | Spearman | top-3 | top-5 | top-3 share (a / b) |"
        )
        print("|---|---|---|---|---|---|---|")
        print("\n".join(pair_rows(PAIRS, label)))
    print("\n### Op-type ranking (only mode), worst first\n")
    print(
        "| checkpoint | all-uint8 SQNR | op types, worst first (nodes, share of summed noise) |"
    )
    print("|---|---|---|")
    for f in sorted(R.glob("*_A.json")):
        d = load(f.stem[:-2])
        rep = d["reports"]["op_type_only"]
        tot = sum(10 ** (-g["score"] / 10) for g in rep["groups"])
        s = ", ".join(
            f"{g['group']} ({g['n_nodes']} nodes, {10 ** (-g['score'] / 10) / tot:.0%})"
            for g in rep["groups"][:4]
        )
        print(f"| {d['model']} | {rep['all_quantized_score']['uint8']:.1f} dB | {s} |")
    print("\n### Worst 3 blocks (only mode)\n")
    print("| checkpoint | worst blocks (share of summed per-block noise) |")
    print("|---|---|")
    for f in sorted(R.glob("*.json")):
        d = load(*f.stem.rsplit("_", 1))
        a = sens(d, "block_only")
        tot = sum(a.values())
        g = sorted(a.items(), key=lambda x: -x[1])[:3]
        print(f"| {f.stem} | " + ", ".join(f"{k} ({v / tot:.0%})" for k, v in g) + " |")
