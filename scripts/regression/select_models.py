#!/usr/bin/env python3
"""Build / refresh the regression model set (models.json) from the Hugging Face
`onnxmodelzoo` org.

`models.json` is a *curated* roster — a representative spread across
architecture families (classic CNNs, ViT/hybrid transformers, NLP transformers,
detectors) rather than every export. This script makes that roster regenerable
and maintainable:

  * verify every curated id still exists in the org;
  * `--refresh` bumps each entry to the newest Opset export of the same model;
  * `--add REGEX` pulls in new families (e.g. NLP transformers) as one
    representative per model, newest opset;
  * `--report` (default) prints coverage: which architecture families in the
    org are represented and which are not, so gaps are a decision, not an
    accident.

Existing `baseline_seconds` / `baseline_*_nodes` / `known_slow` / `size` are
always preserved across a regenerate — only ids and membership change. New
entries get null baselines (fill them with a run of run_regression.py), a null
`size`, and known_slow=false.

`size` is the byte size of the repo's largest `.onnx` file — the one the
converters (WASM UI / regression workers) download. It powers the size preview
in the browser converter. Fill or refresh it with `--refresh-sizes`, which
queries the HF API for each entry (online only).

Model ids come from the live HF API (`huggingface_hub`) or, offline, from a
cached JSON list via `--ids-file` (a plain list of "onnxmodelzoo/<name>").
Nothing is written unless `--write` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(HERE, "models.json")
AUTHOR = "onnxmodelzoo"

# Vendor / training-recipe prefixes that sit in front of the real architecture
# name (tf_efficientnet -> efficientnet, gluon_resnet -> resnet). Stripped only
# to derive the family key for grouping/coverage, never from the id itself.
VENDOR_PREFIXES = (
    "tf", "gluon", "jx", "ssl", "swsl", "legacy", "adv", "tv", "timm", "gcresnext",
)

OPSET_RE = re.compile(r"_Opset(\d+)$")
TRAILING_VER_RE = re.compile(r"-\d+$")


def name_of(model_id):
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def stem(model_id):
    """Model identity without the opset export suffix — `bert_Opset18` and
    `bert_Opset17` share the stem `bert`, so they are the same model."""
    n = name_of(model_id)
    n = OPSET_RE.sub("", n)
    return n


def opset_of(model_id):
    m = OPSET_RE.search(name_of(model_id))
    return int(m.group(1)) if m else -1


def arch_family(model_id):
    """Coarse architecture root used for coverage grouping: resnet, vit, bert,
    efficientnet, ... Heuristic but stable."""
    n = stem(model_id).lower()
    n = TRAILING_VER_RE.sub("", n)          # drop -10, -7, -12 version tags
    n = n.replace("-int8", "").replace("-qdq", "")
    tokens = re.split(r"[_\-]", n)
    tokens = [t for t in tokens if t]
    while tokens and tokens[0] in VENDOR_PREFIXES:
        tokens.pop(0)
    if not tokens:
        return n
    # Family root = leading alpha run of the first token (efficientnetv2 -> keep
    # as-is; resnet18d -> resnet; vit -> vit).
    head = tokens[0]
    m = re.match(r"^([a-z]+?)(v\d+)?(\d.*)?$", head)
    if m:
        root = m.group(1) + (m.group(2) or "")
        return root or head
    return head


def load_org_ids(ids_file):
    if ids_file:
        with open(ids_file) as f:
            ids = json.load(f)
        return [i for i in ids if name_of(i)]
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed; pass --ids-file with a cached listing")
    api = HfApi()
    return [m.id for m in api.list_models(author=AUTHOR)]


def load_models_json(path):
    with open(path) as f:
        return json.load(f)


def new_entry(model_id):
    return {
        "id": model_id,
        "size": None,
        "baseline_seconds": 0.0,
        "baseline_orig_nodes": None,
        "baseline_simp_nodes": None,
        "known_slow": False,
    }


def largest_onnx_size(api, repo):
    """Byte size of the repo's largest `.onnx` file (the one the converters
    download), or None when the repo has no `.onnx`. Mirrors the "largest wins"
    selection in scripts/convertmodel/hf_models.mjs and the regression workers."""
    info = api.model_info(repo, files_metadata=True)
    onnx = [s for s in (info.siblings or []) if s.rfilename.endswith(".onnx")]
    if not onnx:
        return None
    onnx.sort(key=lambda s: s.size or 0, reverse=True)
    return onnx[0].size


def representative(ids_for_stem):
    """Among exports of one model, take the newest opset (then lexical)."""
    return sorted(ids_for_stem, key=lambda i: (-opset_of(i), i))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-file", help="cached JSON list of org model ids (offline)")
    ap.add_argument("--models-json", default=MODELS_JSON)
    ap.add_argument("--refresh", action="store_true",
                    help="bump each curated entry to the newest opset of the same model")
    ap.add_argument("--add", action="append", default=[], metavar="REGEX",
                    help="add newest-opset representative for every model whose id "
                         "matches REGEX (repeatable), e.g. --add '(distilbert|roberta|albert)'")
    ap.add_argument("--refresh-sizes", action="store_true",
                    help="fill/refresh each entry's `size` (largest .onnx bytes) "
                         "from the live HF API; online only")
    ap.add_argument("--report", action="store_true",
                    help="print family coverage report (default action)")
    ap.add_argument("--write", action="store_true", help="write the updated models.json")
    args = ap.parse_args()

    org_ids = load_org_ids(args.ids_file)
    org_by_stem = {}
    for i in org_ids:
        org_by_stem.setdefault(stem(i), []).append(i)
    org_id_set = set(org_ids)

    data = load_models_json(args.models_json)
    models = data["models"]
    by_id = {m["id"]: m for m in models}
    # Preserve baselines by stem too, so a --refresh that swaps the opset keeps them.
    by_stem = {stem(m["id"]): m for m in models}

    changed = []

    # 1. Verify existence.
    for m in models:
        if m["id"] not in org_id_set:
            alt = org_by_stem.get(stem(m["id"]))
            if alt:
                changed.append(f"MISSING {m['id']} -> exists as {representative(alt)} "
                               f"(use --refresh to update)")
            else:
                changed.append(f"MISSING {m['id']} -> no export of '{stem(m['id'])}' in org")

    # 2. Refresh opsets.
    if args.refresh:
        for m in models:
            alt = org_by_stem.get(stem(m["id"]))
            if not alt:
                continue
            best = representative(alt)
            if best != m["id"]:
                changed.append(f"REFRESH {m['id']} -> {best}")
                m["id"] = best

    # 3. Add new families.
    added = []
    for pattern in args.add:
        rx = re.compile(pattern, re.I)
        matched_stems = sorted({stem(i) for i in org_ids if rx.search(name_of(i))})
        for st in matched_stems:
            if st in by_stem:
                continue  # already represented
            rep = representative(org_by_stem[st])
            entry = new_entry(rep)
            models.append(entry)
            by_id[rep] = entry
            by_stem[st] = entry
            added.append(rep)
    if added:
        changed += [f"ADD {a}" for a in added]

    # 4. Refresh sizes (largest .onnx bytes) from the live HF API.
    if args.refresh_sizes:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            sys.exit("huggingface_hub not installed; --refresh-sizes needs the live API")
        api = HfApi()
        for m in models:
            try:
                size = largest_onnx_size(api, m["id"])
            except Exception as e:  # noqa: BLE001 — report and continue
                changed.append(f"SIZE-FAIL {m['id']}: {e}")
                continue
            if m.get("size") != size:
                changed.append(f"SIZE {m['id']} -> {size}")
                m["size"] = size

    models.sort(key=lambda m: m["id"].lower())

    # 5. Coverage report.
    if args.report or not (args.refresh or args.add or args.refresh_sizes):
        covered = {arch_family(m["id"]) for m in models}
        org_fams = {}
        for i in org_ids:
            org_fams.setdefault(arch_family(i), 0)
            org_fams[arch_family(i)] += 1
        missing = sorted((f for f in org_fams if f not in covered),
                         key=lambda f: -org_fams[f])
        print(f"# org models: {len(org_ids)} across {len(org_fams)} families")
        print(f"# curated set: {len(models)} models covering {len(covered)} families")
        print(f"# families in org NOT represented: {len(missing)}")
        for f in missing[:40]:
            print(f"    {f:24} ({org_fams[f]} export(s))")
        if len(missing) > 40:
            print(f"    ... and {len(missing) - 40} more")

    if changed:
        print("\n# changes:")
        for c in changed:
            print("   ", c)
    else:
        print("\n# no id/membership changes")

    if args.write:
        with open(args.models_json, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.models_json} ({len(models)} models)")
    elif changed:
        print("\n(dry run — pass --write to persist)")


if __name__ == "__main__":
    main()
