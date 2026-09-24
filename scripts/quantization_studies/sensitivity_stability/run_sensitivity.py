"""Block ("only" + "all_but") and op-type ("only") activation sensitivity for one checkpoint.

python run_sensitivity.py <model> [split A|B]  ->  $SENSSTAB_WORK/results/<model>_<split>.json
"""

import json
import sys
import time

import onnx
from common import (
    BLOCK_REGEX,
    MODELS,
    WORK,
    load_batches,
    mean_logit_sqnr,
    norm_params,
    onnx_path,
    split,
)

from onnxsim import analyze_activation_sensitivity

name = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "A"
fam = MODELS[name][0]
model = onnx.load(str(onnx_path(name)))
mean, std = norm_params(name)
cal_f, ev_f = split(which)
cal, ev = load_batches(cal_f, mean, std), load_batches(ev_f, mean, std)

out = {"model": name, "family": fam, "split": which, "reports": {}}
for label, kw in [
    ("block_only", dict(groups="block", mode="only", block_regex=BLOCK_REGEX[fam])),
    (
        "block_all_but",
        dict(groups="block", mode="all_but", block_regex=BLOCK_REGEX[fam]),
    ),
    ("op_type_only", dict(groups="op_type", mode="only")),
]:
    t = time.time()
    rep = analyze_activation_sensitivity(model, cal, ev, metric=mean_logit_sqnr, **kw)
    out["reports"][label] = {
        "float_score": rep.float_score,
        "all_quantized_score": rep.all_quantized_score,
        "groups": [
            {
                "group": g.group,
                "n_nodes": len(g.nodes),
                "score": g.score,
                "sensitivity": g.sensitivity,
            }
            for g in rep.groups
        ],
    }
    print(
        f"{name} {which} {label}: {len(rep.groups)} groups, all-uint8 {rep.all_quantized_score}, "
        f"top3 {[g.group for g in rep.groups[:3]]} ({time.time() - t:.0f}s)",
        flush=True,
    )

(WORK / "results").mkdir(parents=True, exist_ok=True)
(WORK / "results" / f"{name}_{which}.json").write_text(json.dumps(out, indent=1))
