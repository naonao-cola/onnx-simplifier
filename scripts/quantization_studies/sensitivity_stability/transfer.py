"""Policy transfer: search a mixed-precision policy on checkpoint A, apply it to checkpoint B.

python transfer.py <A> <B> <budget_dB>
"""

import sys

import onnx
from common import (
    BLOCK_REGEX,
    MODELS,
    load_batches,
    mean_logit_sqnr,
    norm_params,
    onnx_path,
    split,
    top1_agreement,
)

from onnxsim import search_activation_precision_for_budget
from onnxsim.calibration_pick import run_outputs
from onnxsim.full_qdq import quantize_full_qdq

a, b, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])
fam = MODELS[a][0]
assert MODELS[b][0] == fam
cal_f, ev_f = split("A")


def data(name):
    mean, std = norm_params(name)
    return load_batches(cal_f, mean, std), load_batches(ev_f, mean, std)


def search(name):
    cal, ev = data(name)
    m = onnx.load(str(onnx_path(name)))
    r = search_activation_precision_for_budget(
        m,
        cal,
        ev,
        budget=budget,
        groups="block",
        block_regex=BLOCK_REGEX[fam],
        metric=mean_logit_sqnr,
        costs="macs",
    )
    return m, cal, ev, r


ma, _, _, ra = search(a)
mb, calb, evb, rb = search(b)
qb = quantize_full_qdq(
    mb, calb, exclude_nodes=ra.exclude_nodes, tensor_dtypes=ra.tensor_dtypes
)
fo, qo = run_outputs(mb, evb), run_outputs(qb, evb)
s = mean_logit_sqnr(fo, qo)
promoted = {g: lv for g, lv in ra.levels.items() if lv != "uint8"}
own = {g: lv for g, lv in rb.levels.items() if lv != "uint8"}
print(
    f"| {a} -> {b} | {budget:.0f} dB | A: {ra.promoted_nodes} nodes {promoted} | "
    f"A's policy on B: {s:.1f} dB, top-1 {top1_agreement(fo, qo):.2f}, {'meets' if s >= budget else 'MISSES'} | "
    f"B's own: {rb.promoted_nodes} nodes {own} ({rb.score:.1f} dB) |",
    flush=True,
)
