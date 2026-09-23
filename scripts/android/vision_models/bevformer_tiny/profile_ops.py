#!/usr/bin/env python3
"""Per-op-type share of one HTP execute, from a QNN detailed-profiling CSV.

  profile_ops.py <model.onnx> <prof.csv>
The CSV comes from qnn_run_multi with QNN_EXTRA='profiling_level=detailed,profiling_file_path=...'
(see README). Node events repeat once per execute; the last execute is used. Detailed profiling
serializes ops, so shares are relative (scale by the plain run's latency), not absolute times.
"""
import collections
import csv
import re
import sys

import onnx

m = onnx.load(sys.argv[1], load_external_data=False)
op = {n.name: n.op_type for n in m.graph.node}
rows = [r for r in csv.DictReader(open(sys.argv[2])) if r["Message"] == "NODE"]
names = [re.sub(r":OpId_\d+ \(cycles\)$", "", r["Event Identifier"]) for r in rows]
last = len(names) - 1 - names[::-1].index(names[0]) if names else 0  # start of the last execute
by_op, by_node = collections.Counter(), collections.Counter()
for name, r in zip(names[last:], rows[last:]):
    base = re.sub(r"_token_\d+$", "", name)
    t = op.get(base) or op.get(name) or "QNN:" + re.sub(r"[_/].*", "", name.split("/")[-1])
    by_op[t] += int(r["Time"])
    by_node[name] += int(r["Time"])
tot = sum(by_op.values())
print(f"{len(names) - last} QNN nodes, {tot / 1e6:.1f} Mcycles")
for t, c in by_op.most_common(15):
    print(f"  {t:22s} {100 * c / tot:5.1f}%")
print("top nodes:")
for n, c in by_node.most_common(12):
    print(f"  {n:60s} {100 * c / tot:5.1f}%")
