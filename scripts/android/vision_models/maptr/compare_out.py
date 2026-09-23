#!/usr/bin/env python3
"""compare_out.py <work> <piece> <suffix> <mode>: phone outputs vs export.py's torch fp32 reference."""
import re
import sys
from pathlib import Path

import numpy as np

w, piece, suf, mode = sys.argv[1:5]
w = Path(w)
outs = re.findall(r"^out (\d+) (\S+) f32", (w / f"{piece}{suf}.{mode}.out").read_text(), re.M)
for i, name in outs:
    ref = np.fromfile(w / f"{piece}.in" / f"ref_{name}.bin", np.float32).astype(np.float64)
    got = np.fromfile(w / f"{piece}{suf}.{mode}.o{i}.bin", np.float32).astype(np.float64)
    cos = ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-30)
    rel = np.abs(ref - got).max() / (np.abs(ref).max() + 1e-30)
    print(f"  {mode:12s} {name}: cos {cos:.5f}  max abs {np.abs(ref - got).max():.3g} (rel to max {rel:.3g})")
