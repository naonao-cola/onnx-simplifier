#!/usr/bin/env python3
"""compare_q8.py <work> <stem> <mode>: the phone's integer outputs vs the host ORT CPU int8 ones
(q8_inputs.py): exact-byte share, max |diff| in quantization steps."""
import re
import sys
from pathlib import Path

import numpy as np

NP = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32, "i64": np.int64}
w, stem, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for i, name, dt in re.findall(r"^out (\d+) (\S+) (\S+)", (w / f"{stem}.{mode}.out").read_text(), re.M):
    ref = np.fromfile(w / f"{stem}.in" / f"ref_{name}.bin", NP[dt]).astype(np.int64)
    got = np.fromfile(w / f"{stem}.{mode}.o{i}.bin", NP[dt]).astype(np.int64)
    d = np.abs(ref - got)
    print(f"  {mode:12s} {name}: {100 * (d == 0).mean():.2f}% exact, max diff {d.max()} steps, mean {d.mean():.3f}")
