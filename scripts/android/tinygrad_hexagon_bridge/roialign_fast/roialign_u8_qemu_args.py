#!/usr/bin/env python3
"""argv for roialign_u8_qemu.c from one capture_merged_io.py directory:
    python3 roialign_u8_qemu_args.py CAPTURE_DIR box|mask [sort]"""
import sys
from pathlib import Path

d, tag = Path(sys.argv[1]), sys.argv[2]
sort = int(len(sys.argv) > 3 and sys.argv[3] == "sort")
maps, counts, head = {}, {}, None
for f in (l.split() for l in (d / "meta.txt").read_text().splitlines()):
    if f[0] == "map":
        maps[int(f[1])] = f[2:]  # H W C s z spatial_scale
    elif f[0] == f"{tag}_level":
        counts[int(f[1])] = int(f[2])
    elif f[0] == tag:
        head = f[1:]  # N OH OW sr s_out z_out
N, OH, _, sr, s_out, z_out = head
args = [str(d / f"{tag}_ref_u8.bin"), maps[0][2], OH, sr, s_out, z_out, N, str(sort)]
for k in range(4):
    H, W, _, s, z, sp = maps[k]
    n = counts.get(k, 0)
    paths = [str(d / f"{tag}_rois{k}.bin"), str(d / f"{tag}_rows{k}.bin")] if n else ["-", "-"]
    args += [str(d / f"l{k}_u8.bin"), H, W, z, s, sp, str(n)] + paths
print(" ".join(args))
