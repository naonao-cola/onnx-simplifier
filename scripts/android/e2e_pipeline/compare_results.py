#!/usr/bin/env python3
"""Detection agreement of phone results vs the all-ONNX-Runtime reference, with
scripts/android/maskrcnn_e2e/README.md's metric (score > 0.5, same label, box IoU > 0.5).

  python compare_results.py REF_DIR RES_DIR STAGE [STAGE...]
REF_DIR: prepare_inputs.py's ref/ (npy); RES_DIR/<stage>/<stem>_<k>.{bin,shape} from e2e_run.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maskrcnn_e2e"))
from eval_common import compare  # noqa: E402

DT = {1: np.float32, 2: np.uint8, 7: np.int64}


def load(res, stem):
    out = []
    for k in range(4):
        f = res / f"{stem}_{k}"
        t, *shape = (int(x) for x in (f.with_suffix(".shape")).read_text().split())
        out.append(np.fromfile(f.with_suffix(".bin"), DT[t]).reshape(shape))
    return out


def main():
    ref, res, stages = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    stems = sorted(p.name[:-6] for p in ref.glob("*_0.npy"))
    for st in stages:
        rows = []
        for s in stems:
            if not (res / st / f"{s}_0.shape").exists():
                continue
            R = [np.load(ref / f"{s}_{k}.npy") for k in range(4)]
            rows.append((s, compare(R, load(res / st, s))))
        if not rows:
            print(f"{st}: no results")
            continue
        tot_ref = sum(r["ref_detections"] for _, r in rows)
        tot_got = sum(r["tvm_detections"] for _, r in rows)
        tot_m = sum(r["matched"] for _, r in rows)
        w = lambda key: np.average([r[key] for _, r in rows if r[key] is not None],  # noqa: E731
                                   weights=[r["matched"] for _, r in rows if r[key] is not None])
        print(f"{st}: matched {tot_m}/{tot_ref} ref dets ({tot_got} dets), box IoU {w('mean_box_iou'):.3f}, "
              f"score |d| {w('mean_score_absdiff'):.3f}, mask IoU {w('mean_mask_iou'):.3f}")
        for s, r in rows:
            print(f"   {s}: {r['matched']}/{r['ref_detections']} ({r['tvm_detections']} dets) box {r['mean_box_iou']} "
                  f"mask {r['mean_mask_iou']}")


if __name__ == "__main__":
    main()
