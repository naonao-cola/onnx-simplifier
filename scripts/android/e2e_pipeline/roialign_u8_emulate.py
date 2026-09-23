#!/usr/bin/env python3
"""End-to-end host check of the merged uint8 RoiAlign kernel (../tinygrad_hexagon_bridge/roialign_fast/
roialign_u8_kernel.h) inside pipe_e_opt.txt, against the all-ONNX-Runtime reference.

For each image this runs pipe_e_opt.txt with host_emulate.run_pipe (every step on ORT CPU), then
replays the pipe's tail from the box RoiAlign span on with that span -- dq x4, roialign x4, the seg2
ScatterND merge -- replaced by ONE call of the kernel (its CPU build, roialign_u8_cpu.c, through
ctypes; byte-identical to the CDSP skel), and likewise the mask span (roialign x4 + seg4). The kernel
emits the head's uint8 input directly; the *_nhwc heads here start with QuantizeLinear at exactly
those qparams, so feeding them s*(q - z) makes them see q -- the same bytes the *_u8 heads of
pipe_e_u8.txt would get. Prints compare_results.py's metrics for both runs (unmodified e_opt and
e_opt with the kernel) and the head-input byte agreement.

  python roialign_u8_emulate.py OUT_DIR REF_DIR image.bin[,image.bin...]
OUT_DIR: build_models.py's --out; REF_DIR: prepare_inputs.py's ref/. Needs clang (vector builtins).
"""
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maskrcnn_e2e"))
import host_emulate  # noqa: E402
from eval_common import compare  # noqa: E402

KSRC = HERE.parent / "tinygrad_hexagon_bridge" / "roialign_fast"
FINAL = ["6568", "6570", "6572", "6887"]


def load_kernel():
    so = Path(tempfile.gettempdir()) / f"libroialign_u8_cpu_{os.getpid()}.so"
    subprocess.run([os.environ.get("CC", "clang"), "-O2", "-shared", "-fPIC", "-o", str(so),
                    str(KSRC / "roialign_u8_cpu.c")], check=True)
    lib = ctypes.CDLL(str(so))
    so.unlink()
    lib.roialign_u8_cpu_run.restype = ctypes.c_int
    return lib


def ptr(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def run_u8(lib, maps, rois_rows, OH, sr, s_out, z_out):
    """maps: [(uint8 NHWC (1,H,W,C), s, z, spatial_scale)] x4; rois_rows: [(rois (n,4), rows (n,))] x4"""
    C = maps[0][0].shape[3]
    keep = [np.ascontiguousarray(m[0][0]) for m in maps]
    marr = (ctypes.POINTER(ctypes.c_uint8) * 4)(*[ptr(m, ctypes.c_uint8) for m in keep])
    geom = np.array([v for m in maps for v in (m[0].shape[1], m[0].shape[2], m[2])], np.int32)
    fp = np.array([v for m in maps for v in (m[1], m[3])], np.float32)
    counts = np.array([r.shape[0] for r, _ in rois_rows], np.int32)
    rois = np.ascontiguousarray(np.concatenate([r for r, _ in rois_rows]).astype(np.float32))
    rows = np.ascontiguousarray(np.concatenate([w for _, w in rois_rows]).astype(np.int32))
    N = int(counts.sum())
    out = np.empty((N, OH, OH, C), np.uint8)
    rc = lib.roialign_u8_cpu_run(marr, ptr(geom, ctypes.c_int32), ptr(fp, ctypes.c_float), ptr(counts, ctypes.c_int32),
                                 ptr(rois, ctypes.c_float), ptr(rows, ctypes.c_int32), C, OH, OH, sr,
                                 ctypes.c_float(s_out), z_out, 0, ptr(out, ctypes.c_uint8), out.size)
    assert rc == 0, rc
    return out


def replay(out, store, lines):
    """host_emulate.run_pipe's ort / ortpad dispatch, on an existing store"""
    L = lambda s: [] if s == "-" else s.split(",")  # noqa: E731
    for f in (l.split() for l in lines):
        if f[0] == "ort":
            host_emulate.run_model(out / f[2], store, L(f[5]), L(f[6]))
        elif f[0] == "ortpad":
            padin, buckets, ins, outs = f[4], f[5], L(f[6]), L(f[7])
            x = store[padin]
            n = x.shape[0]
            b, model = min(((int(k), v) for k, v in (e.split(":") for e in buckets.split(",")) if int(k) >= n),
                           key=lambda t: t[0])
            shp = host_emulate.sess(out / model).get_inputs()[0].shape
            xp = np.zeros([b] + list(x.shape[1:]), x.dtype)
            xp[:n] = x
            host_emulate.run_model(out / model, store, ins, outs, {0: xp.reshape(shp)})
            for o in outs:
                store[o] = store[o][:n]
        else:
            raise ValueError(f)


def metrics(ref, stems, results):
    rows = [(s, compare([np.load(ref / f"{s}_{k}.npy") for k in range(4)], results[s])) for s in stems]
    m = sum(r["matched"] for _, r in rows)
    tr = sum(r["ref_detections"] for _, r in rows)
    tg = sum(r["tvm_detections"] for _, r in rows)
    w = lambda key: np.average([r[key] for _, r in rows if r[key] is not None],  # noqa: E731
                               weights=[r["matched"] for _, r in rows if r[key] is not None])
    return (f"matched {m}/{tr} ref dets ({tg} dets), box IoU {w('mean_box_iou'):.3f}, "
            f"score |d| {w('mean_score_absdiff'):.3f}, mask IoU {w('mean_mask_iou'):.3f}")


def main():
    out, ref, images = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
    sys.path.insert(0, str(KSRC))
    from capture_merged_io import head_input_qparams
    lib = load_kernel()
    lines = (out / "pipe_e_opt.txt").read_text().splitlines()
    F = [l.split() for l in lines]
    dq = {f[2]: (f[1], float(f[3]), int(f[4])) for f in F if f[0] == "dq"}
    idx = {f[1]: i for i, f in enumerate(F) if f[0] == "ort"}
    ra = [(i, f) for i, f in enumerate(F) if f[0] == "roialign"]
    box_ra, mask_ra = [f for _, f in ra if f[4] == "7"], [f for _, f in ra if f[4] == "14"]
    first_box = min(i for i, f in ra if f[4] == "7")
    bq = head_input_qparams(out / "box_head_1000_nhwc.onnx")
    mq = head_input_qparams(out / "mask_head_100_nhwc.onnx")
    base, new, stems = {}, {}, []
    for im in images:
        stem = Path(im).stem
        stems.append(stem)
        x = np.fromfile(im, np.float32).reshape(3, 800, 1088)
        st = host_emulate.run_pipe(out, "e_opt", x)
        base[stem] = [st[n] for n in FINAL]
        maps = [(st[dq[f[1]][0]], dq[f[1]][1], dq[f[1]][2], float(f[7])) for f in box_ra]
        new_st = dict(st)
        for n in FINAL:
            del new_st[n]
        agree = []
        for tag, spans, seg, (hs, hz), oh in (("box", box_ra, "seg2", bq, 7), ("mask", mask_ra, "seg4", mq, 14)):
            if tag == "mask":  # run the (new) box head + seg3 first: the mask RoIs depend on it
                replay(out, new_st, lines[idx["seg2"] + 1: idx["seg3"] + 1])
            f = F[idx[seg]]
            ins = f[5].split(",")
            rows_by_out = {ins[i + 1]: ins[i] for i in range(1, len(ins) - 1, 2)}
            rr = [(new_st[sp[2]], new_st[rows_by_out[sp[3]]].reshape(-1)) for sp in spans]
            q = run_u8(lib, maps, rr, oh, int(spans[0][6]), hs, hz)
            if all(np.array_equal(new_st[sp[2]], st[sp[2]]) for sp in spans):
                ref_q = np.clip(np.rint(st[f[6]].reshape(q.shape) / np.float32(hs)) + hz, 0, 255).astype(np.uint8)
                agree.append(f"{tag} {100 * np.mean(q == ref_q):.3f}% exact")
            else:  # the uint8 box path moved the detections the mask RoIs come from
                agree.append(f"{tag} n/a (RoIs changed by the box path: {q.shape[0]} vs {st[f[6]].shape[0]})")
            new_st[f[6]] = ((q.astype(np.int32) - hz).astype(np.float32) * np.float32(hs)).reshape(
                (q.shape[0],) + st[f[6]].shape[1:])
        replay(out, new_st, lines[idx["seg4"] + 1:])
        new[stem] = [new_st[n] for n in FINAL]
        print(stem, "head input bytes vs quant(fp32 merged):", ", ".join(agree), flush=True)
    assert first_box < idx["seg2"]
    print("e_opt (host, ORT RoiAlign)    :", metrics(ref, stems, base))
    print("e_opt + roialign_u8 (host)    :", metrics(ref, stems, new))


if __name__ == "__main__":
    main()
