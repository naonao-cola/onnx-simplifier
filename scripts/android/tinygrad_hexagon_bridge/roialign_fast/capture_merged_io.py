#!/usr/bin/env python3
"""Capture the real inputs/outputs of the whole *merged* RoiAlign span of the e2e pipeline (PR #1841)
for the uint8 / merged-rows kernel (roialign_u8_kernel.h).

In pipe_e_opt.txt the box (and mask) span is
    dq x4 (backbone uint8 NHWC FPN maps -> fp32) ; roialign x4 (one per FPN level) ;
    seg2 (ScatterND of the 4 per-level outputs into the merged rows) ; [quant to the head's uint8 input]
This runs ../../e2e_pipeline/host_emulate.py's run_pipe on the host (every step on ORT CPU) and saves,
per image, exactly what one merged kernel call needs and must reproduce:

  maps  : l{0..3}_u8.bin  uint8 NHWC maps (the order of the roialign lines), meta.txt has their qparams
  box   : box_rois{k}.bin (fp32 [n_k,4]), box_rows{k}.bin (int32 [n_k] destination row per RoI),
          box_ref_f32.bin (the merged fp32 rows, (N,7,7,256) NHWC), box_ref_u8.bin (QuantizeLinear of
          that with the box head's input scale/zp)
  mask  : the same with 14x14 and the mask head's input scale/zp

    python capture_merged_io.py E2E_OUT_DIR IMAGE.bin[,IMAGE.bin...] DEST_DIR
E2E_OUT_DIR is build_models.py's --out (it must contain pipe_e_opt.txt and the models it names).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e_pipeline"))
import host_emulate  # noqa: E402

# The heads' uint8 input qparams: the first QuantizeLinear of box_head_1000_nhwc.onnx / mask_head_*.
def head_input_qparams(model_path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(model_path), load_external_data=False)
    inp = m.graph.input[0].name
    inits = {i.name: i for i in m.graph.initializer}
    consts = {}
    for n in m.graph.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    frontier = {inp}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] in frontier:
            def val(t):
                return numpy_helper.to_array(inits[t]) if t in inits else consts[t]
            return float(val(n.input[1])), int(val(n.input[2]))
        if n.op_type in ("Transpose", "Reshape", "Flatten", "Identity") and n.input[0] in frontier:
            frontier.update(n.output)
    raise RuntimeError(f"no input QuantizeLinear in {model_path}")


def quant(x, s, z):
    return np.clip(np.rint(x / np.float32(s)) + z, 0, 255).astype(np.uint8)


def main():
    out, images, dest = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
    lines = (out / "pipe_e_opt.txt").read_text().splitlines()
    dq = {f[2]: (f[1], float(f[3]), int(f[4])) for f in (l.split() for l in lines) if f[0] == "dq"}
    ra = [l.split() for l in lines if l.startswith("roialign ")]
    box_ra, mask_ra = [f for f in ra if f[4] == "7"], [f for f in ra if f[4] == "14"]
    segs = {f[1]: f for f in (l.split() for l in lines) if f[0] == "ort"}
    bq = head_input_qparams(out / "box_head_1000_nhwc.onnx")
    mq = head_input_qparams(out / "mask_head_100_nhwc.onnx")
    for im in images:
        x = np.fromfile(im, np.float32).reshape(3, 800, 1088)
        st = host_emulate.run_pipe(out, "e_opt", x)
        d = dest / Path(im).stem
        d.mkdir(parents=True, exist_ok=True)
        meta = []
        maps = [f[1] for f in box_ra]  # e.g. 391@nhwc (P5) .. 487@nhwc (P2)
        assert maps == [f[1] for f in mask_ra]
        for k, mname in enumerate(maps):
            src, s, z = dq[mname]
            q = st[src]
            assert q.dtype == np.uint8 and q.ndim == 4 and q.shape[0] == 1, (src, q.shape, q.dtype)
            q[0].tofile(d / f"l{k}_u8.bin")
            H, W, C = q.shape[1:]
            meta.append(f"map {k} {H} {W} {C} {s!r} {z} {box_ra[k][7]}")
        for tag, spans, seg, (hs, hz), oh in (("box", box_ra, "seg2", bq, 7), ("mask", mask_ra, "seg4", mq, 14)):
            f = segs[seg]
            ins = f[5].split(",")
            merged = st[f[6]]
            N = merged.shape[0]
            # seg ins: data, rows_0, roi_out_0, rows_1, roi_out_1, ... (ScatterND data + index/update pairs)
            rows_by_out = {ins[i + 1]: st[ins[i]] for i in range(1, len(ins) - 1, 2)}
            covered = np.zeros(N, np.int32)
            for k, sp in enumerate(spans):
                rois = st[sp[2]].astype(np.float32)
                rows = rows_by_out[sp[3]].reshape(-1).astype(np.int64)
                assert rows.shape[0] == rois.shape[0]
                covered[rows] += 1
                rois.tofile(d / f"{tag}_rois{k}.bin")
                rows.astype(np.int32).tofile(d / f"{tag}_rows{k}.bin")
                meta.append(f"{tag}_level {k} {rois.shape[0]}")
            assert (covered == 1).all(), f"{tag}: rows not covered exactly once: {np.bincount(covered)}"
            ref = np.ascontiguousarray(merged.reshape(N, oh, oh, -1), dtype=np.float32)  # stored as NHWC rows
            ref.tofile(d / f"{tag}_ref_f32.bin")
            quant(ref, hs, hz).tofile(d / f"{tag}_ref_u8.bin")
            meta.append(f"{tag} {N} {oh} {oh} {spans[0][6]} {hs!r} {hz}")
        (d / "meta.txt").write_text("\n".join(meta) + "\n")
        print(d, "\n  " + "\n  ".join(meta), flush=True)


if __name__ == "__main__":
    main()
