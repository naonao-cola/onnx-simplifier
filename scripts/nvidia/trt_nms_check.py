"""Check onnxsim's ``rewrite_trt_batched_nms`` pass against the REAL TensorRT NMS plugin.

The pass (``onnxsim/passes/rewrite_trt_batched_nms.h``) decomposes mmdeploy's closed
``TRTBatchedNMS`` op into standard ONNX ops (``NonMaxSuppression`` + top-K merge), and
its header/tests explicitly say tie-breaking and boundary behavior against the real
plugin were never verified. TensorRT's own ``BatchedNMSDynamic_TRT`` plugin (shipped in
``libnvinfer_plugin``) has the same I/O contract, so it is the closest runnable
reference: same ``[boxes (N,B,1|C,4), scores (N,B,C)] -> [num_detections (N,1) int32,
boxes (N,K,4), scores (N,K), classes (N,K)]`` layout and the same attributes. (mmdeploy's
own ``TRTBatchedNMS`` plugin is a fork of it and is not available here.) There is no
rotated-NMS plugin in TensorRT 10.3's registry, so ``rewrite_trt_batched_rotated_nms``
cannot be checked this way.

Three stages, split by interpreter because JetPack's TensorRT Python bindings are cp310
only while onnxsim needs Python >= 3.11 (models/inputs/outputs are exchanged as files):

    gen      (onnxsim + onnxruntime)  build cases, run the pass, run the rewritten graph on ORT CPU
    trt      (tensorrt, system py3.10) build/run the real plugin via the network API
    compare  (numpy only)             per-case match / differ report

    python trt_nms_check.py all OUT_DIR --py-onnxsim /path/py3.12 --py-trt /usr/bin/python3

(The ONNX parser path was tried first: importing a ``BatchedNMS*_TRT`` node segfaults the
TensorRT 10.3 parser, so the plugin is instantiated directly with ``add_plugin_v2``.)
Engines are tiny; run on an otherwise idle GPU if you also time other things.

Findings (Jetson Orin Nano, JetPack 6, TensorRT 10.3.0, onnxsim 0.7.3.dev3225; 28 cases):
  MATCH (num_detections, kept set, exact order, padding): random multi-class multi-batch
    inputs, background_label_id, keepTopK truncation, scoreThreshold, exact score ties
    (both emit ties in ascending box index; the first index survives a tied overlap),
    near ties down to 1e-6 (scoreBits 16 and 32), IoU == threshold (kept: suppress only
    if IoU > thr), score == scoreThreshold (dropped: needs score > thr), clipBoxes,
    padding (nd < keepTopK -> zero box, score 0, class -1).
  DIFFER: (1) topK -- the plugin feeds only the top-topK boxes per class INTO nms
    (pre-NMS), the rewrite maps it to max_output_boxes_per_class (a cap AFTER nms), so with
    topK smaller than the candidate count the rewrite can return extra detections;
    (2) isNormalized=0 -- TensorRT uses the pixel (+1) area convention, the rewrite
    ignores isNormalized (documented in the pass header).
  OUT OF SCOPE: shareLocation=0 -- the pass declines (graph untouched).
  PLUGIN QUIRKS (invalid configs, not rewrite bugs): the plugin returns num_detections=0
    if topK < keepTopK, and empty results for batch items > 0 when topK >> numBoxes
    (seen at N=2, B=32, topK>=100), so cases keep topK = numBoxes and keepTopK <= topK.
  No rotated-NMS plugin exists in TensorRT 10.3, so the rotated pass is not checkable.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_NAMES = ("nd", "bx", "sc", "cl")
SENTINEL = -777.0  # pre-fill outputs so unwritten padding is distinguishable from 0


# --------------------------------------------------------------------------- cases


def _boxes(rng, n, b, scale, lo=0.5, hi=2.0):
    c = rng.uniform(0, scale, (n, b, 2))
    s = rng.uniform(lo, hi, (n, b, 2))
    return np.concatenate([c - s / 2, c + s / 2], -1).astype(np.float32)


def _case(name, boxes, scores, note="", **kw):
    # Plugin constraints found empirically (TensorRT 10.3): it returns num_detections=0 when
    # topK < keepTopK, and can return nothing for batch items > 0 when topK is much larger than
    # numBoxes (seen at N=2, B=32, topK>=100). So default topK = numBoxes and keepTopK <= topK.
    p = dict(bg=-1, topK=boxes.shape[1], keepTopK=min(20, boxes.shape[1]), scoreThreshold=0.05, iouThreshold=0.5,
             isNormalized=1, clipBoxes=0, share=1, scoreBits=16)
    p.update(kw)
    return dict(name=name, note=note, params=p, boxes=boxes.astype(np.float32),
                scores=scores.astype(np.float32))


def make_cases():
    cases = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        bx = _boxes(rng, 2, 100, 4.0)
        cases.append(_case(f"random_{seed}", bx[:, :, None, :], rng.uniform(0, 1, (2, 100, 6)),
                           "clustered random boxes, 6 classes", keepTopK=30))
    rng = np.random.default_rng(3)
    bx = _boxes(rng, 3, 200, 5.0)
    cases.append(_case("random_big", bx[:, :, None, :], rng.uniform(0, 1, (3, 200, 10)),
                       "N=3, 200 boxes, 10 classes", keepTopK=50))
    bx = _boxes(rng, 4, 50, 3.0)
    cases.append(_case("random_batch4", bx[:, :, None, :], rng.uniform(0, 1, (4, 50, 3)),
                       "N=4, 50 boxes, 3 classes", keepTopK=10))
    # Many exact score ties across boxes and classes (scores on a 0.1 grid), overlapping boxes.
    bx = _boxes(rng, 2, 64, 3.0)
    cases.append(_case("random_quantized_scores", bx[:, :, None, :],
                       np.round(rng.uniform(0.1, 1, (2, 64, 4)), 1), "scores on a 0.1 grid: massive ties",
                       keepTopK=20))
    rng = np.random.default_rng(10)
    bx = _boxes(rng, 2, 32, 4.0)[:, :, None, :]
    sc = rng.uniform(0, 1, (2, 32, 4))
    cases.append(_case("background_0", bx, sc, "background_label_id=0", bg=0))
    cases.append(_case("background_2", bx, sc, "background_label_id=2", bg=2))
    cases.append(_case("keep_top_k_trunc", bx, sc, "keepTopK=5 truncation", keepTopK=5))
    cases.append(_case("score_thr_high", bx, sc, "scoreThreshold=0.9", scoreThreshold=0.9))
    cases.append(_case("no_detections", bx, sc * 0.01, "every score below threshold: padding"))

    # Exact ties. Disjoint boxes, identical scores: which order do the ties come out in?
    n = 8
    xs = np.arange(n, dtype=np.float32) * 2
    bxd = np.stack([xs, np.zeros(n), xs + 1, np.ones(n)], -1)[None, :, None, :]
    cases.append(_case("ties_disjoint", bxd, np.full((1, n, 2), 0.6), "identical scores, no overlap, keepTopK=8 cuts inside the ties",
                       keepTopK=8))
    cases.append(_case("ties_disjoint_1cls", bxd, np.full((1, n, 1), 0.6), "one class, all 8 tied, all kept"))
    # Identical scores on heavily overlapping boxes: which one survives suppression?
    bxo = np.array([[[0, 0, 1, 1], [0.05, 0, 1.05, 1], [0.1, 0, 1.1, 1], [0.15, 0, 1.15, 1]]],
                   np.float32)[:, :, None, :]
    cases.append(_case("ties_overlap", bxo, np.full((1, 4, 1), 0.7), "identical scores, IoU ~0.8"))
    # Scores that differ by less than a 16-bit sort key can resolve.
    sc_near = np.array([[[0.700], [0.701], [0.702], [0.703]]], np.float32)
    cases += [_case("near_ties_sb16", bxo, sc_near, "scores 0.700..0.703, IoU ~0.8, scoreBits=16"),
              _case("near_ties_sb32", bxo, sc_near, "same, scoreBits=32", scoreBits=32)]

    sc_1e6 = np.array([[[0.700000], [0.700001], [0.700002], [0.700003]]], np.float32)
    cases += [_case("near_ties_1e-6_sb16", bxo, sc_1e6, "scores differ by 1e-6, IoU ~0.8, scoreBits=16"),
              _case("near_ties_1e-6_sb32", bxo, sc_1e6, "same, scoreBits=32", scoreBits=32)]

    # Exact boundary conditions (dyadic values: no float rounding involved).
    pair = np.array([[[0, 0, 1, 1], [0, 0, 1, 0.5]]], np.float32)[:, :, None, :]  # IoU == 0.5
    two = np.array([[[0.9], [0.8]]], np.float32)
    cases += [_case("iou_eq_thr", pair, two, "IoU exactly == iouThreshold (0.5)", iouThreshold=0.5),
              _case("iou_below_thr", pair, two, "IoU 0.5 vs threshold 0.49 -> suppress", iouThreshold=0.49),
              _case("iou_above_thr", pair, two, "IoU 0.5 vs threshold 0.51 -> keep", iouThreshold=0.51)]
    cases.append(_case("score_eq_thr", np.tile(bxd[:, :3], (1, 1, 1, 1)),
                       np.array([[[0.75], [0.5], [0.25]]], np.float32),
                       "a score exactly == scoreThreshold (0.5)", scoreThreshold=0.5))

    # clipBoxes: normalized coordinates spilling outside [0,1].
    cb = np.array([[[-0.2, -0.1, 0.5, 0.6], [0.4, 0.3, 1.3, 1.2], [0.6, 0.6, 0.9, 0.9]]],
                  np.float32)[:, :, None, :]
    cases.append(_case("clip_boxes", cb, np.array([[[0.9], [0.8], [0.7]]], np.float32),
                       "clipBoxes=1, boxes exceed [0,1]", clipBoxes=1))
    cases.append(_case("no_clip_boxes", cb, np.array([[[0.9], [0.8], [0.7]]], np.float32),
                       "same boxes, clipBoxes=0"))
    # isNormalized=0: TensorRT measures areas in pixel convention (width+1). The rewrite
    # ignores isNormalized. Normalized IoU=0.444, pixel IoU=0.5; threshold in between.
    pix = np.array([[[0, 0, 9, 9], [0, 0, 9, 4]]], np.float32)[:, :, None, :]
    cases.append(_case("not_normalized", pix, two, "isNormalized=0, IoU 0.444 (plain) / 0.5 (pixel+1), thr 0.47",
                       isNormalized=0, iouThreshold=0.47))
    cases.append(_case("normalized_control", pix, two, "same boxes, isNormalized=1", iouThreshold=0.47))

    # topK semantics. TensorRT: topK = candidates fed INTO nms (top-K by score, pre-NMS).
    # Rewrite: topK -> NonMaxSuppression max_output_boxes_per_class (cap AFTER nms).
    # The 4 best boxes overlap each other (1 survivor); 4 lower-scored boxes are disjoint.
    cl = [[0, 0, 1, 1], [0.02, 0, 1.02, 1], [0.04, 0, 1.04, 1], [0.06, 0, 1.06, 1]]
    dj = [[3, 0, 4, 1], [5, 0, 6, 1], [7, 0, 8, 1], [9, 0, 10, 1]]
    cases.append(_case("topk_pre_vs_post_nms", np.array([cl + dj], np.float32)[:, :, None, :],
                       np.array([[[.9], [.85], [.8], [.75], [.5], [.45], [.4], [.35]]], np.float32),
                       "topK=4: pre-NMS candidates (TRT -> 1 det) vs post-NMS cap (ONNX -> 4 dets)",
                       topK=4, keepTopK=4))

    # shareLocation=0 (per-class boxes): the rewrite must decline this; plugin still runs.
    rng = np.random.default_rng(20)
    bpc = np.stack([_boxes(rng, 1, 16, 3.0) for _ in range(3)], 2)[0][None]  # (1,16,3,4)
    cases.append(_case("share_location_0", bpc, rng.uniform(0, 1, (1, 16, 3)),
                       "shareLocation=0 (boxes (N,B,C,4)): out of the pass's scope", share=0))
    return cases


# ------------------------------------------------------------------------ stage gen


def stage_gen(out):
    import onnx
    import onnxruntime as ort
    import onnxsim
    from onnx import parser

    out.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    arrays, meta = {}, []
    for c in cases:
        p, bx, sc = c["params"], c["boxes"], c["scores"]
        n, b, cb, _ = bx.shape
        ncls = sc.shape[2]
        model = parser.parse_model(f"""
            <ir_version: 10, opset_import: ["": 13, "mmdeploy": 1]>
            g (float[{n},{b},{cb},4] boxes, float[{n},{b},{ncls}] scores)
              => (int32[{n},1] nd, float[{n},{p['keepTopK']},4] bx,
                  float[{n},{p['keepTopK']}] sc, float[{n},{p['keepTopK']}] cl)
            {{
              nd, bx, sc, cl = mmdeploy.TRTBatchedNMS
                <background_label_id={p['bg']}, num_classes={ncls}, topK={p['topK']},
                 keepTopK={p['keepTopK']}, scoreThreshold={p['scoreThreshold']},
                 iouThreshold={p['iouThreshold']}, isNormalized={p['isNormalized']},
                 clipBoxes={p['clipBoxes']}>(boxes, scores)
            }}""")
        sim, ok = onnxsim.simplify(model, check_n=0, extra_optimizers=["rewrite_trt_batched_nms"])
        rewritten = ok and "TRTBatchedNMS" not in [x.op_type for x in sim.graph.node]
        rec = dict(name=c["name"], note=c["note"], params=p, rewritten=bool(rewritten))
        arrays[f"{c['name']}.boxes"], arrays[f"{c['name']}.scores"] = bx, sc
        if rewritten:
            onnx.save(sim, out / f"{c['name']}.rewritten.onnx")
            s = ort.InferenceSession(sim.SerializeToString(), providers=["CPUExecutionProvider"])
            res = s.run(None, {"boxes": bx, "scores": sc})
            for k, r in zip(OUT_NAMES, res):
                arrays[f"{c['name']}.ort.{k}"] = np.asarray(r)
        meta.append(rec)
    np.savez(out / "cases.npz", **arrays)
    (out / "cases.json").write_text(json.dumps(meta, indent=1))
    print(f"gen: {len(meta)} cases, rewritten: {sum(m['rewritten'] for m in meta)}")


# ------------------------------------------------------------------------ stage trt


class PluginRunner:
    """Builds and runs one real ``BatchedNMSDynamic_TRT`` plugin engine (tensorrt interpreter)."""

    def __init__(self):
        import tensorrt as trt

        sys.path.insert(0, str(HERE))
        from trt_harness import Cudart

        self.trt, self.cuda = trt, Cudart()
        self.log = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.log, "")
        self.creator = trt.get_plugin_registry().get_plugin_creator("BatchedNMSDynamic_TRT", "1", "")

    def run(self, bx, sc, p):
        """-> dict of the four outputs (unwritten elements keep SENTINEL), or None on build failure."""
        import ctypes

        trt, cuda = self.trt, self.cuda

        def field(name, val, dt):
            kind = trt.PluginFieldType.INT32 if dt == np.int32 else trt.PluginFieldType.FLOAT32
            return trt.PluginField(name, np.array([val], dtype=dt), kind)

        fields = [field("shareLocation", p["share"], np.int32), field("backgroundLabelId", p["bg"], np.int32),
                  field("numClasses", sc.shape[2], np.int32), field("topK", p["topK"], np.int32),
                  field("keepTopK", p["keepTopK"], np.int32),
                  field("scoreThreshold", p["scoreThreshold"], np.float32),
                  field("iouThreshold", p["iouThreshold"], np.float32),
                  field("isNormalized", p["isNormalized"], np.int32),
                  field("clipBoxes", p["clipBoxes"], np.int32),
                  field("scoreBits", p["scoreBits"], np.int32), field("caffeSemantics", 1, np.int32)]
        plugin = self.creator.create_plugin("nms", trt.PluginFieldCollection(fields))
        builder = trt.Builder(self.log)
        net = builder.create_network(0)
        i0 = net.add_input("boxes", trt.float32, bx.shape)
        i1 = net.add_input("scores", trt.float32, sc.shape)
        layer = net.add_plugin_v2([i0, i1], plugin)
        for i, nm in enumerate(OUT_NAMES):
            layer.get_output(i).name = nm
            net.mark_output(layer.get_output(i))
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
        blob = builder.build_serialized_network(net, cfg)
        if blob is None:
            return None
        engine = trt.Runtime(self.log).deserialize_cuda_engine(bytes(blob))
        ctx = engine.create_execution_context()
        bufs, host = {}, {}
        for i in range(engine.num_io_tensors):
            t = engine.get_tensor_name(i)
            shape = tuple(ctx.get_tensor_shape(t))
            dt = np.dtype(trt.nptype(engine.get_tensor_dtype(t)))
            arr = np.ascontiguousarray({"boxes": bx, "scores": sc}[t]) if t in ("boxes", "scores") \
                else np.full(shape, SENTINEL, dt)
            bufs[t] = cuda.malloc(max(arr.nbytes, 4))
            cuda.memcpy_htod(bufs[t], arr)
            ctx.set_tensor_address(t, bufs[t].value)
            host[t] = arr
        stream = ctypes.c_void_p()
        cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(stream)))
        if not ctx.execute_async_v3(stream.value):
            print("trt: execute_async_v3 FAILED")
        cuda.sync()
        for t in OUT_NAMES:
            cuda.memcpy_dtoh(host[t], bufs[t])
        for ptr in bufs.values():
            cuda.free(ptr)
        cuda.check(cuda.lib.cudaStreamDestroy(stream))
        return {t: host[t] for t in OUT_NAMES}


def stage_trt(out):
    meta = json.loads((out / "cases.json").read_text())
    data = np.load(out / "cases.npz")
    runner = PluginRunner()
    res = {}
    for m in meta:
        name = m["name"]
        got = runner.run(data[f"{name}.boxes"], data[f"{name}.scores"], m["params"])
        if got is None:
            print(f"trt: {name}: BUILD FAILED")
            continue
        res.update({f"{name}.trt.{t}": v for t, v in got.items()})
    np.savez(out / "trt.npz", **res)
    print(f"trt: ran {len(res) // 4} cases")


# -------------------------------------------------------------------- stage compare


def _rows(nd, bx, sc, cl, b):
    k = int(nd[b, 0])
    return [(int(round(float(cl[b, i]))), float(sc[b, i]), *[float(v) for v in bx[b, i]]) for i in range(k)]


def _key(r):
    return (-round(r[1], 4), r[0], *[round(v, 4) for v in r[2:]])


def _same(a, b, tol=1e-5):
    return len(a) == len(b) and all(x[0] == y[0] and np.allclose(x[1:], y[1:], atol=tol) for x, y in zip(a, b))


def _pad(nd, bx, sc, cl):
    vals = set()
    for b in range(nd.shape[0]):
        for i in range(int(nd[b, 0]), sc.shape[1]):
            vals.add((round(float(sc[b, i]), 4), round(float(cl[b, i]), 4), *[round(float(v), 4) for v in bx[b, i]]))
    return sorted(vals)


def stage_compare(out):
    meta = json.loads((out / "cases.json").read_text())
    ort_d, trt_d = np.load(out / "cases.npz"), np.load(out / "trt.npz")
    hdr = f"{'case':<22}{'nd':>4}{'set':>5}{'order':>7}{'pad':>5}   note"
    print(hdr + "\n" + "-" * len(hdr))
    details = []
    for m in meta:
        name = m["name"]
        if f"{name}.trt.nd" not in trt_d:
            print(f"{name:<22}  plugin did not run")
            continue
        if not m["rewritten"]:
            print(f"{name:<22}{'-':>4}{'-':>5}{'-':>7}{'-':>5}   pass DECLINED (graph untouched, plugin ran)  [{m['note']}]")
            continue
        r = {k: ort_d[f"{name}.ort.{k}"] for k in OUT_NAMES}
        t = {k: trt_d[f"{name}.trt.{k}"] for k in OUT_NAMES}
        if (t["nd"] == SENTINEL).any():
            print(f"{name:<22}  plugin left num_detections UNWRITTEN (invalid plugin config?)  [{m['note']}]")
            continue
        nd_ok = np.array_equal(r["nd"].reshape(-1), t["nd"].reshape(-1))
        set_ok = order_ok = True
        diff_lines = []
        for b in range(r["nd"].shape[0]):
            ra, ta = _rows(r["nd"], r["bx"], r["sc"], r["cl"], b), _rows(t["nd"], t["bx"], t["sc"], t["cl"], b)
            if not _same(sorted(ra, key=_key), sorted(ta, key=_key)):
                set_ok = False
                only_r = [x for x in ra if _key(x) not in {_key(y) for y in ta}]
                only_t = [x for x in ta if _key(x) not in {_key(y) for y in ra}]
                diff_lines.append(f"    batch {b}: rewrite-only {[(x[0], round(x[1], 4)) for x in only_r][:4]}"
                                  f"  plugin-only {[(x[0], round(x[1], 4)) for x in only_t][:4]}"
                                  f"  (nd {int(r['nd'][b, 0])} vs {int(t['nd'][b, 0])})")
            if not _same(ra, ta):
                order_ok = False
                if set_ok and not diff_lines:
                    diff_lines.append(f"    batch {b}: same set, different order:\n"
                                      f"      rewrite {[(x[0], x[2], round(x[1], 3)) for x in ra][:6]}\n"
                                      f"      plugin  {[(x[0], x[2], round(x[1], 3)) for x in ta][:6]}")
        pad_r, pad_t = _pad(r["nd"], r["bx"], r["sc"], r["cl"]), _pad(t["nd"], t["bx"], t["sc"], t["cl"])
        pad_ok = pad_r == pad_t
        if not pad_ok:
            diff_lines.append(f"    padding rewrite {pad_r[:2]} vs plugin {pad_t[:2]}  (score, class, box)")
        flag = lambda v: "ok" if v else "DIFF"  # noqa: E731
        print(f"{name:<22}{flag(nd_ok):>4}{flag(set_ok):>5}{flag(order_ok):>7}{flag(pad_ok):>5}   {m['note']}")
        details += [f"  {name}:"] + diff_lines if diff_lines else []
    if details:
        print("\nDetails of differences:\n" + "\n".join(details))


# ------------------------------------------------------------------------- driver


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["gen", "trt", "compare", "all"])
    ap.add_argument("out")
    ap.add_argument("--py-onnxsim", default=sys.executable, help="interpreter with onnxsim + onnxruntime")
    ap.add_argument("--py-trt", default=sys.executable, help="interpreter with tensorrt")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.stage == "all":
        for py, stage in ((a.py_onnxsim, "gen"), (a.py_trt, "trt")):
            subprocess.run([py, str(Path(__file__).resolve()), stage, str(out)], check=True, cwd="/")
        stage_compare(out)
    else:
        {"gen": stage_gen, "trt": stage_trt, "compare": stage_compare}[a.stage](out)


if __name__ == "__main__":
    main()
