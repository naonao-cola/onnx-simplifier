"""Monocular geometry for MCC's seen points: MoGe-2 ViT-S (MIT) -> metric point map.

  python depth.py static  --onnx moge.onnx --h 640 --w 480 --tokens 1200   static-shape ONNX + ORT check
  python depth.py mcc     --ckpt <mcc pth> --work <dir>                     MCC fed MoGe points vs iPhone points

The official ONNX (Ruicheng/moge-2-vits-normal-onnx) takes a dynamic image and a num_tokens scalar;
the HTP wants static shapes, so num_tokens becomes a constant and the image shape is fixed, then
onnxsim folds the shape arithmetic.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402


def static(a):
    import onnx
    from onnx import numpy_helper

    import onnxsim

    m = onnx.load(a.onnx)
    g = m.graph
    (nt,) = [i for i in g.input if i.name == "num_tokens"]
    g.input.remove(nt)
    g.initializer.append(
        numpy_helper.from_array(np.array(a.tokens, np.int64), "num_tokens")
    )
    ms, ok = onnxsim.simplify(
        m,
        overwrite_input_shapes={"image": [1, 3, a.h, a.w]},
        skipped_optimizers=["fuse_attention"],
    )
    assert ok
    ms, n_gelu = fuse_gelu(ms)
    print(
        f"fused {n_gelu} erf-GELU chains into Gelu; Erf left: {sum(n.op_type == 'Erf' for n in ms.graph.node)}"
    )
    out = a.out or a.onnx.replace(".onnx", f".{a.h}x{a.w}.t{a.tokens}.onnx")
    onnx.save(ms, out)
    import onnxruntime as ort

    x = np.random.default_rng(0).random((1, 3, a.h, a.w), dtype=np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ref = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"]).run(
        None, {"image": x, "num_tokens": np.array(a.tokens, np.int64)}
    )
    got = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"]).run(
        None, {"image": x}
    )
    names = [o.name for o in ms.graph.output]
    for n, r, gg in zip(names, ref, got):
        print(f"{n}: {tuple(gg.shape)} max abs vs dynamic {np.abs(r - gg).max():.3g}")
    shapes = {(n.op_type) for n in ms.graph.node}
    print(f"static graph: {len(ms.graph.node)} nodes, ops {sorted(shapes)} -> {out}")


def fuse_gelu(m):
    """opset -> 20 and x*0.5*(1+erf(x/sqrt2)) / 0.5*x*(1+erf(..)) chains -> one Gelu node
    (QNN has no Erf; its Gelu is native)."""
    import onnx
    from onnx import helper, numpy_helper, version_converter

    m = version_converter.convert_version(m, 20)
    g = m.graph
    const = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def cval(name):
        v = const.get(name)
        return None if v is None or v.size != 1 else float(v.reshape(-1)[0])

    drop, add, at = set(), [], {}
    for erf in [n for n in g.node if n.op_type == "Erf"]:
        div = prod.get(erf.input[0])
        if div is None or div.op_type not in ("Div", "Mul"):
            continue
        c = cval(div.input[1])
        if c is None or not (
            abs(c - 2**0.5) < 1e-3 if div.op_type == "Div" else abs(c - 2**-0.5) < 1e-3
        ):
            continue
        x = div.input[0]
        (add1,) = cons[erf.output[0]]
        if (
            add1.op_type != "Add"
            or cval([i for i in add1.input if i != erf.output[0]][0]) != 1.0
        ):
            continue
        # the remaining multiplies by x and 0.5, in either order
        (m1,) = cons[add1.output[0]]
        chain = [m1]
        other = [i for i in m1.input if i != add1.output[0]][0]
        if other == x:  # (x*(1+erf)) * 0.5
            (m2,) = cons[m1.output[0]]
            if cval([i for i in m2.input if i != m1.output[0]][0]) != 0.5:
                continue
            chain.append(m2)
        else:  # (x*0.5) * (1+erf)
            mh = prod.get(other)
            if (
                mh is None
                or mh.op_type != "Mul"
                or x not in mh.input
                or cval([i for i in mh.input if i != x][0]) != 0.5
            ):
                continue
            chain.append(mh)
        last = chain[-1] if other == x else m1
        drop.update(id(t) for t in [div, erf, add1] + chain)
        node = helper.make_node("Gelu", [x], [last.output[0]], name=erf.name + "_gelu")
        add.append(node)
        at[id(last)] = (
            node  # emitted where the chain's last node was: stays topologically sorted
        )
    keep = [at.get(id(n), n) for n in g.node if id(n) not in drop or id(n) in at]
    del g.node[:]
    g.node.extend(keep)
    m = onnx.shape_inference.infer_shapes(m)
    from onnxsim import simplify

    m, ok = simplify(
        m, skipped_optimizers=["fuse_attention"]
    )  # also topologically re-sorts
    assert ok
    return m, len(add)


def moge_points(onnx_path, rgb, h, w):
    """rgb (H0,W0,3) uint8 -> MoGe points (h,w,3) and mask (h,w) at the model's static size."""
    import cv2
    import onnxruntime as ort

    x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    outs = dict(
        zip(
            [o.name for o in s.get_outputs()],
            s.run(None, {"image": x.transpose(2, 0, 1)[None]}),
        )
    )
    return outs["points"][0], outs["mask"][0] > 0.5


def mcc_with_moge(a):
    import cv2

    repo = os.environ.get("MCC_REPO", os.path.expanduser("~/.cache/onnxsim-mcc/MCC"))
    d = os.path.join(repo, "demo")
    rgb = cv2.imread(os.path.join(d, "quest2.jpg"))[..., ::-1].copy()
    H, W = rgb.shape[:2]
    pts, mmask = moge_points(a.onnx, rgb, a.h, a.w)
    # MoGe: OpenCV camera frame (x right, y down, z forward). MCC's demo point clouds (iPhone /
    # ARKit, and CO3D-trained): x right, y up, z toward the viewer -> flip y and z.
    pts = pts * np.array([1.0, -1.0, -1.0], np.float32)
    xyz = torch.nn.functional.interpolate(
        torch.from_numpy(pts).permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    seg = cv2.imread(os.path.join(d, "quest2_seg.png"), cv2.IMREAD_UNCHANGED)
    mask = torch.tensor(cv2.resize(seg, (W, H))).bool()
    if a.align:  # diagnostic: rotate MoGe into the iPhone cloud's frame (best per-pixel rotation)
        ip = np.array(
            [
                list(map(float, ln.split()[1:4]))
                for ln in open(os.path.join(d, "quest2.obj"))
                if ln.startswith("v ")
            ],
            np.float32,
        ).reshape(H, W, 3)
        x = xyz.numpy()
        mm = mask.numpy() & np.isfinite(ip).all(-1) & np.isfinite(x).all(-1)
        A, B = x[mm].astype(np.float64), ip[mm].astype(np.float64)
        A0, B0 = A - A.mean(0), B - B.mean(0)
        U, S, Vt = np.linalg.svd(A0.T @ B0)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = (U @ D @ Vt).T
        xyz = torch.from_numpy(
            (x.reshape(-1, 3) @ R.T).reshape(x.shape).astype(np.float32)
        )
    img, xyz112 = M.prep(torch.from_numpy(rgb.astype(np.float32) / 255), xyz, mask)
    m = M.load_mcc(a.ckpt, repo)
    enc, dec = M.Encoder(m).eval(), M.QueryDecoder(m).eval()
    with torch.no_grad():
        win, val = M.xyz_windows(xyz112)
        k, v = enc(img, win, val)
        pts_q = M.grid(0.1)
        occ = torch.cat(
            [
                dec(pts_q[:, s : s + 4096], k, v)[0][0]
                for s in range(0, pts_q.shape[1], 4096)
            ]
        )
    p = torch.sigmoid(occ).numpy()
    ref = np.load(os.path.join(a.work, "ref_quest2_0.1.npz"))
    pr = 1 / (1 + np.exp(-ref["occ"]))
    import queries as Qs

    a_pts, r_pts = ref["xyz"][p > 0.3], ref["xyz"][pr > 0.3]
    inter = ((p > 0.3) & (pr > 0.3)).sum()
    print(
        f"MCC with MoGe-2 points vs with iPhone points (granularity 0.1, p > 0.3): "
        f"{len(a_pts)} vs {len(r_pts)} occupied, IoU {inter / ((p > 0.3) | (pr > 0.3)).sum():.3f}, "
        f"chamfer {Qs.chamfer(a_pts, r_pts):.4f}"
    )
    np.savez(os.path.join(a.work, "moge_quest2.npz"), points=pts, mask=mmask, p=p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--onnx", required=True)
    s.add_argument("--h", type=int, default=640)
    s.add_argument("--w", type=int, default=480)
    s.add_argument("--tokens", type=int, default=1200)
    s.add_argument("--out")
    c = sub.add_parser("mcc")
    c.add_argument("--onnx", required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--work", required=True)
    c.add_argument("--h", type=int, default=640)
    c.add_argument("--w", type=int, default=480)
    c.add_argument("--align", action="store_true")
    a = ap.parse_args()
    {"static": static, "mcc": mcc_with_moge}[a.cmd](a)


if __name__ == "__main__":
    main()
