"""MCC on the phone: export, phone runs, scoring.

  mcc.py export  --ckpt <pth>             enc.onnx + dec_q<Q>.onnx (fp32) -> onnxsim; ORT CPU vs torch
  mcc.py phone   <piece> [--iters N]      run one piece strict all-HTP (fp16) under the phone lock
  mcc.py recon   --gran G [--strategy S]  whole reconstruction on the phone -> score vs ref_<g>.npz

Work dir: $MCC_WORK (default ~/.cache/onnxsim-mcc/work). Upstream code: $MCC_REPO.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

WORK = Path(os.environ.get("MCC_WORK", Path.home() / ".cache/onnxsim-mcc/work"))
LOCK = [str(Path.home() / ".cache/android-phone/phone-run")]


def simplify(src, dst):
    import onnx

    import onnxsim

    m, ok = onnxsim.simplify(onnx.load(src), skipped_optimizers=["fuse_attention"])
    assert ok, src
    odd = {
        (x.domain, x.op_type) for x in m.graph.node if x.domain not in ("", "ai.onnx")
    }
    assert not odd, f"{src}: non-standard ops after onnxsim: {odd}"
    onnx.save(m, dst)
    Path(src).unlink()


def ort_run(path, feeds):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return s.run(None, {k: np.ascontiguousarray(v) for k, v in feeds.items()})


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def export(a):
    torch.set_grad_enabled(False)
    WORK.mkdir(parents=True, exist_ok=True)
    m = M.load_mcc(a.ckpt)
    inp = np.load(WORK / "inputs_quest2.npz")
    img, win, val = (torch.from_numpy(inp[k]) for k in ("img", "xyz_win", "valid"))
    enc = M.Encoder(m).eval()
    k, v = enc(img, win, val)
    torch.onnx.export(
        enc,
        (img, win, val),
        WORK / "enc_g.onnx",
        input_names=["img", "xyz_win", "valid"],
        output_names=["k", "v"],
        opset_version=20,
        dynamo=False,
    )
    simplify(WORK / "enc_g.onnx", WORK / "enc.onnx")
    ok, ov = ort_run(
        WORK / "enc.onnx",
        {"img": img.numpy(), "xyz_win": win.numpy(), "valid": val.numpy()},
    )
    print(
        f"enc ORT vs torch: k max {np.abs(ok - k.numpy()).max():.3g}, v max {np.abs(ov - v.numpy()).max():.3g}"
    )
    np.savez(WORK / "kv_quest2.npz", k=k.numpy(), v=v.numpy())
    dec = M.QueryDecoder(m).eval()
    pts = M.grid(0.1)
    for q in a.chunks:
        x = pts[
            :, pts.shape[1] // 2 : pts.shape[1] // 2 + q
        ]  # grid middle: some occupied points
        occ, rgb = dec(x, k, v)
        name = f"dec_q{q}"
        torch.onnx.export(
            dec,
            (x, k, v),
            WORK / f"{name}_g.onnx",
            input_names=["xyz", "k", "v"],
            output_names=["occ", "rgb"],
            opset_version=20,
            dynamo=False,
        )
        simplify(WORK / f"{name}_g.onnx", WORK / f"{name}.onnx")
        oo, orr = ort_run(
            WORK / f"{name}.onnx", {"xyz": x.numpy(), "k": k.numpy(), "v": v.numpy()}
        )
        print(
            f"{name} ORT vs torch: occ max {np.abs(oo - occ.numpy()).max():.3g}, "
            f"rgb max {np.abs(orr - rgb.numpy()).max():.3g}"
        )


def write_raw(d, name, arr, dtype="f32"):
    arr = np.ascontiguousarray(
        arr.astype({"f32": np.float32, "u8": np.uint8, "u16": np.uint16}[dtype])
    )
    arr.tofile(d / f"{name}.bin")
    return f"{name} {dtype} {name}.bin {','.join(map(str, arr.shape))}"


def phone(model_path, name, feeds, iters, loc, mode="htp"):
    """feeds: list of (input name, array, dtype). Returns (median ms, [output arrays])."""
    loc.mkdir(parents=True, exist_ok=True)
    man = [write_raw(loc, n, arr, dt) for n, arr, dt in feeds]
    (loc / "m0.txt").write_text("\n".join(man) + "\n")
    env = dict(os.environ, PHONE_LOCK_OWNER="codex/android-mcc")
    out = subprocess.run(
        LOCK
        + [
            str(HERE / "phone.sh"),
            str(model_path),
            mode,
            str(iters),
            str(loc),
            name,
            "m0.txt",
        ],
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    ms = None
    outs = []
    for line in out.splitlines():
        if line.startswith("median_ms"):
            ms = float(line.split()[1])
        if line.startswith("out "):
            _, i, _, dt, shape = line.split()[:5]
            dims = [
                int(t) for t in shape.strip("[]()").replace("x", ",").split(",") if t
            ]
            arr = np.fromfile(
                loc / f"out0_o{i}.bin",
                dtype={"f32": np.float32, "u8": np.uint8, "u16": np.uint16}[dt],
            )
            outs.append(
                arr.reshape(dims) if dims and np.prod(dims) == arr.size else arr
            )
    if "PASS" not in out:
        print(out[-3000:])
    return ms, outs


def cmd_phone(a):
    inp = np.load(WORK / "inputs_quest2.npz")
    kv = np.load(WORK / "kv_quest2.npz")
    if a.piece == "enc":
        feeds = [
            ("img", inp["img"], "f32"),
            ("xyz_win", inp["xyz_win"], "f32"),
            ("valid", inp["valid"], "f32"),
        ]
        ref = [kv["k"], kv["v"]]
    else:
        q = int(a.piece.split("q")[1])
        pts = M.grid(0.1)
        x = pts[:, pts.shape[1] // 2 : pts.shape[1] // 2 + q].numpy()
        feeds = [("xyz", x, "f32"), ("k", kv["k"], "f32"), ("v", kv["v"], "f32")]
        m = M.load_mcc(a.ckpt)
        with torch.no_grad():
            o, r = M.QueryDecoder(m)(
                torch.from_numpy(x),
                torch.from_numpy(kv["k"]),
                torch.from_numpy(kv["v"]),
            )
        ref = [o.numpy(), r.numpy()]
    ms, outs = phone(
        WORK / f"{a.piece}.onnx", a.piece, feeds, a.iters, WORK / "phone" / a.piece
    )
    print(
        f"{a.piece}: {ms} ms on the HTP; "
        + ", ".join(
            f"out{i} cos {cos(o, r):.6f} max {np.abs(o.reshape(r.shape) - r).max():.3g}"
            for i, (o, r) in enumerate(zip(outs, ref))
        )
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--chunks", type=int, nargs="+", default=[2048, 4096, 8192])
    p = sub.add_parser("phone")
    p.add_argument("piece")
    p.add_argument(
        "--ckpt",
        default=str(Path.home() / ".cache/onnxsim-mcc/co3dv2_all_categories.pth"),
    )
    p.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()
    t = time.time()
    {"export": export, "phone": cmd_phone}[a.cmd](a)
    print(f"[{time.time() - t:.0f} s]")


if __name__ == "__main__":
    main()
