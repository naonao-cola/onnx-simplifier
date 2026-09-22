#!/usr/bin/env python3
"""Re-measure every 1x1-conv shape this project has previously claimed a speedup on, using
hex_gemm_signed_kernel.py (vrmpybusv, signed int8 weights matching the real backbone) instead of
hex_gemm_kernel.py (vrmpyub, unsigned synthetic weights -- the original measurement). See
hex_gemm_signed_kernel.py's own docstring for why this re-measurement is needed: every prior
1x1-conv coverage number in this project was measured against synthetic UNSIGNED weight data,
which the real network never actually has (its 1x1 conv weights are genuinely signed int8).

Stock TVM's own numbers are NOT re-measured here: TVM's compiled schedule has always operated
correctly on the real network's real signed weights (that was never in question -- only this
project's own hand-written kernel had the unsigned-only bug). So each shape's original stock-TVM
GMAC/s figure (hardcoded below, copied verbatim from README.md's existing tables) is reused as
the comparison baseline; only the custom-kernel side is re-measured, now with signed data.

Weight data: signed-random (`rng.integers(-128, 127, ...)`), matching the real weights'
*distribution* (bytes anywhere in int8's full range), not real extracted backbone weights --
extracting real weights for all 13 shapes was out of scope for the time available in this pass
(a couple of these shapes are also not exact single real layers, e.g. the "small-channel"/
"larger-channel" survey shapes are picked from the profile by size, not necessarily unique single
real convs). This directly tests whether vrmpybusv's different calling convention has a real,
measurable throughput cost vs vrmpyub -- the actual open question -- independent of extracting
real weights, which would change correctness data but not performance.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("DEV", "DSP")
os.environ.setdefault("MOCKDSP", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).parent

# (cin, cout_real, spatial(H,W), stride, stock_tvm_gmacs, orig_speedup, label)
# cout_real < 32 means "pad to 32, only cout_real columns are meaningful" (matches the tiny
# RPN/mask-head coverage's own established padding trick).
SHAPES = [
    (64, 256, (200, 272), 1, 3.51, "8.65x", "flagship (It works / small-channel #1)"),
    (128, 512, (100, 136), 1, 6.03, "6.41x", "small-channel #2"),
    (64, 64, (200, 272), 1, 2.88, "4.99x", "small-channel #3"),
    (256, 64, (200, 272), 1, 7.60, "2.39x", "small-channel #4"),
    (512, 128, (100, 136), 1, 12.72, "2.00x", "small-channel #5"),
    (256, 128, (200, 272), 2, 5.38, "4.62x", "strided"),
    (256, 12, (200, 272), 1, 2.00, "1.94x", "tiny RPN cout=12 (padded to 32)"),
    (256, 3, (200, 272), 1, 0.80, "1.22x", "tiny mask-head cout=3 (padded to 32)"),
    (256, 512, (200, 272), 2, 7.54, "5.81x", "larger-channel #1 (strided)"),
    (256, 256, (200, 272), 1, 9.78, "3.89x", "larger-channel #2"),
    (512, 256, (100, 136), 1, 14.40, "2.55x", "larger-channel #3"),
    (256, 1024, (50, 68), 1, 20.10, "2.22x", "larger-channel #4"),
    (1024, 256, (50, 68), 1, 24.43, "1.44x", "larger-channel #5"),
]


def capture(cin: int, cout_pad: int, m: int, stride: int, ih: int, iw: int, seed: int, idx: int = 0) -> tuple[str, str, bool]:
    """Build+capture the signed kernel's C source for one shape, verifying qemu correctness
    against a numpy reference with SIGNED weight data. Returns (kernel_src, kernel_name, correct)."""
    import hex_gemm_signed_kernel as sk
    from hex_gemm_kernel import pack_b
    from tinygrad import Tensor
    from tinygrad.renderer.cstyle import ClangRenderer

    captured: dict[str, str] = {}
    orig_render = ClangRenderer.render

    def _capture(self, uops):
        src = orig_render(self, uops)
        captured["src"] = src
        return src

    ClangRenderer.render = _capture
    try:
        rng = np.random.default_rng(seed)
        kname = f"gsig_{cin}_{cout_pad}_{m}_{stride}_i{idx}"
        if stride == 1:
            a_np = rng.integers(0, 100, (m, cin)).astype(np.uint8)
            w_np = rng.integers(-128, 127, (cin, cout_pad)).astype(np.int8)
            a = Tensor(a_np, device="DSP")
            bp = Tensor(pack_b(w_np.astype(np.uint8)), device="DSP")
            out = sk.build_kernel(cin, cout_pad, m, a, bp, kernel_name=kname)
            ref = a_np.astype(np.int64) @ w_np.astype(np.int64)
        else:
            a_np = rng.integers(0, 100, (ih * iw, cin)).astype(np.uint8)
            w_np = rng.integers(-128, 127, (cin, cout_pad)).astype(np.int8)
            a = Tensor(a_np, device="DSP")
            bp = Tensor(pack_b(w_np.astype(np.uint8)), device="DSP")
            out = sk.build_strided_kernel(cin, cout_pad, ih, iw, stride, a, bp, kernel_name=kname)
            oh, ow = ih // stride, iw // stride
            a_sub = a_np.reshape(ih, iw, cin)[::stride, ::stride, :].reshape(oh * ow, cin)
            ref = a_sub.astype(np.int64) @ w_np.astype(np.int64)
        out.realize()
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
    finally:
        ClangRenderer.render = orig_render
    src = captured["src"]
    marker = src.find("/* DSP boilerplate */")
    kernel_src = (src[:marker] if marker >= 0 else src).rstrip() + "\n"
    if f"void {kname}(" not in kernel_src:
        raise ValueError(f"couldn't find kernel function {kname} in captured source")
    return kernel_src, kname, correct


def compile_kernel(hexagon_clang, kernel_c, out_o, hex_arch):
    subprocess.run(
        [str(hexagon_clang), "-c", "-O2", "-Wall", "-fno-stack-protector", "-x", "c", "-fPIC",
         "-ffreestanding", f"-mcpu=hexagon{hex_arch}", f"-mhvx={hex_arch}", "-mhvx-length=128b",
         "-o", str(out_o), str(kernel_c)], check=True)


def compile_wrapper(hexagon_clang, wrapper_c, out_o, hex_arch, tvm_root):
    subprocess.run(
        [str(hexagon_clang), "-c", "-O2", "-fPIC", f"-mcpu=hexagon{hex_arch}", f"-mhvx={hex_arch}",
         "-mhvx-length=128b", "-I", str(tvm_root / "include"),
         "-I", str(tvm_root / "3rdparty" / "dlpack" / "include"), "-o", str(out_o), str(wrapper_c)],
        check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hexagon-toolchain", type=Path, required=True)
    p.add_argument("--tvm-root", type=Path, required=True)
    p.add_argument("--device", default="239dbd8f")
    p.add_argument("--hex-arch", default="v73")
    p.add_argument("--rpc-port", type=int, default=9231)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--workdir", type=Path, default=Path("reverify_work"))
    p.add_argument("--only", default=None, help="comma-separated shape indices to run (default: all)")
    args = p.parse_args()
    only = {int(x) for x in args.only.split(",")} if args.only else None
    args.workdir.mkdir(exist_ok=True)

    import tvm
    from tvm.contrib.hexagon.build import HexagonLauncher
    from tvm.contrib.hexagon.tools import link_shared
    from tvm.rpc.tracker import Tracker

    hexagon_clang = args.hexagon_toolchain / "bin" / "hexagon-clang"
    template = (HERE / "wrapper_template.c").read_text()

    built = []
    for i, (cin, cout_real, (ih, iw), stride, tvm_gmacs, orig_speedup, label) in enumerate(SHAPES):
        if only is not None and i not in only:
            continue
        cout_pad = ((cout_real + 31) // 32) * 32
        m = (ih // stride) * (iw // stride)
        print(f"=== [{i}] {label}: cin={cin} cout={cout_real}(pad {cout_pad}) {ih}x{iw} stride={stride} m={m} ===")
        kernel_src, kname, correct = capture(cin, cout_pad, m, stride, ih, iw, seed=5 + i, idx=i)
        print(f"  qemu correctness: {correct}")
        if not correct:
            print("  SKIPPING real-hardware run -- qemu correctness failed")
            continue
        kernel_c = args.workdir / f"kernel_{i}.c"
        kernel_c.write_text(kernel_src)
        kernel_o = args.workdir / f"kernel_{i}.o"
        wrapper_c = args.workdir / f"wrapper_{i}.c"
        wrapper_o = args.workdir / f"wrapper_{i}.o"
        so_path = args.workdir / f"bridge_{i}.so"
        wrapper_name = f"wrap_{i}"
        compile_kernel(hexagon_clang, kernel_c, kernel_o, args.hex_arch)
        wrapper_c.write_text(template.replace("KERNEL_NAME", kname).replace("WRAPPER_NAME", wrapper_name))
        compile_wrapper(hexagon_clang, wrapper_c, wrapper_o, args.hex_arch, args.tvm_root)
        link_shared(str(so_path), [str(kernel_o), str(wrapper_o)], {"hex_arch": args.hex_arch})
        built.append((i, cin, cout_real, cout_pad, ih, iw, stride, m, tvm_gmacs, orig_speedup, label, so_path, wrapper_name))

    target = tvm.target.hexagon(args.hex_arch)  # noqa: F841
    tracker = Tracker(host="127.0.0.1", port=args.rpc_port, port_end=args.rpc_port + 1)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={"rpc_tracker_host": "127.0.0.1", "rpc_tracker_port": args.rpc_port,
                  "rpc_server_port": args.rpc_port - 1120, "workspace_base": "/data/local/tmp/tg_hex_signed_reverify",
                  "adb_server_socket": None},
    )
    results = []
    try:
        launcher.start_server()
        with launcher.create_session() as session:
            for (i, cin, cout_real, cout_pad, ih, iw, stride, m, tvm_gmacs, orig_speedup, label, so_path, wrapper_name) in built:
                remote_path = session.upload(str(so_path), so_path.name)
                mod = session.load_module(str(remote_path))
                fn = mod.get_function(wrapper_name)
                dev = session.device
                rng = np.random.default_rng(100 + i)
                if stride == 1:
                    a_np = rng.integers(0, 100, (m, cin)).astype(np.uint8)
                else:
                    a_np = rng.integers(0, 100, (ih * iw, cin)).astype(np.uint8)
                a = tvm.nd.array(a_np, device=dev)
                # packed-weight buffer shape: (cout_pad//32, cin//4, 128), matching pack_b()'s output
                b_np = rng.integers(0, 100, (cout_pad // 32, cin // 4, 128)).astype(np.uint8)
                b = tvm.nd.array(b_np, device=dev)
                out_rows = m if stride == 1 else (ih // stride) * (iw // stride)
                out = tvm.nd.array(np.zeros((out_rows, cout_pad), dtype=np.int32), device=dev)
                fn(out, a, b)
                times = []
                for _ in range(args.repeat):
                    t0 = time.time()
                    fn(out, a, b)
                    times.append(time.time() - t0)
                median = float(np.median(times))
                macs = out_rows * cin * cout_real
                gmacs = macs / median / 1e9
                new_speedup = gmacs / tvm_gmacs
                print(f"[{i}] {label}: median={median*1e3:.3f}ms  {gmacs:.2f} GMAC/s (signed)  "
                      f"vs TVM {tvm_gmacs:.2f} GMAC/s -> {new_speedup:.2f}x  (orig unsigned claim: {orig_speedup})")
                results.append((label, cin, cout_real, ih, iw, stride, tvm_gmacs, gmacs, new_speedup, orig_speedup))
    finally:
        launcher.stop_server()
        tracker.terminate()

    print("\n=== SUMMARY ===")
    print(f"{'label':45s} {'cin':>5s} {'cout':>5s} {'spatial':>10s} {'str':>3s} {'TVM':>7s} {'signed':>7s} {'new':>7s} {'orig':>7s}")
    for (label, cin, cout, ih, iw, stride, tvm_gmacs, gmacs, new_speedup, orig_speedup) in results:
        print(f"{label:45s} {cin:5d} {cout:5d} {f'{ih}x{iw}':>10s} {stride:3d} {tvm_gmacs:7.2f} {gmacs:7.2f} {new_speedup:6.2f}x {orig_speedup:>7s}")


if __name__ == "__main__":
    main()
