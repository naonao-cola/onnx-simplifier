#!/usr/bin/env python3
"""Sweep the NCHW output-width tile used by the TVM Hexagon conv probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tvm
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import register_linker
from tvm.rpc.tracker import Tracker

from test_tvm_hexagon_maskrcnn import _conv_module, _hexagon_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--tiles", default="4,8,16,32")
    parser.add_argument("--ops", default="1x1,3x3", help="Comma-separated workload suffixes")
    args = parser.parse_args()
    # Newer Qualcomm LLVM toolchains ship the clang driver but not TVM's
    # legacy `hexagon-link` executable. Let clang drive the same link step.
    toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
    if toolchain:
        clang_link = Path(toolchain) / "bin" / "hexagon-clang++"
        wrapper = Path("/tmp/tvm-hexagon-link-wrapper")
        wrapper.write_text(
            "#!/usr/bin/env python3\n"
            "import os, subprocess, sys\n"
            f"clang = {str(clang_link)!r}\n"
            "args = ['-Wl,--export-dynamic' if x == '-export-dynamic' else x for x in sys.argv[1:]]\n"
            "raise SystemExit(subprocess.call([clang, *args]))\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        register_linker(lambda: str(wrapper))
    tiles = [int(value) for value in args.tiles.split(",")]
    selected_ops = tuple(args.ops.split(","))
    workloads = [
        ("resnet_bottleneck_1x1", (1, 64, 56, 56), (64, 64, 1, 1), (1, 1), (0, 0)),
        ("resnet_bottleneck_3x3", (1, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1)),
    ]
    workloads = [workload for workload in workloads if workload[0].endswith(selected_ops)]
    target = _hexagon_target()
    tracker = Tracker(host="127.0.0.1", port=9194)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9194,
            "rpc_server_port": 7074,
            "workspace_base": "/data/local/tmp/tvm_hexagon_width_tiles",
            "adb_server_socket": None,
        },
    )
    rng = np.random.default_rng(31)
    try:
        launcher.start_server()
        for name, data_shape, weight_shape, stride, padding in workloads:
            x_np = rng.normal(0, 0.1, data_shape).astype("float32")
            w_np = rng.normal(0, 0.05, weight_shape).astype("float32")
            b_np = rng.normal(0, 0.02, (weight_shape[0],)).astype("float32")
            refmod, out_shape = _conv_module(
                name + "_reference", data_shape, weight_shape, stride, padding, padding,
                tvm.target.Target("llvm -mcpu=native"), width_tile=8,
            )
            refout = tvm.nd.empty(out_shape)
            refmod["main"](*(tvm.nd.array(v) for v in (x_np, w_np, b_np)), refout)
            expected = refout.numpy()
            for tile in tiles:
                mod, _ = _conv_module(
                    name, data_shape, weight_shape, stride, padding, padding, target,
                    width_tile=tile,
                )
                path = Path("/tmp") / f"tvm_hexagon_{name}_w{tile}.so"
                mod.save(str(path))
                with launcher.create_session() as session:
                    dsp = session.device
                    remote = session.load_module(session.upload(str(path), path.name))
                    dargs = [tvm.nd.array(v, dsp) for v in (x_np, w_np, b_np)]
                    dout = tvm.nd.empty(out_shape, "float32", dsp)
                    remote["main"](*dargs, dout)
                    got = dout.numpy()
                    max_error = float(np.max(np.abs(got - expected)))
                    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)
                    samples = remote.time_evaluator(
                        "main", dsp, number=1, repeat=5
                    )(*dargs, dout).results
                    print(
                        f"{name} width_tile={tile:2d} median_ms="
                        f"{np.median(samples) * 1e3:.3f} max_abs_err={max_error:.3g}",
                        flush=True,
                    )
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
