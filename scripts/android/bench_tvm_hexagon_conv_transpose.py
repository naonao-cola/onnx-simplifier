#!/usr/bin/env python3
"""Compare TVM's generic and parity-specialized stride-2 ConvTranspose on DSP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tvm
from tvm import te
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import register_linker
from tvm.rpc.tracker import Tracker

from test_tvm_hexagon_maskrcnn import _conv_transpose_module, _hexagon_target, _model_workloads


def _configure_linker():
    toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
    if not toolchain:
        return
    clang_link = Path(toolchain) / "bin" / "hexagon-clang++"
    wrapper = Path("/tmp/tvm-hexagon-link-wrapper")
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        f"clang = {str(clang_link)!r}\n"
        "args = ['-Wl,--export-dynamic' if x == '-export-dynamic' else x for x in sys.argv[1:]]\n"
        "raise SystemExit(subprocess.call([clang, *args]))\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    register_linker(lambda: str(wrapper))


def _direct_stride2_module(data_shape, weight_shape, bias_shape, target, width_tile):
    """Compute the four output parity planes without dilating and padding input."""
    n, in_channels, in_height, in_width = data_shape
    _, out_channels, kernel_h, kernel_w = weight_shape
    assert kernel_h == kernel_w == 2
    out_height, out_width = in_height * 2, in_width * 2
    data = te.placeholder(data_shape, name="data", dtype="float32")
    weight = te.placeholder(weight_shape, name="weight", dtype="float32")
    bias = te.placeholder(bias_shape, name="bias", dtype="float32")
    rc = te.reduce_axis((0, in_channels), name="rc")
    output = te.compute(
        (n, out_channels, out_height, out_width),
        lambda b, oc, y, x: te.sum(
            data[b, rc, y // 2, x // 2] * weight[rc, oc, y % 2, x % 2]
            + bias[oc] / in_channels,
            axis=rc,
        ),
        name="conv_transpose_direct",
    )
    schedule = te.create_schedule(output.op)
    batch, channel, height, width = schedule[output].op.axis
    width_outer, width_inner = schedule[output].split(width, factor=width_tile)
    outer = schedule[output].fuse(batch, channel, height, width_outer)
    schedule[output].reorder(outer, width_inner, *schedule[output].op.reduce_axis)
    schedule[output].vectorize(width_inner)
    schedule[output].parallel(outer)
    module = tvm.build(schedule, [data, weight, bias, output], target=target, name="main")
    return module, (n, out_channels, out_height, out_width)


def _numpy_stride2_reference(data, weight, bias):
    batch, _, height, width = data.shape
    _, out_channels, kh, kw = weight.shape
    result = np.empty((batch, out_channels, height * 2, width * 2), dtype="float32")
    for y in range(kh):
        for x in range(kw):
            result[:, :, y::2, x::2] = np.einsum(
                "nchw,co->nohw", data, weight[:, :, y, x], optimize=True
            )
    result += bias[None, :, None, None]
    return result


def _run_kernel(session, module_path, inputs, output_shape, expected, repeat):
    remote = session.load_module(session.upload(str(module_path), module_path.name))
    device = session.device
    device_inputs = [tvm.nd.array(value, device) for value in inputs]
    output = tvm.nd.empty(output_shape, "float32", device)
    remote["main"](*device_inputs, output)
    result = output.numpy()
    max_error = float(np.max(np.abs(result - expected)))
    np.testing.assert_allclose(result, expected, rtol=2e-3, atol=2e-3)
    times = remote.time_evaluator("main", device, number=1, repeat=repeat)(
        *device_inputs, output
    ).results
    return float(np.median(times) * 1e3), max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--tiles", default="4,8,16")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _configure_linker()

    _, _, _, _, workload, _ = _model_workloads(args.model, args.roi_batch)
    name, data_shape, weight_shape, stride, pads, output_padding = workload
    if stride != (2, 2) or pads not in ((0, 0), (0, 0, 0, 0)) or output_padding != (0, 0):
        raise ValueError("This specialization currently requires stride 2, zero padding, and no output padding")
    target = _hexagon_target()
    rng = np.random.default_rng(59)
    data = rng.normal(0, 0.1, data_shape).astype("float32")
    weight = rng.normal(0, 0.05, weight_shape).astype("float32")
    bias = rng.normal(0, 0.01, (weight_shape[1],)).astype("float32")
    inputs = [data, weight, bias]
    expected = _numpy_stride2_reference(data, weight, bias)
    tracker = Tracker(host="127.0.0.1", port=9197)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9197,
            "rpc_server_port": 7077,
            "workspace_base": "/data/local/tmp/tvm_hexagon_deconv_tune",
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        baseline, out_shape = _conv_transpose_module(
            data_shape, weight_shape, stride, pads, output_padding, target
        )
        assert out_shape == expected.shape
        baseline_path = Path("/tmp/tvm_hexagon_conv_transpose_topi.so")
        baseline.save(str(baseline_path))
        with launcher.create_session() as session:
            baseline_ms, baseline_error = _run_kernel(
                session, baseline_path, inputs, out_shape, expected, args.repeat
            )
        print(
            f"topi_transpose: median={baseline_ms:.3f} ms, "
            f"max_abs_err={baseline_error:.3g}",
            flush=True,
        )

        macs = data_shape[0] * weight_shape[1] * data_shape[2] * data_shape[3]
        macs *= data_shape[1] * weight_shape[2] * weight_shape[3]
        for tile in map(int, args.tiles.split(",")):
            module, out_shape = _direct_stride2_module(
                data_shape, weight_shape, (weight_shape[1],), target, tile
            )
            path = Path("/tmp") / f"tvm_hexagon_conv_transpose_direct_w{tile}.so"
            module.save(str(path))
            with launcher.create_session() as session:
                elapsed_ms, max_error = _run_kernel(
                    session, path, inputs, out_shape, expected, args.repeat
                )
            print(
                f"direct_parity width_tile={tile}: median={elapsed_ms:.3f} ms, "
                f"speedup={baseline_ms / elapsed_ms:.2f}x, "
                f"MACs={macs:,}, max_abs_err={max_error:.3g}",
                flush=True,
            )
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
