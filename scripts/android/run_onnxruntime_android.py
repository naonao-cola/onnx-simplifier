#!/usr/bin/env python3
"""Run an original/simplified ONNX pair with ONNX Runtime on Android."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/android/runner"
ANDROID_APP = ROOT / "scripts/android/app"
def run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture else ""


def make_models(directory: Path) -> tuple[Path, Path, Path]:
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["X"], ["R"]),
            helper.make_node("Identity", ["R"], ["T"]),
            helper.make_node("Identity", ["T"], ["Y"]),
        ],
        "relu_smoke",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    )
    original = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.checker.check_model(original)
    simplified, valid = simplify(original)
    if not valid:
        raise RuntimeError("onnxsim validation failed for the Android smoke model")
    if len(simplified.graph.node) >= len(original.graph.node):
        raise RuntimeError("onnxsim did not remove the redundant Identity nodes")
    if not any(node.op_type == "Relu" for node in simplified.graph.node):
        raise RuntimeError("onnxsim smoke graph lost its Relu compute node")
    original_path = directory / "original.onnx"
    simplified_path = directory / "simplified.onnx"
    input_path = directory / "input.f32"
    onnx.save(original, original_path)
    onnx.save(simplified, simplified_path)
    np.array([-2.5, -0.25, 0.75, 4.0], dtype=np.float32).tofile(input_path)
    return original_path, simplified_path, input_path


def make_qnn_htp_models(directory: Path) -> tuple[Path, Path, Path]:
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="scale")
    input_zero_point = numpy_helper.from_array(
        np.array(128, dtype=np.uint8), name="input_zero_point"
    )
    output_zero_point = numpy_helper.from_array(
        np.array(0, dtype=np.uint8), name="output_zero_point"
    )
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["X", "scale", "input_zero_point"], ["Xq"]),
            helper.make_node("DequantizeLinear", ["Xq", "scale", "input_zero_point"], ["Xdq"]),
            helper.make_node("Relu", ["Xdq"], ["R"]),
            helper.make_node("QuantizeLinear", ["R", "scale", "output_zero_point"], ["Yq"]),
            helper.make_node("DequantizeLinear", ["Yq", "scale", "output_zero_point"], ["Y"]),
        ],
        "quantized_relu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        [scale, input_zero_point, output_zero_point],
    )
    original = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.checker.check_model(original)
    simplified, valid = simplify(original)
    if not valid:
        raise RuntimeError("onnxsim validation failed for the quantized QNN smoke graph")
    original_path = directory / "qnn_original.onnx"
    simplified_path = directory / "qnn_simplified.onnx"
    input_path = directory / "qnn_input.f32"
    onnx.save(original, original_path)
    onnx.save(simplified, simplified_path)
    np.array([-2.5, -0.5, 0.5, 4.0], dtype=np.float32).tofile(input_path)
    return original_path, simplified_path, input_path


def select_device(adb: str, serial: str | None) -> str:
    lines = run([adb, "devices"], capture=True).splitlines()[1:]
    devices = [line.split()[0] for line in lines if len(line.split()) > 1 and line.split()[1] == "device"]
    if serial:
        if serial not in devices:
            raise RuntimeError(f"ADB device {serial!r} is not connected and authorized")
        return serial
    if len(devices) != 1:
        raise RuntimeError(
            f"expected exactly one authorized Android device, found {len(devices)}; pass --serial"
        )
    return devices[0]


def gradle_executable() -> str:
    cached = sorted(Path.home().glob(
        ".gradle/wrapper/dists/gradle-*-bin/*/gradle-*/bin/gradle"
    ))
    return str(cached[-1]) if cached else "gradle"


def run_qnn_in_app(work: Path, sdk: Path, ndk: Path, adb_prefix: list[str],
                   runtime_lib: Path, include_dir: Path, qnn_library: Path,
                   original: Path, simplified: Path, input_file: Path,
                   target: str) -> tuple[bool, str]:
    """Run a QNN probe in Android's app linker namespace, where vendor libraries are visible."""
    project = work / "android-app"
    shutil.copytree(ANDROID_APP, project, dirs_exist_ok=True)
    app_main = project / "app/src/main"
    jni_libs = app_main / "jniLibs/arm64-v8a"
    headers = app_main / "cpp/headers"
    assets = app_main / "assets"
    jni_libs.mkdir(parents=True, exist_ok=True)
    headers.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime_lib, jni_libs / "libonnxruntime.so")
    shutil.copy2(qnn_library, jni_libs / "libonnxruntime_providers_qnn.so")
    shutil.copytree(include_dir, headers, dirs_exist_ok=True)
    for source, asset_name in zip(
        (original, simplified, input_file),
        ("original.onnx", "simplified.onnx", "input.f32"),
    ):
        shutil.copy2(source, assets / asset_name)
    (project / "local.properties").write_text(
        f"sdk.dir={sdk}\nndk.dir={ndk}\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["ANDROID_HOME"] = str(sdk)
    env["ANDROID_SDK_ROOT"] = str(sdk)
    build = subprocess.run(
        [gradle_executable(), "--offline", "-p", str(project),
         f"-PonnxsimNdkVersion={ndk.name}", "assembleDebug"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    if build.returncode != 0:
        return False, "Android app build failed:\n" + build.stdout[-4000:]
    apk = project / "app/build/outputs/apk/debug/app-debug.apk"
    package = "org.onnxsim.androidtest"
    install = subprocess.run([*adb_prefix, "install", "-r", str(apk)],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if install.returncode != 0:
        return False, "APK install failed: " + install.stdout[-2000:]
    subprocess.run([*adb_prefix, "shell", "am", "force-stop", package],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    launch = subprocess.run(
        [*adb_prefix, "shell", "am", "start", "-W", "-n",
         f"{package}/.MainActivity", "--es", "target", target],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if launch.returncode != 0:
        return False, "Activity launch failed: " + launch.stdout[-2000:]
    result_path = f"files/result_{target}.txt"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = subprocess.run([*adb_prefix, "shell", "run-as", package,
                                 "cat", result_path], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and result.stdout.strip():
            content = result.stdout.strip()
            return content.startswith("PASS "), content
        time.sleep(0.25)
    return False, "Android test app produced no result (native crash or timeout)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-aar", required=True, type=Path)
    ap.add_argument("--qnn-aar", type=Path,
                    help="Qualcomm ONNX Runtime QNN provider AAR; enables HTP and GPU probes")
    ap.add_argument("--require-htp", action="store_true",
                    help="fail unless both original and simplified models run on QNN HTP")
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail unless both original and simplified models run on QNN GPU")
    ap.add_argument("--require-nnapi-hw", action="store_true",
                    help="fail unless both models pass NNAPI with its CPU device disabled")
    ap.add_argument("--require-nnapi-dsp", action="store_true",
                    help="fail unless a direct NNAPI RELU compiles and runs on qti-dsp")
    ap.add_argument("--android-sdk", type=Path, default=os.environ.get("ANDROID_HOME"))
    ap.add_argument("--ndk-version", default="27.2.12479018")
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--serial")
    ap.add_argument("--work-dir", type=Path)
    args = ap.parse_args()

    if args.android_sdk is None:
        ap.error("--android-sdk or ANDROID_HOME is required")
    sdk = args.android_sdk.expanduser().resolve()
    ndk = sdk / "ndk" / args.ndk_version
    toolchain = ndk / "build/cmake/android.toolchain.cmake"
    if not toolchain.is_file():
        ap.error(f"Android NDK toolchain not found: {toolchain}")
    if not args.runtime_aar.is_file():
        ap.error(f"Android ONNX Runtime AAR not found: {args.runtime_aar}")
    if args.qnn_aar is not None and not args.qnn_aar.is_file():
        ap.error(f"Android ONNX Runtime QNN AAR not found: {args.qnn_aar}")
    if (args.require_htp or args.require_gpu or args.require_nnapi_hw or
            args.require_nnapi_dsp) and args.qnn_aar is None:
        ap.error("accelerator requirements need --qnn-aar for the Android probe app")

    serial = select_device(args.adb, args.serial)
    work_ctx = tempfile.TemporaryDirectory(prefix="onnxsim-android-") if args.work_dir is None else None
    work = Path(work_ctx.name) if work_ctx else args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    extracted = work / "ort-aar"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.runtime_aar) as aar:
        names = set(aar.namelist())
        runtime_name = "jni/arm64-v8a/libonnxruntime.so"
        if runtime_name not in names or not any(n.startswith("headers/") for n in names):
            raise RuntimeError(f"{args.runtime_aar} has no arm64 ONNX Runtime library and headers")
        for name in names:
            if name.startswith("headers/") or name == runtime_name:
                aar.extract(name, extracted)
    qnn_library = None
    if args.qnn_aar is not None:
        qnn_library_name = "jni/arm64-v8a/libonnxruntime_providers_qnn.so"
        qnn_extracted = work / "qnn-aar"
        qnn_extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(args.qnn_aar) as aar:
            if qnn_library_name not in aar.namelist():
                ap.error(f"{args.qnn_aar} has no arm64 QNN execution provider library")
            qnn_library = Path(aar.extract(qnn_library_name, qnn_extracted))

    model_a, model_b, input_file = make_models(work)
    qnn_model_a, qnn_model_b, qnn_input_file = make_qnn_htp_models(work)
    build = work / "build"
    runtime_lib = extracted / "jni/arm64-v8a/libonnxruntime.so"
    include_dir = extracted / "headers"
    run([
        "cmake", "-S", str(RUNNER), "-B", str(build),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-26", f"-DORT_INCLUDE_DIR={include_dir}",
        f"-DORT_LIBRARY={runtime_lib}",
    ])
    run(["cmake", "--build", str(build), "--config", "Release", "--parallel"])
    binary = build / "onnxsim_android_runner"

    remote = f"/data/local/tmp/onnxsim_android_{os.getpid()}"
    adb_prefix = [args.adb, "-s", serial]
    run([*adb_prefix, "shell", "mkdir", "-p", remote])
    try:
        files_to_push = [
            (binary, "runner"), (runtime_lib, "libonnxruntime.so"),
            (model_a, "original.onnx"), (model_b, "simplified.onnx"),
            (input_file, "input.f32"),
        ]
        for host, name in files_to_push:
            run([*adb_prefix, "push", str(host), f"{remote}/{name}"])
        expected = np.maximum(np.fromfile(input_file, dtype=np.float32), 0)

        def invoke(model_name: str, output_name: str, target: str | None = None) -> tuple[bool, str]:
            args_list = [f"./runner", model_name, "input.f32", output_name]
            if target is not None:
                args_list.extend((target, "./libonnxruntime_providers_qnn.so"))
            shell_command = (
                f"cd {remote} && chmod 755 runner && "
                f"LD_LIBRARY_PATH={remote} " + " ".join(args_list)
            )
            result = subprocess.run(
                [*adb_prefix, "shell", shell_command], text=True, capture_output=True
            )
            return result.returncode == 0, (result.stdout + result.stderr).strip()

        def get_output(output_name: str) -> np.ndarray:
            host_output = work / output_name
            run([*adb_prefix, "pull", f"{remote}/{output_name}", str(host_output)])
            return np.fromfile(host_output, dtype=np.float32)

        cpu_outputs = []
        for model_name, output_name in (("original.onnx", "original_cpu.out"),
                                        ("simplified.onnx", "simplified_cpu.out")):
            ok, detail = invoke(model_name, output_name)
            if not ok:
                raise RuntimeError(f"Android CPU run failed: {detail}")
            cpu_outputs.append(get_output(output_name))
        if not all(out.shape == expected.shape and np.allclose(out, expected, rtol=1e-5, atol=1e-6)
                   for out in cpu_outputs):
            raise RuntimeError(f"Android CPU output mismatch: expected {expected}, got {cpu_outputs}")
        print(f"PASS Android ONNX Runtime CPU: original == simplified == reference on {serial}")

        if qnn_library is None:
            print("SKIP QNN HTP/GPU: pass --qnn-aar to enable hardware runs")
        else:
            for target, required in (("qnn-htp", args.require_htp),
                                     ("qnn-gpu", args.require_gpu),
                                     ("nnapi-no-cpu", args.require_nnapi_hw),
                                     ("nnapi-dsp-direct", args.require_nnapi_dsp)):
                if target == "qnn-htp":
                    target_model_a, target_model_b, target_input = (
                        qnn_model_a, qnn_model_b, qnn_input_file
                    )
                else:
                    target_model_a, target_model_b, target_input = model_a, model_b, input_file
                ok, detail = run_qnn_in_app(
                    work, sdk, ndk, adb_prefix, runtime_lib, include_dir,
                    qnn_library, target_model_a, target_model_b, target_input, target,
                )
                if not ok:
                    print(f"{'FAIL' if required else 'SKIP'} {target}: {detail[-1500:]}")
                    if required:
                        raise RuntimeError(f"required {target} test failed")
                    continue
                print(f"{detail} on {serial}")
        if qnn_library is not None and not args.require_htp and not args.require_gpu:
            print("Use --require-htp and/or --require-gpu to make backend skips fail the command.")
    finally:
        subprocess.run([*adb_prefix, "shell", "rm", "-rf", remote], check=False)
    if work_ctx:
        work_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
