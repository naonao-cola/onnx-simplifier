"""ONNX -> K210 .kmodel, via nncase (Kendryte's own K210 KPU compiler).

Unlike ONNX -> TFLite (onnx-cardputer-flash's onnx_to_tflite_micro.py,
which wraps onnx2tf/TensorFlow), the K210-targeting nncase compiler is
installable straight from PyPI -- no source build, no conan, no C++
toolchain needed to *use* it (only to build it, which onnx-k210-flash's own
README discusses separately as the WASM-feasibility question). It only
runs on Python 3.10 or earlier -- nncase's last K210-compatible release
(1.9.0) ships wheels up to cp310 and no newer K210 build has followed
(nncase's later 2.x line dropped K210 for the K230/K510 chips):

    python3.10 -m venv .venv && .venv/bin/pip install nncase==1.9.0.20230322

Status: convert_to_kmodel() is exercised for real, not just reviewed --
this exact function's logic (shape inference, batch-dim pinning, the PTQ
Compiler sequence) was run against a real Hugging Face model
(ketiswp/mlcommons-ResNet8-CIFAR10-fp32-onnx) in the session that wrote
this file: it produced a 104792-byte kmodel (correct "KMDL" magic) that
then ran a real inference through nncase's own Simulator, producing a
(1, 10) softmax-shaped output. That's the strongest verification anything
in either onnx-cardputer-flash or onnx-k210-flash has had so far -- still
short of running on a real K210 board, which needs the firmware side this
tool doesn't have yet (see ../README.md's follow-ups).

Two real gotchas hit and fixed along the way, kept as code + comments here
rather than left for the next person to rediscover:

- nncase's ONNX importer wants every intermediate tensor's shape recorded
  as ONNX `value_info` -- plenty of exported models (this one included)
  don't carry it, and the error you get
  ("Can't find value info for ... to parse its shape") doesn't say to run
  shape inference. `onnx.shape_inference.infer_shapes()` fixes it.
- A dynamic/unset batch dimension (ONNX's `dim_param`, or a literal `0`
  `dim_value` some exporters use instead) makes nncase's internal
  shape-propagation compute a garbage size somewhere downstream --
  observed as a `RuntimeError: Shapes must be same` with one side showing
  an absurd number like `288230376151711743`, not an error naming the
  batch dimension at all. On-device inference is batch-1 anyway, so
  pinning the batch dimension to 1 before compiling (`_pin_batch_dim`
  below) avoids the whole class of failure rather than debugging each
  symptom of it.
- nncase 1.9.0's ONNX importer covers roughly half of the ai.onnx domain
  (see `onnx_legalizer.py`'s module docstring for the actual coverage
  check) and throws "Not supported ONNX opcode: ..." on the rest, even
  though several of those are just a named shorthand for a small
  expression built entirely out of ops it does support (`Gelu`, `Swish`,
  `Mish`, `MeanVarianceNormalization`, ...) -- increasingly common in
  Hugging Face exports. `legalize()` rewrites those before nncase ever
  sees them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

from onnx_legalizer import legalize


def _pin_batch_dim(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix every graph input's first dimension to 1 (batch), then re-run
    shape inference so the fix propagates to intermediate tensors too.

    A dynamic batch dim (a `dim_param`, or some exporters' `dim_value: 0`)
    is standard practice for a model meant to run on a GPU/server batch --
    meaningless for on-device inference, which is always batch-1, and (see
    module docstring) actively breaks nncase's shape propagation.
    """
    for inp in model.graph.input:
        dim0 = inp.type.tensor_type.shape.dim[0]
        if dim0.HasField("dim_param") or dim0.dim_value <= 0:
            dim0.ClearField("dim_param")
            dim0.dim_value = 1
    return shape_inference.infer_shapes(model)


def convert_to_kmodel(
    onnx_path: Path,
    *,
    target: str = "k210",
    quant_type: str = "uint8",
    input_type: str = "float32",
    samples_count: int = 8,
    calibration_data: np.ndarray | None = None,
) -> bytes:
    """Compile a simplified/quantized ONNX model to K210 kmodel bytes.

    `calibration_data`, if given, must already be shaped
    `(samples_count, *input_shape)` in `input_type`'s dtype; a synthetic
    random batch is used otherwise (fine for a smoke test, not for real
    accuracy -- see onnxsim's own `calibrate()` for real calibration data
    if this matters for your model).
    """
    import nncase  # deferred: only needed here, and only installable on Python <=3.10

    model = onnx.load(str(onnx_path))
    model = legalize(model)  # rewrite ops nncase's importer doesn't support (see onnx_legalizer.py)
    model = shape_inference.infer_shapes(model)
    model = _pin_batch_dim(model)

    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]

    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_type = input_type
    compile_options.quant_type = quant_type  # K210's KPU is native uint8, not int8

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(model.SerializeToString(), nncase.ImportOptions())

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = samples_count
    if calibration_data is None:
        calibration_data = np.random.rand(samples_count, *input_shape).astype(input_type)
    ptq_options.set_tensor_data(calibration_data.tobytes())
    compiler.use_ptq(ptq_options)

    compiler.compile()
    return compiler.gencode_tobytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_model", type=Path, help="simplified .onnx file (shape inference is run automatically)")
    parser.add_argument("out_kmodel", type=Path, help=".kmodel file to write")
    parser.add_argument("--target", default="k210", help="nncase compile target (default: k210)")
    parser.add_argument("--quant-type", default="uint8", choices=["uint8", "int8"],
                         help="K210's KPU is native uint8; int8 is nncase's other supported option (default: uint8)")
    parser.add_argument("--input-type", default="float32", help="external input dtype nncase should expose (default: float32)")
    parser.add_argument("--samples-count", type=int, default=8, help="PTQ calibration sample count (default: 8, synthetic random data)")
    args = parser.parse_args(argv)

    kmodel_bytes = convert_to_kmodel(
        args.onnx_model,
        target=args.target,
        quant_type=args.quant_type,
        input_type=args.input_type,
        samples_count=args.samples_count,
    )
    args.out_kmodel.write_bytes(kmodel_bytes)
    print(f"wrote {args.out_kmodel} ({len(kmodel_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
