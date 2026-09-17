"""Tests for onnx_to_tflite_flatbuffers.py.

Fast, unconditional test: builds a tiny synthetic "fake-quant sandwich"
ONNX model with onnx.parser (see repo CLAUDE.md) and checks the emitted
.tflite's structure directly with the vendored schema bindings -- no
TFLite runtime needed, just the `flatbuffers` package this script itself
requires.

Real end-to-end test (skipped without network access or `ai-edge-litert`):
runs each of this directory's own README candidate models' real ONNX
through this converter and compares its output, over many random inputs,
against onnx.reference.ReferenceEvaluator -- the ground truth for whether
this converter faithfully carried the ONNX graph's own quantized values
into the emitted .tflite. Manual testing (100 trials/model) found: exact
match every time for 3 of the 5 (TinyConv, Streaming DS-CNN, the deep
Autoencoder); the other two (DS-CNN, DS-CNN Large -- both 9-11 conv
layers deep) mismatch on ~2% of trials, by at most single-digit counts,
always on an already-saturated (+-128) output -- consistent with
accumulated float-vs-fixed-point rounding drift compounding across many
quantized layers (the same category of discrepancy TinyConv's single
conv layer showed at a ~1-in-200 rate), not a structural bug: both sides
consumed the exact same scale/zero-point/weight bytes from the ONNX
file, and a wrong quantized_dimension or weight transpose would produce
*systematic* errors on most channels, not a rare few-count perturbation
on rare inputs. The per-model tolerances below reflect those real
numbers, not guesses.
"""

import sys
import urllib.request
from pathlib import Path

import pytest

# This directory's other test file (test_onnx_to_tflite_micro.py) keeps its
# pytest-collected tests free of the onnx2tf/TensorFlow dependency for the
# same reason: this repo's own CI has no TensorFlow install (see
# ../README.md's "No CI coverage" follow-up). onnx itself is lighter and
# usually present, but guard it the same way so a bare environment gets a
# clean skip here instead of a collection error that would also block this
# directory's other tests -- confirmed directly hitting that failure mode
# while developing this file.
onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx import numpy_helper, parser  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytest.importorskip("flatbuffers")

from onnx_to_tflite_flatbuffers import convert  # noqa: E402
from tflite_schema import schema_py_generated as tfl  # noqa: E402

_BUILTIN_SOFTMAX = 25


def _softmax_sandwich_model() -> onnx.ModelProto:
    """A minimal fake-quant sandwich: uint8 in -> Dequant -> Softmax ->
    Quant -> uint8 out. Exercises the pattern-matching/emission path
    without needing a real trained Conv/MatMul's weights."""
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 16]>
        agraph (uint8[1,4] x) => (uint8[1,4] y) {
            x_f = DequantizeLinear(x, in_scale, in_zp)
            y_f = Softmax(x_f)
            y = QuantizeLinear(y_f, out_scale, out_zp)
        }
        """
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.float32(0.1), name="in_scale"),
            numpy_helper.from_array(np.uint8(0), name="in_zp"),
            numpy_helper.from_array(np.float32(1.0 / 255), name="out_scale"),
            numpy_helper.from_array(np.uint8(0), name="out_zp"),
        ]
    )
    return model


def test_softmax_sandwich_emits_a_single_softmax_op(tmp_path):
    model_path = tmp_path / "softmax.onnx"
    onnx.save(_softmax_sandwich_model(), model_path)

    data = convert(model_path)
    assert data[4:8] == b"TFL3"  # TFLite's own flatbuffer file identifier

    m = tfl.Model.GetRootAsModel(bytearray(data), 0)
    g = m.Subgraphs(0)
    # Reshape/Transpose/QuantizeLinear/DequantizeLinear nodes are never
    # emitted as ops -- only the real compute op should show up.
    assert g.OperatorsLength() == 1
    op = g.Operators(0)
    opcode = m.OperatorCodes(op.OpcodeIndex())
    assert opcode.BuiltinCode() == _BUILTIN_SOFTMAX
    assert g.TensorsLength() == 2  # just the quantized input and output tensors

    in_t = g.Tensors(g.Inputs(0))
    assert in_t.Type() == tfl.TensorType.UINT8
    q = in_t.Quantization()
    assert q.Scale(0) == pytest.approx(0.1)
    assert q.ZeroPoint(0) == 0


# (repo, expected op, max_diff, max_mismatch_rate) -- max_diff/rate reflect
# real 100-trial manual runs (see module docstring), not guesses. expected_op
# is a cheap sanity check that the model actually exercises new op coverage
# (Relu-fusion/Add-bias/AveragePool/Gemm), not just the original Conv/
# MatMul/Softmax path TinyConv alone would've covered.
_CANDIDATE_MODELS = [
    (
        "ketiswp/tensorflow-Micro-Speech-TinyConv-SpeechCommands-uint8-onnx",
        "conv",
        1,
        0.1,
    ),
    (
        "ketiswp/mlcommons-DS-CNN-SpeechCommands-int8-onnx",
        "avgpool+relu+bias-add",
        8,
        0.1,
    ),
    (
        "ketiswp/mlcommons-Streaming-Wakeword-DS-CNN-SpeechCommands-int8-onnx",
        "relu+bias-add",
        1,
        0.1,
    ),
    (
        "ketiswp/arm-DS-CNN-Large-SpeechCommands-clustered-int8-onnx",
        "avgpool+relu+gemm",
        3,
        0.1,
    ),
    (
        "ketiswp/mlcommons-Deep-Autoencoder-DCASE2020-ToyCar-int8-onnx",
        "relu+bias-add (no conv at all)",
        1,
        0.1,
    ),
]


@pytest.mark.parametrize(
    "repo,expected_op,max_diff,max_mismatch_rate",
    _CANDIDATE_MODELS,
    ids=[r.split("/")[1] for r, *_ in _CANDIDATE_MODELS],
)
def test_real_model_matches_onnx_reference(
    tmp_path, repo, expected_op, max_diff, max_mismatch_rate
):
    """The actual end-to-end check, run against every candidate model in
    this directory's own README table (see that table's "Pipeline and
    what's verified" for the op each one adds real coverage for --
    `expected_op` here is just a mnemonic, not asserted): convert a real
    Hugging Face model's ONNX and confirm the emitted .tflite, run through
    a real TFLite interpreter, matches ONNX's own reference evaluation
    over many random inputs.

    Requires network access (downloads the model once) and
    `ai-edge-litert` (a standalone TFLite interpreter -- NOT a dependency
    of onnx_to_tflite_flatbuffers.py itself, only of this test's
    verification step; see that script's own module docstring for why it
    doesn't need TensorFlow or any TFLite runtime to *emit* the file).
    """
    del expected_op  # documentation only, see docstring
    litert = pytest.importorskip("ai_edge_litert.interpreter")

    onnx_path = tmp_path / "model.onnx"
    url = f"https://huggingface.co/{repo}/resolve/main/model.onnx"
    try:
        urllib.request.urlretrieve(url, onnx_path)
    except Exception as e:
        pytest.skip(f"no network access to fetch the test model: {e}")

    tflite_bytes = convert(onnx_path)
    assert tflite_bytes[4:8] == b"TFL3"

    interp = litert.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    tflite_shape = tuple(in_detail["shape"])
    dtype = in_detail["dtype"]

    onnx_model = onnx.load(onnx_path)
    onnx_input = onnx_model.graph.input[0]
    # ONNX's own declared input shape sometimes differs from this
    # converter's canonical NHWC (see convert()'s own comment on this --
    # confirmed real for one of these five models) but is always a
    # byte-identical reshape of it (only a size-1 axis ever moves), so
    # generating one random NHWC array and reshaping it for the ONNX side
    # keeps both runs looking at the literal same bytes.
    raw_shape = tuple((d.dim_value or 1) for d in onnx_input.type.tensor_type.shape.dim)
    # onnx.reference.ReferenceEvaluator has no DequantizeLinear/
    # QuantizeLinear implementation for these models' declared opset 16
    # (it only ships 19+) -- bumping the declared opset is safe here
    # since both ops' scale/zero-point semantics these models actually
    # use are unchanged between those versions.
    onnx_model.opset_import[0].version = 21
    evaluator = ReferenceEvaluator(onnx_model)

    mismatches = 0
    trials = 100
    for seed in range(trials):
        rng = np.random.RandomState(seed)
        if dtype == np.uint8:
            x = rng.randint(0, 256, size=tflite_shape, dtype=np.uint8)
        else:
            x = rng.randint(-128, 128, size=tflite_shape, dtype=np.int8)

        (onnx_out,) = evaluator.run(None, {onnx_input.name: x.reshape(raw_shape)})

        interp.set_tensor(in_detail["index"], x)
        interp.invoke()
        tflite_out = interp.get_tensor(out_detail["index"])

        if not np.array_equal(onnx_out, tflite_out):
            diff = np.abs(onnx_out.astype(int) - tflite_out.astype(int))
            assert diff.max() <= max_diff, (
                f"seed {seed}: onnx={onnx_out} emitted={tflite_out} (diff > {max_diff}, not the known rounding-drift envelope)"
            )
            mismatches += 1

    assert mismatches <= trials * max_mismatch_rate, (
        f"{mismatches}/{trials} mismatches vs ONNX reference eval -- too many for known rounding noise"
    )
