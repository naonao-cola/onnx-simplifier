// A minimal, hand-rolled protobuf reader for exactly one path through an
// ONNX ModelProto -- graph -> node[].op_type and graph -> input[].type's
// shape -- plus the pure analysis this file's docstring-equivalent comment
// below describes. Same convention as onnx_node_metadata.mjs (not a general
// protobuf runtime, field numbers taken directly from the installed `onnx`
// Python package's own protobuf descriptors, every other field skipped by
// its wire type rather than assumed absent):
//
//   ModelProto.graph                 = field 7,  message
//   GraphProto.node                  = field 1,  repeated message
//   GraphProto.input                 = field 11, repeated message
//   NodeProto.op_type                = field 4,  string
//   ValueInfoProto.type               = field 2,  message
//   TypeProto.tensor_type             = field 1,  message
//   TypeProto.Tensor.shape            = field 2,  message
//   TensorShapeProto.dim              = field 1,  repeated message
//   TensorShapeProto.Dimension.dim_value = field 1, int64 (varint)
//   TensorShapeProto.Dimension.dim_param = field 2, string
//
// This is the browser-side counterpart of scripts/edgeai/tidl_ops.py's
// static op-support heuristic for TI's TIDL (edgeai) accelerator -- see
// that module's docstring for where each blocker category and the
// dynamic-shape check come from (edgeai-tidl-tools' own docs/operators.md
// and docs/vision_transformers.md). Deliberately a smaller subset of that
// module, not a full port:
//
// - Does NOT recurse into `If`/`Loop`/`Scan` subgraph attributes -- only
//   top-level nodes are scanned. A blocker nested inside a subgraph is
//   missed; the top-level `If`/`Loop`/`Scan` node itself is still caught
//   (it's one of the blocker types).
// - Does NOT check for STRING tensors (tidl_ops.has_string_tensor's
//   equivalent) -- a much rarer case in a browser-converted model than
//   dynamic shapes or control flow.
// - Does NOT check the decomposed-LayerNorm signature
//   (has_decomposed_normalization) -- that one's about steering a
//   transformer graph toward the fused op TIDL prefers, a finer-grained
//   suggestion than the coarse "will this even run" checks below.
//
// A correspondingly smaller, still-real claim: this is a static heuristic
// derived from published documentation, not a real TIDL compiler run --
// see scripts/edgeai/real_compile.py (Python-only; no on-device or in-browser
// equivalent exists) for what an actual compile confirms instead.

const WIRE_VARINT = 0;
const WIRE_FIXED64 = 1;
const WIRE_LEN = 2;
const WIRE_FIXED32 = 5;

class Reader {
  constructor(buf, start = 0, end = buf.length) {
    this.buf = buf;
    this.pos = start;
    this.end = end;
  }

  eof() {
    return this.pos >= this.end;
  }

  readVarint() {
    let result = 0n;
    let shift = 0n;
    for (;;) {
      if (this.pos >= this.end) {
        throw new Error("truncated varint");
      }
      const b = this.buf[this.pos++];
      result |= BigInt(b & 0x7f) << shift;
      if ((b & 0x80) === 0) break;
      shift += 7n;
    }
    return result;
  }

  readTag() {
    const tag = Number(this.readVarint());
    return { field: tag >>> 3, wireType: tag & 0x7 };
  }

  readLenDelimited() {
    const len = Number(this.readVarint());
    const start = this.pos;
    this.pos += len;
    if (this.pos > this.end) {
      throw new Error("length-delimited field runs past message end");
    }
    return this.buf.subarray(start, this.pos);
  }

  skip(wireType) {
    switch (wireType) {
      case WIRE_VARINT:
        this.readVarint();
        break;
      case WIRE_FIXED64:
        this.pos += 8;
        break;
      case WIRE_LEN:
        this.readLenDelimited();
        break;
      case WIRE_FIXED32:
        this.pos += 4;
        break;
      default:
        throw new Error(`unsupported protobuf wire type ${wireType}`);
    }
  }
}

const utf8 = new TextDecoder();

function readNodeOpType(bytes) {
  const r = new Reader(bytes);
  let opType = "";
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 4 && wireType === WIRE_LEN) {
      opType = utf8.decode(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  return opType;
}

// True if `bytes` (a TensorShapeProto) has any dim with neither a
// dim_value nor a dim_param set (unranked-along-that-axis) or a dim_param
// (symbolic) -- either way, not a static integer.
function shapeHasNonStaticDim(bytes) {
  const r = new Reader(bytes);
  let sawStatic = false;
  let sawAny = false;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      sawAny = true;
      const dimBytes = r.readLenDelimited();
      const dr = new Reader(dimBytes);
      let isStatic = false;
      while (!dr.eof()) {
        const { field: dField, wireType: dWireType } = dr.readTag();
        if (dField === 1 && dWireType === WIRE_VARINT) {
          dr.readVarint();
          isStatic = true;
        } else if (dField === 2 && dWireType === WIRE_LEN) {
          dr.readLenDelimited(); // dim_param: symbolic, not static
        } else {
          dr.skip(dWireType);
        }
      }
      if (isStatic) sawStatic = true;
      else return true;
    } else {
      r.skip(wireType);
    }
  }
  return sawAny && !sawStatic ? true : false;
}

function readValueInfoIsDynamic(bytes) {
  const r = new Reader(bytes);
  let sawShape = false;
  let dynamic = false;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 2 && wireType === WIRE_LEN) {
      // TypeProto
      const tr = new Reader(r.readLenDelimited());
      while (!tr.eof()) {
        const { field: tField, wireType: tWireType } = tr.readTag();
        if (tField === 1 && tWireType === WIRE_LEN) {
          // Tensor
          const tenR = new Reader(tr.readLenDelimited());
          while (!tenR.eof()) {
            const { field: tenField, wireType: tenWireType } = tenR.readTag();
            if (tenField === 2 && tenWireType === WIRE_LEN) {
              sawShape = true;
              if (shapeHasNonStaticDim(tenR.readLenDelimited())) dynamic = true;
            } else {
              tenR.skip(tenWireType);
            }
          }
        } else {
          tr.skip(tWireType);
        }
      }
    } else {
      r.skip(wireType);
    }
  }
  // No shape field at all is unranked -- even less static than a dynamic
  // dim, so also flagged (matches tidl_ops.has_dynamic_shape's own
  // "no shape at all... also unsupported" note, except that Python version
  // treats a totally missing `shape` message as ambiguous and skips it;
  // this port treats "TypeProto.Tensor present but no shape" the same way
  // as a dynamic dim, since either way there is no static shape to report).
  return dynamic || !sawShape;
}

/**
 * Every top-level node's `op_type`, and whether any graph input has a
 * non-static (symbolic or unranked) dimension.
 *
 * @param {Uint8Array} modelBytes
 * @returns {{ opTypes: string[], hasDynamicShape: boolean }}
 */
export function readModelSummary(modelBytes) {
  const r = new Reader(modelBytes);
  const opTypes = [];
  let hasDynamicShape = false;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field !== 7 || wireType !== WIRE_LEN) {
      r.skip(wireType);
      continue;
    }
    const gr = new Reader(r.readLenDelimited());
    while (!gr.eof()) {
      const { field: gField, wireType: gWireType } = gr.readTag();
      if (gField === 1 && gWireType === WIRE_LEN) {
        opTypes.push(readNodeOpType(gr.readLenDelimited()));
      } else if (gField === 11 && gWireType === WIRE_LEN) {
        if (readValueInfoIsDynamic(gr.readLenDelimited())) hasDynamicShape = true;
      } else {
        gr.skip(gWireType);
      }
    }
  }
  return { opTypes, hasDynamicShape };
}

// Ported from scripts/edgeai/tidl_ops.py -- see that module's docstring for
// exactly which edgeai-tidl-tools doc each category comes from.
export const CONTROL_FLOW_OPS = new Set(["If", "Loop", "Scan"]);
export const SEQUENCE_OPTIONAL_OPS = new Set([
  "SequenceConstruct",
  "SequenceAt",
  "SequenceEmpty",
  "SequenceErase",
  "SequenceInsert",
  "SequenceLength",
  "SequenceMap",
  "SplitToSequence",
  "ConcatFromSequence",
  "Optional",
  "OptionalGetElement",
  "OptionalHasElement",
]);
export const DATA_DEPENDENT_SHAPE_OPS = new Set(["NonZero", "Unique", "Compress"]);
export const HOST_ONLY_OPS = new Set(["NonMaxSuppression"]);
export const QOPERATOR_OPS = new Set([
  "QLinearConv",
  "QLinearMatMul",
  "QGemm",
  "QLinearAdd",
  "QLinearMul",
  "QLinearAveragePool",
  "QLinearGlobalAveragePool",
  "QLinearLeakyRelu",
  "QLinearSigmoid",
  "QLinearConcat",
  "ConvInteger",
  "MatMulInteger",
]);

function blockerReason(opType) {
  if (CONTROL_FLOW_OPS.has(opType)) {
    return "control flow has no fixed-function TIDL accelerator equivalent";
  }
  if (SEQUENCE_OPTIONAL_OPS.has(opType)) {
    return "Sequence/Optional container ops are not accelerator-schedulable";
  }
  if (DATA_DEPENDENT_SHAPE_OPS.has(opType)) {
    return "output shape depends on runtime data, not just input shape";
  }
  if (HOST_ONLY_OPS.has(opType)) {
    return "documented as running on the host ARM core, not the accelerator";
  }
  if (QOPERATOR_OPS.has(opType)) {
    return "QOperator-format fused-integer op; TIDL only supports the QDQ form";
  }
  return null;
}

/**
 * Analyze `modelBytes` against the same coarse blocker categories
 * scripts/edgeai/tidl_ops.py checks (minus subgraph recursion and the
 * STRING-tensor / decomposed-LayerNorm checks -- see this module's header
 * comment).
 *
 * @param {Uint8Array} modelBytes
 * @returns {{
 *   opTypeCounts: Map<string, number>,
 *   blockers: {opType: string, reason: string}[],
 *   hasDynamicShape: boolean,
 *   coverage: "full" | "partial",
 * }}
 */
export function analyzeTidlCompat(modelBytes) {
  const { opTypes, hasDynamicShape } = readModelSummary(modelBytes);
  const opTypeCounts = new Map();
  const blockerTypes = new Set();
  for (const opType of opTypes) {
    opTypeCounts.set(opType, (opTypeCounts.get(opType) || 0) + 1);
    if (blockerReason(opType)) blockerTypes.add(opType);
  }
  const blockers = [...blockerTypes]
    .sort()
    .map((opType) => ({ opType, reason: blockerReason(opType) }));
  return {
    opTypeCounts,
    blockers,
    hasDynamicShape,
    coverage: blockers.length > 0 || hasDynamicShape ? "partial" : "full",
  };
}

// From scripts/edgeai/README.md's quantization survey (docs/quantization.md,
// docs/model_compilation.md's `tensor_bits` option) -- purely informational
// here, since there is no in-browser equivalent of a real TIDL compile to
// actually apply a precision choice against (see scripts/edgeai/real_compile.py).
export const PRECISION_NOTES = {
  int8: "8-bit is TIDL's default, “recommended for optimal performance”, on every supported SoC. Asymmetric activation quantization is supported on all SoCs except J721E/TDA4VM (symmetric-only there); weights are quantized per-channel, activations per-tensor.",
  int16: "16-bit is available “for cases requiring higher precision”, at a performance cost — typically applied to specific layers (mixed precision) rather than the whole model.",
  mixed: "Mixed precision runs most layers at 8-bit and select layers at 16-bit, either picked manually (output/parameter name lists) or automatically from a latency budget (mixed_precision_factor = tolerable mixed-precision latency ÷ 8-bit latency).",
};
