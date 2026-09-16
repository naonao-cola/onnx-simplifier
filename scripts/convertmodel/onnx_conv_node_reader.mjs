// A minimal, hand-rolled protobuf reader (see onnx_node_metadata.mjs's own
// docstring for why this style, not a general protobuf runtime) that reads
// exactly what onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel itself
// reads off a Conv node -- op_type/domain, input/output names, shapes, and
// the handful of attributes that function cares about (kernel_shape,
// strides, dilations, group, pads, auto_pad) -- so a caller can *regenerate*
// that same kernel entirely client-side (see
// test/pyodide_webgpu_single_node_codegen.test.mjs) instead of only reading
// a kernel some offline Python run already attached to the model.
//
// Field numbers below are NOT guessed: they were read directly off the
// installed `onnx` Python package's own protobuf descriptors (same method
// onnx_node_metadata.mjs's own docstring describes), and protobuf's own
// wire-format contract guarantees a field's number never changes across
// versions once assigned.
//
//   GraphProto.node                      = 1,  repeated message
//   GraphProto.initializer               = 5,  repeated message (TensorProto)
//   GraphProto.input                     = 11, repeated message (ValueInfoProto)
//   NodeProto.input                      = 1,  repeated string
//   NodeProto.output                     = 2,  repeated string
//   NodeProto.name                       = 3,  string
//   NodeProto.op_type                    = 4,  string
//   NodeProto.attribute                  = 5,  repeated message
//   NodeProto.domain                     = 7,  string
//   AttributeProto.name                  = 1,  string
//   AttributeProto.i                     = 3,  int64 (varint)
//   AttributeProto.s                     = 4,  bytes
//   AttributeProto.ints                  = 8,  repeated int64 (packed varint)
//   ValueInfoProto.name                  = 1,  string
//   ValueInfoProto.type                  = 2,  message (TypeProto)
//   TypeProto.tensor_type                = 1,  message (TypeProto.Tensor)
//   TypeProto.Tensor.shape               = 2,  message (TensorShapeProto)
//   TensorShapeProto.dim                 = 1,  repeated message (Dimension)
//   TensorShapeProto.Dimension.dim_value = 1,  int64 (varint)
//   TensorProto.dims                     = 1,  repeated int64 (packed varint)
//   TensorProto.name                     = 8,  string
//
// `ints`/`dims` are repeated scalar-numeric fields -- verified by hand
// (serializing a real TensorProto/AttributeProto and reading the raw wire
// bytes) that onnx's own .proto emits these *unpacked* (one separate
// field-8/field-1 varint tag per element, proto2-style default; onnx.proto
// predates proto3's pack-by-default rule), not as one length-delimited
// packed run -- so this reader accepts an element from either wire
// representation (a lone VARINT tag appends one value; a LEN tag decodes a
// packed run and appends all of them), which is what a spec-compliant
// protobuf reader must do for any repeated scalar field regardless of which
// encoding a particular writer happens to use.
//
// Scope matches onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel's own
// scope: a Conv node whose X/W (and optional B) shapes are fully static and
// readable from either a graph-level input or an initializer -- no
// shape-inference fallback for an intermediate tensor's shape, unlike the
// Python side's own `_static_shape` (which also consults onnx's shape
// inference and value_info). That's exactly what every fixture here needs
// (the node's own inputs, not some upstream node's output), so it's not a
// meaningful scope reduction in practice, just a smaller reader.

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

function readPackedVarints(bytes) {
  const r = new Reader(bytes);
  const out = [];
  while (!r.eof()) out.push(Number(r.readVarint()));
  return out;
}

function readAttribute(bytes) {
  const r = new Reader(bytes);
  const attr = { name: "" };
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      attr.name = utf8.decode(r.readLenDelimited());
    } else if (field === 3 && wireType === WIRE_VARINT) {
      attr.i = Number(r.readVarint());
    } else if (field === 4 && wireType === WIRE_LEN) {
      attr.s = utf8.decode(r.readLenDelimited());
    } else if (field === 8 && wireType === WIRE_VARINT) {
      (attr.ints ??= []).push(Number(r.readVarint()));
    } else if (field === 8 && wireType === WIRE_LEN) {
      (attr.ints ??= []).push(...readPackedVarints(r.readLenDelimited()));
    } else {
      r.skip(wireType);
    }
  }
  return attr;
}

function readNode(bytes) {
  const r = new Reader(bytes);
  const node = { name: "", opType: "", domain: "", inputs: [], outputs: [], attributes: new Map() };
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      node.inputs.push(utf8.decode(r.readLenDelimited()));
    } else if (field === 2 && wireType === WIRE_LEN) {
      node.outputs.push(utf8.decode(r.readLenDelimited()));
    } else if (field === 3 && wireType === WIRE_LEN) {
      node.name = utf8.decode(r.readLenDelimited());
    } else if (field === 4 && wireType === WIRE_LEN) {
      node.opType = utf8.decode(r.readLenDelimited());
    } else if (field === 5 && wireType === WIRE_LEN) {
      const attr = readAttribute(r.readLenDelimited());
      node.attributes.set(attr.name, attr);
    } else if (field === 7 && wireType === WIRE_LEN) {
      node.domain = utf8.decode(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  return node;
}

function readTensorDimsAndName(bytes) {
  const r = new Reader(bytes);
  let name = "";
  const dims = [];
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_VARINT) {
      dims.push(Number(r.readVarint()));
    } else if (field === 1 && wireType === WIRE_LEN) {
      dims.push(...readPackedVarints(r.readLenDelimited()));
    } else if (field === 8 && wireType === WIRE_LEN) {
      name = utf8.decode(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  return { name, dims };
}

function readDim(bytes) {
  const r = new Reader(bytes);
  let dimValue; // left undefined for a symbolic/unknown dim (dim_param, or neither field set)
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_VARINT) {
      dimValue = Number(r.readVarint());
    } else {
      r.skip(wireType);
    }
  }
  return dimValue;
}

function readTensorShapeProto(bytes) {
  const r = new Reader(bytes);
  const dims = [];
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      dims.push(readDim(r.readLenDelimited()));
    } else {
      r.skip(wireType);
    }
  }
  return dims;
}

function readTypeProtoTensorShape(bytes) {
  const r = new Reader(bytes);
  let shape = null;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      const tensorTypeBytes = r.readLenDelimited();
      const tr = new Reader(tensorTypeBytes);
      while (!tr.eof()) {
        const t = tr.readTag();
        if (t.field === 2 && t.wireType === WIRE_LEN) {
          shape = readTensorShapeProto(tr.readLenDelimited());
        } else {
          tr.skip(t.wireType);
        }
      }
    } else {
      r.skip(wireType);
    }
  }
  return shape;
}

function readValueInfo(bytes) {
  const r = new Reader(bytes);
  let name = "";
  let shape = null;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      name = utf8.decode(r.readLenDelimited());
    } else if (field === 2 && wireType === WIRE_LEN) {
      shape = readTypeProtoTensorShape(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  return { name, shape };
}

function readGraph(bytes) {
  const r = new Reader(bytes);
  const nodes = [];
  const initializers = new Map();
  const inputs = new Map();
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      nodes.push(readNode(r.readLenDelimited()));
    } else if (field === 5 && wireType === WIRE_LEN) {
      const t = readTensorDimsAndName(r.readLenDelimited());
      initializers.set(t.name, t.dims);
    } else if (field === 11 && wireType === WIRE_LEN) {
      const vi = readValueInfo(r.readLenDelimited());
      inputs.set(vi.name, vi.shape);
    } else {
      r.skip(wireType);
    }
  }
  return { nodes, initializers, inputs };
}

/**
 * Every default-domain ``Conv`` node's own ``NodeProto.name`` in the graph,
 * in graph node order -- regardless of whether its shapes are static enough
 * for ``readConvNodeInfo`` to actually read (a caller finds that out, with a
 * clear error, only once it tries to tune a specific one). This is the
 * "what could I offer to tune" list: unlike ``readConvNodeInfo``, it doesn't
 * require the node to already carry any attached kernel metadata --
 * onnxsim's own kernel-tuning UI uses this to offer a "Tune this kernel"
 * button for every Conv node a model has, not just ones
 * onnxsim.webgpu_target already flagged as a gap.
 *
 * A node with no name (onnx.parser-built graphs, or third-party exports that
 * skip naming) is included as an empty string like any other -- the same
 * "every onnxsim-produced node has a name" caveat readConvNodeInfo's own
 * caller has to handle applies here too.
 *
 * @param {Uint8Array} modelBytes
 * @returns {string[]}
 */
export function listConvNodeNames(modelBytes) {
  const r = new Reader(modelBytes);
  let graph = null;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 7 && wireType === WIRE_LEN) {
      graph = readGraph(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  if (!graph) return [];
  return graph.nodes
    .filter((n) => (n.domain === "" || n.domain === "ai.onnx") && n.opType === "Conv")
    .map((n) => n.name);
}

/**
 * Reads everything ``onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel``
 * itself reads off a ``Conv`` node -- shapes (initializer or graph-input
 * only, see this file's own docstring), attributes (with the same defaults
 * that function applies), and tensor names -- directly from raw ONNX
 * ``ModelProto`` bytes, with the same validation that function performs
 * (default-domain ``Conv``, ``auto_pad == "NOTSET"``, ``kernel_shape``
 * agreeing with W's own shape) so a caller gets the same errors either way.
 *
 * @param {Uint8Array} modelBytes
 * @param {string} nodeName
 * @returns {{xName: string, wName: string, bName: string|null,
 *            outputName: string, xShape: number[], wShape: number[],
 *            bShape: number[]|null, strides: number[], dilations: number[],
 *            group: number, pads: number[]}}
 */
export function readConvNodeInfo(modelBytes, nodeName) {
  const r = new Reader(modelBytes);
  let graph = null;
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 7 && wireType === WIRE_LEN) {
      graph = readGraph(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  if (!graph) {
    throw new Error("model has no graph");
  }
  const node = graph.nodes.find((n) => n.name === nodeName);
  if (!node) {
    throw new Error(`no node named ${JSON.stringify(nodeName)} in the graph`);
  }
  if (!(node.domain === "" || node.domain === "ai.onnx") || node.opType !== "Conv") {
    throw new Error(`node ${JSON.stringify(nodeName)} is ${node.domain}::${node.opType}, not a default-domain Conv`);
  }

  function shapeOf(tensorName) {
    if (graph.initializers.has(tensorName)) {
      return graph.initializers.get(tensorName);
    }
    if (graph.inputs.has(tensorName)) {
      const shape = graph.inputs.get(tensorName);
      if (shape === null || shape.some((d) => d === undefined)) {
        throw new Error(`tensor ${JSON.stringify(tensorName)} has a non-static dimension`);
      }
      return shape;
    }
    throw new Error(
      `no static shape found for tensor ${JSON.stringify(tensorName)} -- only graph initializers and ` +
        "graph-level inputs are supported here (see this file's own docstring on scope)",
    );
  }

  const xName = node.inputs[0];
  const wName = node.inputs[1];
  const bName = node.inputs.length > 2 && node.inputs[2] ? node.inputs[2] : null;
  const xShape = shapeOf(xName);
  const wShape = shapeOf(wName);
  const bShape = bName ? shapeOf(bName) : null;
  const spatialRank = wShape.length - 2;

  const kernelShapeAttr = node.attributes.get("kernel_shape");
  if (kernelShapeAttr && kernelShapeAttr.ints) {
    const expected = wShape.slice(2);
    const actual = kernelShapeAttr.ints;
    if (actual.length !== expected.length || actual.some((v, i) => v !== expected[i])) {
      throw new Error(
        `kernel_shape attribute ${JSON.stringify(actual)} disagrees with W's own shape ${JSON.stringify(expected)}`,
      );
    }
  }

  const autoPadAttr = node.attributes.get("auto_pad");
  const autoPad = autoPadAttr && autoPadAttr.s !== undefined ? autoPadAttr.s : "NOTSET";
  if (autoPad !== "NOTSET") {
    throw new Error(`Conv auto_pad=${JSON.stringify(autoPad)} is not implemented -- only the default NOTSET is`);
  }

  const stridesAttr = node.attributes.get("strides");
  const dilationsAttr = node.attributes.get("dilations");
  const groupAttr = node.attributes.get("group");
  const padsAttr = node.attributes.get("pads");

  return {
    xName,
    wName,
    bName,
    outputName: node.outputs[0],
    xShape,
    wShape,
    bShape,
    strides: stridesAttr && stridesAttr.ints ? stridesAttr.ints : Array(spatialRank).fill(1),
    dilations: dilationsAttr && dilationsAttr.ints ? dilationsAttr.ints : Array(spatialRank).fill(1),
    group: groupAttr && groupAttr.i !== undefined ? groupAttr.i : 1,
    pads: padsAttr && padsAttr.ints ? padsAttr.ints : Array(2 * spatialRank).fill(0),
  };
}
