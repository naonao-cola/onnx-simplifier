// A minimal, hand-rolled protobuf reader for exactly one path through an
// ONNX ModelProto: graph -> node[] -> metadata_props[] -> {key, value}. Not
// a general protobuf runtime and not a port of onnx.js -- onnxruntime-web
// itself doesn't expose arbitrary metadata_props (only input/output names),
// and pulling in a full protobuf library (or onnx's own generated JS, which
// doesn't exist) just to read a handful of string fields out of a message
// this simple is more machinery than the job needs.
//
// This is the browser-side counterpart to onnxsim/webgpu_kernel_metadata.py,
// which writes the "onnxsim.webgpu_kernel" per-node metadata entry this file
// reads back (see that module's docstring for the JSON schema the value
// decodes to). scripts/convertmodel/test/onnx_node_metadata.test.mjs proves
// the two sides agree, by parsing a real .onnx file the Python side wrote.
//
// Field numbers below are NOT guessed: they were read directly off the
// installed `onnx` Python package's own protobuf descriptors
// (`onnx.NodeProto.DESCRIPTOR.fields_by_name[...].number`, etc.), and
// protobuf's own wire-format contract guarantees a field's number never
// changes across versions once assigned (only field *removal*/deprecation,
// never renumbering, is allowed) -- so hardcoding them here is safe for any
// onnx version, not just the one field numbers were read from.
//
//   ModelProto.graph            = field 7,  message
//   GraphProto.node             = field 1,  repeated message
//   NodeProto.name              = field 3,  string
//   NodeProto.metadata_props    = field 9,  repeated message
//   StringStringEntryProto.key   = field 1, string
//   StringStringEntryProto.value = field 2, string
//
// Every other field encountered anywhere in this walk is skipped via its
// wire type (varint/fixed64/length-delimited/fixed32) rather than assumed
// absent -- a real model has plenty of other fields at every level (inputs,
// op_type, attributes, ir_version, ...), and this reader has to walk past
// all of them correctly to find the ones it wants.

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

  // Advances past one field's value, whatever it is -- required for a
  // correct walk of any message, since every level here has fields this
  // reader doesn't care about interleaved with the ones it does.
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
        // Wire type 3/4 (deprecated protobuf "groups") never appears in
        // onnx's own .proto -- surfacing this loudly beats silently
        // mis-parsing the rest of the message.
        throw new Error(`unsupported protobuf wire type ${wireType}`);
    }
  }
}

const utf8 = new TextDecoder();

function readStringStringEntry(bytes) {
  const r = new Reader(bytes);
  let key = "";
  let value = "";
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      key = utf8.decode(r.readLenDelimited());
    } else if (field === 2 && wireType === WIRE_LEN) {
      value = utf8.decode(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  return { key, value };
}

function readNode(bytes) {
  const r = new Reader(bytes);
  let name = "";
  const metadataProps = [];
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 3 && wireType === WIRE_LEN) {
      name = utf8.decode(r.readLenDelimited());
    } else if (field === 9 && wireType === WIRE_LEN) {
      metadataProps.push(readStringStringEntry(r.readLenDelimited()));
    } else {
      r.skip(wireType);
    }
  }
  return { name, metadataProps };
}

function readGraphNodes(bytes) {
  const r = new Reader(bytes);
  const nodes = [];
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 1 && wireType === WIRE_LEN) {
      nodes.push(readNode(r.readLenDelimited()));
    } else {
      r.skip(wireType);
    }
  }
  return nodes;
}

/**
 * Extracts every node's ``metadata_props`` from raw ONNX ``ModelProto``
 * bytes. Nodes with no metadata_props at all are omitted; a node with an
 * empty ``name`` (onnx.parser never assigns one -- see
 * tests/test_webgpu_kernel_metadata.py's module docstring on the Python
 * side) is keyed under the empty string like any other, so a caller relying
 * on this for a specific node needs that node actually named.
 *
 * @param {Uint8Array} modelBytes
 * @returns {Map<string, Map<string, string>>} node name -> (metadata key -> value)
 */
export function readNodeMetadataProps(modelBytes) {
  const r = new Reader(modelBytes);
  let nodes = [];
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    if (field === 7 && wireType === WIRE_LEN) {
      nodes = readGraphNodes(r.readLenDelimited());
    } else {
      r.skip(wireType);
    }
  }
  const result = new Map();
  for (const node of nodes) {
    if (node.metadataProps.length === 0) continue;
    result.set(node.name, new Map(node.metadataProps.map((e) => [e.key, e.value])));
  }
  return result;
}

// Matches onnxsim.webgpu_kernel_metadata's `_KERNEL_METADATA_KEY`
// (`onnxsim.model_info.METADATA_PREFIX + "webgpu_kernel"`).
export const WEBGPU_KERNEL_METADATA_KEY = "onnxsim.webgpu_kernel";

/**
 * Every node's custom WebGPU kernel spec (see
 * onnxsim/webgpu_kernel_metadata.py's docstring for the JSON schema),
 * parsed from raw ONNX ``ModelProto`` bytes.
 *
 * @param {Uint8Array} modelBytes
 * @returns {Map<string, object>} node name -> parsed kernel spec
 */
export function readWebgpuKernelSpecs(modelBytes) {
  const perNode = readNodeMetadataProps(modelBytes);
  const specs = new Map();
  for (const [nodeName, props] of perNode) {
    const raw = props.get(WEBGPU_KERNEL_METADATA_KEY);
    if (raw !== undefined) {
      specs.set(nodeName, JSON.parse(raw));
    }
  }
  return specs;
}
