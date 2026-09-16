// The write-side counterpart to onnx_node_metadata.mjs's readWebgpuKernelSpecs
// -- attaches (or replaces) one node's "onnxsim.webgpu_kernel" metadata_props
// entry entirely in JS, so a browser feature (the kernel tuner panel, see
// webgpu_kernel_tuner.mjs) can export a modified .onnx without a server or
// the compiled onnxsim WASM extension.
//
// This is NOT a general protobuf runtime and does not attempt to understand
// any message's schema beyond the one path it needs to reach (ModelProto ->
// graph -> node[] -> metadata_props[] -> {key, value}, the same path
// onnx_node_metadata.mjs reads). Instead it parses every level it walks
// through into an ordered list of *raw* (field, wireType, value) entries --
// see readAllFields/writeAllFields below -- so every field this module
// doesn't care about (inputs, initializers, attributes, opset_import,
// ir_version, ...) round-trips through completely unexamined and
// byte-identical, at every level from the target node up to the top-level
// ModelProto. That's what makes it safe to use on a real, arbitrary .onnx
// file without this module needing to know onnx's full schema: only the
// fields it explicitly names are ever interpreted or changed.
//
// Field numbers match onnx_node_metadata.mjs's own docstring exactly (same
// source: onnx's installed Python package's protobuf descriptors), plus
// NodeProto.metadata_props' own StringStringEntryProto shape:
//
//   ModelProto.graph            = field 7,  message
//   GraphProto.node             = field 1,  repeated message
//   NodeProto.name              = field 3,  string
//   NodeProto.metadata_props    = field 9,  repeated message
//   StringStringEntryProto.key   = field 1, string
//   StringStringEntryProto.value = field 2, string
//
// Repeated fields in protobuf's wire format are just the same field number
// appearing more than once, in whatever order a writer chose -- appending a
// *new* metadata_props entry (rather than trying to edit one in place) is
// therefore valid wire format on its own; removing a stale same-key entry
// first (see attachWebgpuKernelSpec below) is what gives "at most one entry
// per key" its actual meaning, mirroring
// onnxsim.webgpu_kernel_metadata._set_node_metadata's own
// remove-then-append behavior on the Python side.

const WIRE_VARINT = 0;
const WIRE_FIXED64 = 1;
const WIRE_LEN = 2;
const WIRE_FIXED32 = 5;

const utf8Decoder = new TextDecoder();
const utf8Encoder = new TextEncoder();

class Reader {
  constructor(buf) {
    this.buf = buf;
    this.pos = 0;
    this.end = buf.length;
  }

  eof() {
    return this.pos >= this.end;
  }

  readVarint() {
    let result = 0n;
    let shift = 0n;
    for (;;) {
      if (this.pos >= this.end) throw new Error("truncated varint");
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
    if (this.pos > this.end) throw new Error("length-delimited field runs past message end");
    return this.buf.subarray(start, this.pos);
  }

  readFixed(n) {
    if (this.pos + n > this.end) throw new Error(`truncated fixed${n * 8} field`);
    const bytes = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return bytes;
  }
}

/**
 * Parses ``bytes`` as a protobuf message into an ordered array of raw
 * ``{field, wireType, value}`` entries -- one per field occurrence, in
 * original order, with no schema knowledge at all. ``value`` is a ``bigint``
 * for ``WIRE_VARINT``, or a ``Uint8Array`` (a subarray of ``bytes``, not a
 * copy) for every other wire type -- the tag/length bytes themselves are
 * never included, only the field's own content, so `writeAllFields` can
 * re-derive them.
 *
 * @param {Uint8Array} bytes
 * @returns {Array<{field: number, wireType: number, value: bigint|Uint8Array}>}
 */
function readAllFields(bytes) {
  const r = new Reader(bytes);
  const fields = [];
  while (!r.eof()) {
    const { field, wireType } = r.readTag();
    let value;
    switch (wireType) {
      case WIRE_VARINT:
        value = r.readVarint();
        break;
      case WIRE_LEN:
        value = r.readLenDelimited();
        break;
      case WIRE_FIXED64:
        value = r.readFixed(8);
        break;
      case WIRE_FIXED32:
        value = r.readFixed(4);
        break;
      default:
        throw new Error(`unsupported protobuf wire type ${wireType}`);
    }
    fields.push({ field, wireType, value });
  }
  return fields;
}

function encodeVarint(n) {
  const bytes = [];
  let v = BigInt(n);
  for (;;) {
    const b = Number(v & 0x7fn);
    v >>= 7n;
    if (v === 0n) {
      bytes.push(b);
      break;
    }
    bytes.push(b | 0x80);
  }
  return Uint8Array.from(bytes);
}

function encodeTag(field, wireType) {
  return encodeVarint((field << 3) | wireType);
}

/**
 * Re-serializes an array of ``{field, wireType, value}`` entries (the same
 * shape ``readAllFields`` returns, whether or not every entry actually came
 * from there -- ``attachWebgpuKernelSpec`` below builds a couple by hand) back
 * into protobuf wire format, in array order.
 *
 * @param {Array<{field: number, wireType: number, value: bigint|Uint8Array}>} fields
 * @returns {Uint8Array}
 */
function writeAllFields(fields) {
  const chunks = [];
  let total = 0;
  for (const { field, wireType, value } of fields) {
    const tag = encodeTag(field, wireType);
    const body = wireType === WIRE_VARINT ? encodeVarint(value) : value;
    const lenPrefix = wireType === WIRE_LEN ? encodeVarint(body.length) : null;
    chunks.push(tag, lenPrefix, body);
    total += tag.length + (lenPrefix ? lenPrefix.length : 0) + body.length;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    if (!chunk) continue;
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function readStringField(fields, fieldNumber) {
  for (const f of fields) {
    if (f.field === fieldNumber && f.wireType === WIRE_LEN) {
      return utf8Decoder.decode(f.value);
    }
  }
  return undefined;
}

// Matches onnx_node_metadata.mjs's own WEBGPU_KERNEL_METADATA_KEY.
export const WEBGPU_KERNEL_METADATA_KEY = "onnxsim.webgpu_kernel";

/**
 * Attaches (or replaces) the given node's ``"onnxsim.webgpu_kernel"``
 * metadata -- the write-side counterpart to
 * ``onnx_node_metadata.mjs``'s ``readWebgpuKernelSpecs`` and
 * ``onnxsim.webgpu_kernel_metadata.attach_webgpu_kernel`` (whose JSON schema
 * ``spec`` must already match -- this function does not validate bindings
 * the way the Python side's ``attach_webgpu_kernel`` does, it only writes
 * whatever object it's given).
 *
 * Does not touch any field it doesn't need to reach the target node: every
 * other node, every initializer/input/output/opset_import/... at every
 * level, and every *other* metadata_props entry on the target node itself,
 * round-trips byte-identical (see this module's own docstring).
 *
 * @param {Uint8Array} modelBytes
 * @param {string} nodeName - the target node's ``NodeProto.name``.
 * @param {object} spec - JSON-serializable, matching
 *        ``onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec.to_json()``'s
 *        shape (``{steps: [...], intermediates: {...}}``).
 * @returns {Uint8Array} the modified model, as a new byte array.
 */
export function attachWebgpuKernelSpec(modelBytes, nodeName, spec) {
  if (!nodeName) throw new Error("nodeName must be a non-empty NodeProto.name");

  const modelFields = readAllFields(modelBytes);
  const graphIndex = modelFields.findIndex((f) => f.field === 7 && f.wireType === WIRE_LEN);
  if (graphIndex === -1) throw new Error("model has no graph");

  const graphFields = readAllFields(modelFields[graphIndex].value);
  let nodeIndex = -1;
  for (let i = 0; i < graphFields.length; i++) {
    const f = graphFields[i];
    if (f.field !== 1 || f.wireType !== WIRE_LEN) continue;
    if (readStringField(readAllFields(f.value), 3) === nodeName) {
      nodeIndex = i;
      break;
    }
  }
  if (nodeIndex === -1) throw new Error(`no node named ${JSON.stringify(nodeName)} in the graph`);

  const nodeFields = readAllFields(graphFields[nodeIndex].value);
  const keptNodeFields = nodeFields.filter((f) => {
    if (f.field !== 9 || f.wireType !== WIRE_LEN) return true;
    return readStringField(readAllFields(f.value), 1) !== WEBGPU_KERNEL_METADATA_KEY;
  });
  const entryBytes = writeAllFields([
    { field: 1, wireType: WIRE_LEN, value: utf8Encoder.encode(WEBGPU_KERNEL_METADATA_KEY) },
    { field: 2, wireType: WIRE_LEN, value: utf8Encoder.encode(JSON.stringify(spec)) },
  ]);
  keptNodeFields.push({ field: 9, wireType: WIRE_LEN, value: entryBytes });

  graphFields[nodeIndex] = { field: 1, wireType: WIRE_LEN, value: writeAllFields(keptNodeFields) };
  modelFields[graphIndex] = { field: 7, wireType: WIRE_LEN, value: writeAllFields(graphFields) };
  return writeAllFields(modelFields);
}

// Exported for onnx_metadata_writer.test.mjs's own byte-level verification
// that every *other* node round-trips untouched -- not meant for other
// callers, which should only need attachWebgpuKernelSpec above.
export const _internal = { readAllFields, writeAllFields };
