// onnx-macs.js -- read onnxsim's MAC/FLOP annotations out of an ONNX model and
// help drive an onnxruntime-web (WASM) inference over it.
//
// onnxsim.model_info.annotate_metadata (onnxsim PR #527) bakes the computed
// metrics into the model's protobuf `metadata_props`, at the model level and on
// every compute node, under keys prefixed with "onnxsim.". onnxruntime-web does
// not surface `metadata_props` through its JS API, so this module reads them
// straight from the model bytes with a tiny, dependency-free protobuf scanner
// that walks only the handful of fields it needs.
//
// It is a plain ES module so the same code runs in the browser (imported by
// index.html) and under Node (imported by test_node.mjs), keeping one source of
// truth for the parsing and inference logic.

export const METADATA_PREFIX = "onnxsim.";

// --- minimal protobuf wire reader ------------------------------------------
// We only ever read length-delimited (wire 2) and varint (wire 0) fields, plus
// skip the rest. Varints are accumulated with multiplication (not <<) so values
// wider than 32 bits are read correctly.

function readVarint(buf, pos) {
  let result = 0;
  let shift = 1; // 2 ** (7 * i)
  while (true) {
    const byte = buf[pos++];
    result += (byte & 0x7f) * shift;
    if ((byte & 0x80) === 0) break;
    shift *= 128;
  }
  return [result, pos];
}

// Iterate the fields of a protobuf message buffer, yielding
// { field, wire, value?, bytes? }. Length-delimited payloads are returned as a
// subarray view; varints as a Number in `value`.
function* fields(buf) {
  let pos = 0;
  const len = buf.length;
  while (pos < len) {
    let tag;
    [tag, pos] = readVarint(buf, pos);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (wire === 0) {
      let value;
      [value, pos] = readVarint(buf, pos);
      yield { field, wire, value };
    } else if (wire === 2) {
      let size;
      [size, pos] = readVarint(buf, pos);
      const bytes = buf.subarray(pos, pos + size);
      pos += size;
      yield { field, wire, bytes };
    } else if (wire === 1) {
      pos += 8; // fixed64, skipped
    } else if (wire === 5) {
      pos += 4; // fixed32, skipped
    } else {
      throw new Error(`unsupported protobuf wire type ${wire}`);
    }
  }
}

const utf8 = new TextDecoder("utf-8");
const decode = (bytes) => utf8.decode(bytes);

// StringStringEntryProto { string key = 1; string value = 2; }
function parseStringStringEntry(buf) {
  let key = "";
  let value = "";
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) key = decode(f.bytes);
    else if (f.field === 2 && f.wire === 2) value = decode(f.bytes);
  }
  return [key, value];
}

// Collect all onnxsim.* entries of a metadata_props list into a plain object,
// stripping the prefix (so { macs: "1622336", flops: "3244672", ... }).
function collectOnnxsimMeta(metaEntries) {
  const out = {};
  for (const [key, value] of metaEntries) {
    if (key.startsWith(METADATA_PREFIX)) {
      out[key.slice(METADATA_PREFIX.length)] = value;
    }
  }
  return out;
}

// NodeProto: input=1, output=2, name=3, op_type=4, metadata_props=9
function parseNode(buf) {
  let name = "";
  let opType = "";
  const metaEntries = [];
  for (const f of fields(buf)) {
    if (f.field === 3 && f.wire === 2) name = decode(f.bytes);
    else if (f.field === 4 && f.wire === 2) opType = decode(f.bytes);
    else if (f.field === 9 && f.wire === 2) metaEntries.push(parseStringStringEntry(f.bytes));
  }
  const meta = collectOnnxsimMeta(metaEntries);
  return {
    name,
    opType,
    macs: numOrNull(meta.macs),
    flops: numOrNull(meta.flops),
    memAccess: numOrNull(meta.mem_access),
    // Raw strings too, so symbolic values (e.g. "512*batch") survive display.
    raw: meta,
  };
}

// TensorShapeProto.Dimension: dim_value=1 (int64), dim_param=2 (string)
function parseDim(buf) {
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 0) return f.value; // concrete size
    if (f.field === 2 && f.wire === 2) return decode(f.bytes); // symbolic name
  }
  return null;
}

// TensorShapeProto: dim=1 (repeated Dimension)
function parseShape(buf) {
  const dims = [];
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) dims.push(parseDim(f.bytes));
  }
  return dims;
}

// TypeProto.Tensor: elem_type=1 (int32), shape=2 (TensorShapeProto)
function parseTensorType(buf) {
  let elemType = 0;
  let dims = null;
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 0) elemType = f.value;
    else if (f.field === 2 && f.wire === 2) dims = parseShape(f.bytes);
  }
  return { elemType, dims };
}

// TypeProto: tensor_type=1
function parseType(buf) {
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) return parseTensorType(f.bytes);
  }
  return { elemType: 0, dims: null };
}

// ValueInfoProto: name=1, type=2
function parseValueInfo(buf) {
  let name = "";
  let type = { elemType: 0, dims: null };
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) name = decode(f.bytes);
    else if (f.field === 2 && f.wire === 2) type = parseType(f.bytes);
  }
  return { name, elemType: type.elemType, dims: type.dims };
}

// TensorProto: name=8 -- we only need initializer names here.
function parseTensorName(buf) {
  for (const f of fields(buf)) {
    if (f.field === 8 && f.wire === 2) return decode(f.bytes);
  }
  return "";
}

// GraphProto: node=1, initializer=5, input=11, metadata_props=16
function parseGraph(buf) {
  const nodes = [];
  const inputs = [];
  const initializerNames = new Set();
  const metaEntries = [];
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) nodes.push(parseNode(f.bytes));
    else if (f.field === 5 && f.wire === 2) initializerNames.add(parseTensorName(f.bytes));
    else if (f.field === 11 && f.wire === 2) inputs.push(parseValueInfo(f.bytes));
    else if (f.field === 16 && f.wire === 2) metaEntries.push(parseStringStringEntry(f.bytes));
  }
  return { nodes, inputs, initializerNames, meta: collectOnnxsimMeta(metaEntries) };
}

// ModelProto: graph=7, metadata_props=14
export function readAnnotations(modelBytes) {
  const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  let graph = { nodes: [], inputs: [], initializerNames: new Set(), meta: {} };
  const modelMetaEntries = [];
  for (const f of fields(buf)) {
    if (f.field === 7 && f.wire === 2) graph = parseGraph(f.bytes);
    else if (f.field === 14 && f.wire === 2) modelMetaEntries.push(parseStringStringEntry(f.bytes));
  }
  const model = collectOnnxsimMeta(modelMetaEntries);
  // Real feed inputs = graph inputs that are not initializers.
  const feedInputs = graph.inputs.filter((i) => !graph.initializerNames.has(i.name));
  return {
    model, // { macs, flops, mem_access, memory_footprint, compute_density, model_size }
    nodes: graph.nodes, // [{ name, opType, macs, flops, memAccess, raw }]
    feedInputs, // [{ name, elemType, dims }]
    annotated: Object.keys(model).length > 0 || graph.nodes.some((n) => n.macs != null),
  };
}

function numOrNull(s) {
  if (s == null) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null; // symbolic strings (e.g. "512*batch") -> null
}

// Aggregate per-node MACs/FLOPs by op type, with a count and the running total.
export function perOpSummary(nodes) {
  const byOp = new Map();
  let totalMacs = 0;
  let counted = 0;
  for (const n of nodes) {
    const macs = n.macs || 0;
    if (n.macs != null && n.macs > 0) counted += 1;
    totalMacs += macs;
    const cur = byOp.get(n.opType) || { opType: n.opType, count: 0, macs: 0 };
    cur.count += 1;
    cur.macs += macs;
    byOp.set(n.opType, cur);
  }
  const rows = [...byOp.values()].sort((a, b) => b.macs - a.macs);
  return { rows, totalMacs, countedNodes: counted, totalNodes: nodes.length };
}

// --- onnxruntime-web inference helpers -------------------------------------
// Map an ONNX TensorProto element type to an ORT dtype string + typed array.

const ELEM_TYPES = {
  1: ["float32", Float32Array],
  2: ["uint8", Uint8Array],
  3: ["int8", Int8Array],
  4: ["uint16", Uint16Array],
  5: ["int16", Int16Array],
  6: ["int32", Int32Array],
  7: ["int64", typeof BigInt64Array !== "undefined" ? BigInt64Array : null],
  9: ["bool", Uint8Array],
  10: ["float16", Uint16Array],
  11: ["float64", Float64Array],
  12: ["uint32", Uint32Array],
  13: ["uint64", typeof BigUint64Array !== "undefined" ? BigUint64Array : null],
};

// Concrete dims for feed creation: any symbolic (string) or unknown (0 / null)
// dimension collapses to 1, which is what the metrics also assume per-sample.
function concreteDims(dims) {
  if (!dims || dims.length === 0) return [1];
  return dims.map((d) => (typeof d === "number" && d > 0 ? d : 1));
}

// Build a feeds dict of random tensors for every non-initializer graph input.
// `ort` is the onnxruntime module (browser global or the npm package).
export function buildRandomFeeds(feedInputs, ort) {
  const feeds = {};
  for (const inp of feedInputs) {
    const dims = concreteDims(inp.dims);
    const size = dims.reduce((a, b) => a * b, 1);
    const spec = ELEM_TYPES[inp.elemType] || ELEM_TYPES[1];
    const [dtype, ArrayType] = spec;
    if (!ArrayType) throw new Error(`unsupported input dtype ${inp.elemType} for ${inp.name}`);
    let data;
    if (dtype === "int64" || dtype === "uint64") {
      data = new ArrayType(size); // zeros; safe indices for e.g. embeddings
    } else {
      data = new ArrayType(size);
      const isFloat = dtype === "float32" || dtype === "float64";
      for (let i = 0; i < size; i++) data[i] = isFloat ? Math.random() : (i % 2);
    }
    feeds[inp.name] = new ort.Tensor(dtype, data, dims);
  }
  return feeds;
}

// Run the model with the WASM backend, timing an average over `runs` (after a
// warm-up). Returns { avgMs, feeds, outputNames, session }.
export async function runInference(ort, modelBytes, { runs = 20 } = {}) {
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const ann = readAnnotations(modelBytes);
  const feeds = buildRandomFeeds(ann.feedInputs, ort);

  await session.run(feeds); // warm-up (JIT, allocations)
  const t0 = now();
  for (let i = 0; i < runs; i++) await session.run(feeds);
  const avgMs = (now() - t0) / runs;
  return { avgMs, feeds, outputNames: session.outputNames, session };
}

function now() {
  return typeof performance !== "undefined" && performance.now
    ? performance.now()
    : Number(process.hrtime.bigint() / 1000n) / 1000;
}
