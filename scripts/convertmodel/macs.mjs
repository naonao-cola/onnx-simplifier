// macs.mjs -- read onnxsim's MAC/FLOP annotations out of an ONNX model.
//
// onnxsim.model_info.annotate_metadata (onnxsim PR #527) bakes the computed
// metrics into the model's protobuf `metadata_props`, at the model level and on
// every compute node, under keys prefixed with "onnxsim.". onnxruntime-web does
// not surface `metadata_props` through its JS API, so this module reads them
// straight from the model bytes with a tiny, dependency-free protobuf scanner
// that walks only the handful of fields it needs. The "Run inference" panel
// uses it to report a model's MACs/FLOPs alongside the measured latency, and
// the Node test in test/macs.test.mjs drives the exact same code.

export const METADATA_PREFIX = "onnxsim.";

// --- minimal protobuf wire reader ------------------------------------------
// We only ever read length-delimited (wire 2) and varint (wire 0) fields, and
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
// { field, wire, value?, bytes? }.
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

// Collect all onnxsim.* entries into a plain object, stripping the prefix.
function collectOnnxsimMeta(metaEntries) {
  const out = {};
  for (const [key, value] of metaEntries) {
    if (key.startsWith(METADATA_PREFIX)) {
      out[key.slice(METADATA_PREFIX.length)] = value;
    }
  }
  return out;
}

// NodeProto: name=3, op_type=4, metadata_props=9
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
    raw: meta, // keep raw strings so symbolic values (e.g. "512*batch") survive
  };
}

// GraphProto: node=1, metadata_props=16
function parseGraph(buf) {
  const nodes = [];
  const metaEntries = [];
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) nodes.push(parseNode(f.bytes));
    else if (f.field === 16 && f.wire === 2) metaEntries.push(parseStringStringEntry(f.bytes));
  }
  return { nodes, meta: collectOnnxsimMeta(metaEntries) };
}

// ModelProto: graph=7, metadata_props=14
export function readAnnotations(modelBytes) {
  const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  let graph = { nodes: [], meta: {} };
  const modelMetaEntries = [];
  for (const f of fields(buf)) {
    if (f.field === 7 && f.wire === 2) graph = parseGraph(f.bytes);
    else if (f.field === 14 && f.wire === 2) modelMetaEntries.push(parseStringStringEntry(f.bytes));
  }
  const model = collectOnnxsimMeta(modelMetaEntries);
  return {
    model, // { macs, flops, mem_access, memory_footprint, compute_density, model_size }
    nodes: graph.nodes, // [{ name, opType, macs, flops, memAccess, raw }]
    annotated: Object.keys(model).length > 0 || graph.nodes.some((n) => n.macs != null),
  };
}

function numOrNull(s) {
  if (s == null) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null; // symbolic strings (e.g. "512*batch") -> null
}

// Aggregate per-node MACs by op type, largest first, with the running total.
export function perOpSummary(nodes) {
  const byOp = new Map();
  let totalMacs = 0;
  for (const n of nodes) {
    const macs = n.macs || 0;
    totalMacs += macs;
    const cur = byOp.get(n.opType) || { opType: n.opType, count: 0, macs: 0 };
    cur.count += 1;
    cur.macs += macs;
    byOp.set(n.opType, cur);
  }
  const rows = [...byOp.values()].sort((a, b) => b.macs - a.macs);
  return { rows, totalMacs, totalNodes: nodes.length };
}

// --- formatting (mirrors onnxsim.model_info human_readable_*) ---------------

function human(n, base, units, suffix = "") {
  if (n == null || !Number.isFinite(+n)) return String(n); // symbolic formula
  n = +n;
  for (const u of units) {
    if (Math.abs(n) < base) return `${n < 10 && u ? n.toFixed(1) : n.toFixed(0)}${u}${suffix}`;
    n /= base;
  }
  return `${n.toFixed(1)}${units[units.length - 1]}${suffix}`;
}
export const humanNum = (n) => human(n, 1000, ["", "K", "M", "G", "T", "P"]);
export const humanBytes = (n) => human(n, 1024, ["", "Ki", "Mi", "Gi", "Ti"], "B");
export function humanDensity(n) {
  return Number.isFinite(+n) ? `${(+n).toFixed(2)} FLOP/Byte` : `${n} FLOP/Byte`;
}

// Throughput from a model's annotated FLOPs and a measured average latency.
// Returns null when the FLOPs are unknown or symbolic. `avgMs` is milliseconds.
export function throughput(model, avgMs) {
  const macs = Number(model?.macs);
  if (!Number.isFinite(macs) || !(avgMs > 0)) return null;
  const flops = 2 * macs;
  return {
    macs,
    flops,
    gflops: flops / (avgMs / 1000) / 1e9,
    gmacs: macs / (avgMs / 1000) / 1e9,
  };
}
