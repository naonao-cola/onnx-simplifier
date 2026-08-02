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

// True when a metric string is a symbolic formula (carries a dim name such as
// "batch") rather than a plain number.
export function isSymbolicStr(str) {
  return str != null && /[A-Za-z_]/.test(String(str));
}

// Evaluate a metric string to a concrete number, substituting every symbolic
// dimension (any identifier, e.g. "batch") with 1 -- the representative,
// per-sample value, matching onnxsim's own SymExpr.representative(). Handles the
// integer-polynomial forms onnxsim emits (coefficients with * + - ** and
// parentheses) and the decimal / rational form of compute_density. Returns null
// if the string is missing or cannot be parsed.
//
// A hand-written recursive-descent evaluator, never eval(): the metric strings
// come from a user-supplied model, so they must not be executed as code.
export function evalMetric(str) {
  if (str == null) return null;
  const s = String(str).trim();
  if (s === "") return null;
  if (/^[+-]?\d+(\.\d+)?$/.test(s)) return Number(s); // fast path: plain number

  // Tokenize into numbers, identifiers, and the operators ** * / + - ( ).
  const re = /\s*(\d+\.\d+|\d+|[A-Za-z_]\w*|\*\*|[+\-*/()])/y;
  const tokens = [];
  let m;
  let end = 0; // a sticky regex resets lastIndex to 0 when exec finally fails,
  while ((m = re.exec(s)) !== null) {
    tokens.push(m[1]);
    end = re.lastIndex; // so remember how far we actually consumed.
  }
  if (s.slice(end).trim() !== "") return null; // stray character

  let i = 0;
  const peek = () => tokens[i];

  // expr := term (('+' | '-') term)*
  const parseExpr = () => {
    let v = parseTerm();
    if (v === null) return null;
    while (peek() === "+" || peek() === "-") {
      const op = tokens[i++];
      const r = parseTerm();
      if (r === null) return null;
      v = op === "+" ? v + r : v - r;
    }
    return v;
  };
  // term := pow (('*' | '/') pow)*
  const parseTerm = () => {
    let v = parsePow();
    if (v === null) return null;
    while (peek() === "*" || peek() === "/") {
      const op = tokens[i++];
      const r = parsePow();
      if (r === null) return null;
      v = op === "*" ? v * r : v / r;
    }
    return v;
  };
  // pow := unary ('**' pow)?   (right-associative)
  const parsePow = () => {
    const base = parseUnary();
    if (base === null) return null;
    if (peek() === "**") {
      i++;
      const exp = parsePow();
      return exp === null ? null : Math.pow(base, exp);
    }
    return base;
  };
  const parseUnary = () => {
    if (peek() === "-") {
      i++;
      const v = parseUnary();
      return v === null ? null : -v;
    }
    if (peek() === "+") {
      i++;
      return parseUnary();
    }
    return parseAtom();
  };
  const parseAtom = () => {
    const t = peek();
    if (t === undefined) return null;
    if (t === "(") {
      i++;
      const v = parseExpr();
      if (v === null || tokens[i++] !== ")") return null;
      return v;
    }
    if (/^\d/.test(t)) {
      i++;
      return Number(t);
    }
    if (/^[A-Za-z_]/.test(t)) {
      i++;
      return 1; // symbolic dim -> 1
    }
    return null; // an operator where a value was expected
  };

  const result = parseExpr();
  if (result === null || i !== tokens.length || !Number.isFinite(result)) {
    return null;
  }
  return result;
}

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
// { field, wire, value?, bytes? }. Exported so sibling modules (e.g.
// shapes.mjs, which reads the graph's input/output dim_params) can reuse this
// one dependency-free wire reader instead of duplicating it.
export function* fields(buf) {
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
// Decode a length-delimited protobuf field's bytes as UTF-8. Exported for
// reuse by shapes.mjs (dim_param names are protobuf strings).
export const decode = (bytes) => utf8.decode(bytes);

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
    // Substituted (dims -> 1) numbers; raw formulas kept in `raw`.
    macs: evalMetric(meta.macs),
    flops: evalMetric(meta.flops),
    memAccess: evalMetric(meta.mem_access),
    raw: meta,
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
  if (n == null) return "—";
  if (!Number.isFinite(+n)) return String(n);
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
// Symbolic metrics are substituted (dims -> 1) first, so a dynamic-shape model
// reports its per-sample throughput. Returns null only when the metric is
// missing/unparseable or the latency is non-positive. `avgMs` is milliseconds.
export function throughput(model, avgMs) {
  const macs = evalMetric(model?.macs);
  if (macs === null || !(avgMs > 0)) return null;
  const flops = 2 * macs;
  return {
    macs,
    flops,
    gflops: flops / (avgMs / 1000) / 1e9,
    gmacs: macs / (avgMs / 1000) / 1e9,
  };
}
