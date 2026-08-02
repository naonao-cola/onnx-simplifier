// shapes.mjs -- read a model's graph input/output shapes, and in particular the
// symbolic `dim_param` names, straight out of the ONNX model bytes.
//
// An ONNX tensor axis is either a fixed size (`dim_value`) or a named symbolic
// dimension (`dim_param`, e.g. "batch" / "sequence"). onnxruntime-web surfaces
// input shapes through its JS API but not output shapes, and it collapses
// symbolic axes to nulls without their names -- so, like macs.mjs, this walks
// the model protobuf directly with the tiny dependency-free wire reader that
// module already carries (`fields` / `decode`), reading only the handful of
// fields needed to reach each Dimension. The converter page's "Dynamic
// dimensions" panel uses it to list a model's dim_params before vs after
// simplify/optimize; the Node test in test/shapes.test.mjs drives the same code.
//
// Field numbers below are from onnx.proto:
//   GraphProto            input = 11, output = 12   (repeated ValueInfoProto)
//   ValueInfoProto        name = 1, type = 2
//   TypeProto             tensor_type = 1           (TypeProto.Tensor)
//   TypeProto.Tensor      elem_type = 1, shape = 2  (TensorShapeProto)
//   TensorShapeProto      dim = 1                    (repeated Dimension)
//   Dimension             dim_value = 1, dim_param = 2

import { fields, decode } from "./macs.mjs";

// TensorShapeProto.Dimension: a fixed size (`dim_value`, field 1, varint) or a
// symbolic name (`dim_param`, field 2, string). Neither present means an
// unknown/unnamed axis. Returns { value } | { param } | { unknown: true }.
function parseDimension(buf) {
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 0) return { value: f.value };
    if (f.field === 2 && f.wire === 2) return { param: decode(f.bytes) };
  }
  return { unknown: true };
}

// TensorShapeProto: dim = 1 (repeated Dimension). Order is the axis order.
function parseTensorShape(buf) {
  const dims = [];
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) dims.push(parseDimension(f.bytes));
  }
  return dims;
}

// TypeProto.Tensor: shape = 2 (TensorShapeProto). Absent shape => unknown rank,
// reported as null (distinct from an empty [] which is a rank-0 scalar).
function parseTensorType(buf) {
  let shape = null;
  for (const f of fields(buf)) {
    if (f.field === 2 && f.wire === 2) shape = parseTensorShape(f.bytes);
  }
  return shape;
}

// TypeProto: tensor_type = 1. Only tensor types carry a dim shape; sequence /
// map / optional / sparse types have no dim_params to list, so they yield null.
function parseType(buf) {
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) return parseTensorType(f.bytes);
  }
  return null;
}

// ValueInfoProto: name = 1, type = 2. `shape` is null when the value is not a
// shaped tensor (see parseType / parseTensorType).
function parseValueInfo(buf) {
  let name = "";
  let shape = null;
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) name = decode(f.bytes);
    else if (f.field === 2 && f.wire === 2) shape = parseType(f.bytes);
  }
  return { name, shape };
}

// GraphProto: input = 11, output = 12 (both repeated ValueInfoProto).
function parseGraphIO(buf) {
  const inputs = [];
  const outputs = [];
  for (const f of fields(buf)) {
    if (f.field === 11 && f.wire === 2) inputs.push(parseValueInfo(f.bytes));
    else if (f.field === 12 && f.wire === 2) outputs.push(parseValueInfo(f.bytes));
  }
  return { inputs, outputs };
}

// ModelProto: graph = 7. Read the graph's input/output tensor shapes from the
// model bytes. Returns { inputs, outputs }, each an array of
// { name, shape } where shape is [Dimension, ...] or null (unknown rank).
export function readShapes(modelBytes) {
  const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  let io = { inputs: [], outputs: [] };
  for (const f of fields(buf)) {
    if (f.field === 7 && f.wire === 2) io = parseGraphIO(f.bytes);
  }
  return io;
}

// True for a non-empty symbolic (dim_param) axis.
export function isParamDim(dim) {
  return dim != null && typeof dim.param === "string" && dim.param !== "";
}

// One axis -> a short token: the symbolic name, the fixed size, or "?" for an
// unknown/unnamed axis.
export function formatDim(dim) {
  if (dim == null) return "?";
  if (isParamDim(dim)) return dim.param;
  if (dim.value != null) return String(dim.value);
  return "?";
}

// A whole shape -> "[batch, 4]". null rank -> "(unknown rank)"; [] -> "scalar".
export function formatShape(shape) {
  if (shape == null) return "(unknown rank)";
  if (shape.length === 0) return "scalar";
  return "[" + shape.map(formatDim).join(", ") + "]";
}

// Unique dim_param names across a model's inputs and outputs, in first-seen
// order (inputs before outputs). This is the "list of dim_params".
export function dimParamNames(shapes) {
  const seen = new Set();
  const names = [];
  for (const group of [shapes.inputs, shapes.outputs]) {
    for (const t of group) {
      for (const dim of t.shape || []) {
        if (isParamDim(dim) && !seen.has(dim.param)) {
          seen.add(dim.param);
          names.push(dim.param);
        }
      }
    }
  }
  return names;
}

// For each dim_param name, where it occurs: an array of
// { io: "input"|"output", tensor, axis }. Keyed first-seen order, so callers
// can iterate it as a listing. Lets the UI show *which* tensor axis each
// symbolic dimension drives, not just the bare names.
export function dimParamOccurrences(shapes) {
  const map = new Map();
  const scan = (group, io) => {
    for (const t of group) {
      (t.shape || []).forEach((dim, axis) => {
        if (isParamDim(dim)) {
          if (!map.has(dim.param)) map.set(dim.param, []);
          map.get(dim.param).push({ io, tensor: t.name, axis });
        }
      });
    }
  };
  scan(shapes.inputs, "input");
  scan(shapes.outputs, "output");
  return map;
}

// Compare two models' dim_param name sets. Returns { before, after, common,
// removed, added } as string arrays (removed = in before but not after; added =
// in after but not before), so the UI can highlight how simplification changed
// the model's dynamic axes.
export function diffDimParams(beforeShapes, afterShapes) {
  const before = dimParamNames(beforeShapes);
  const after = dimParamNames(afterShapes);
  const afterSet = new Set(after);
  const beforeSet = new Set(before);
  return {
    before,
    after,
    common: before.filter((n) => afterSet.has(n)),
    removed: before.filter((n) => !afterSet.has(n)),
    added: after.filter((n) => !beforeSet.has(n)),
  };
}
