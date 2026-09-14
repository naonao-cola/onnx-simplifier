// Proves onnx_node_metadata.mjs's hand-rolled protobuf reader actually
// agrees with onnxsim/webgpu_kernel_metadata.py's writer -- not just that
// each side round-trips its own data, which wouldn't catch the two sides
// silently drifting apart (a changed key name, a reordered JSON field,
// wrong field numbers). webgpu_kernel_add.onnx was written by the real
// Python attach_webgpu_kernel() (see make_webgpu_kernel_fixture.py), and
// webgpu_kernel_fixture.json records the exact values passed in, so this
// test parses the real .onnx bytes in plain Node (no browser, no
// onnxruntime-web -- this is pure protobuf-field extraction) and checks the
// two match.
//
// Usage:
//   python3 make_webgpu_kernel_fixture.py   # regenerate fixtures if needed
//   node test/onnx_node_metadata.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  readNodeMetadataProps,
  readWebgpuKernelSpecs,
  WEBGPU_KERNEL_METADATA_KEY,
} from "../onnx_node_metadata.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(HERE, "webgpu_kernel_fixture.json"), "utf8"));
const modelBytes = new Uint8Array(readFileSync(join(HERE, manifest.file)));

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("finds exactly the named node with metadata_props", () => {
  const perNode = readNodeMetadataProps(modelBytes);
  assert.equal(perNode.size, 1, "expected exactly one node with metadata_props");
  assert.ok(perNode.has(manifest.nodeName), `expected node ${manifest.nodeName}`);
});

check("raw metadata_props entry uses the documented key", () => {
  const perNode = readNodeMetadataProps(modelBytes);
  const props = perNode.get(manifest.nodeName);
  assert.ok(props.has(WEBGPU_KERNEL_METADATA_KEY));
  // Must be valid JSON -- readNodeMetadataProps itself doesn't parse it.
  JSON.parse(props.get(WEBGPU_KERNEL_METADATA_KEY));
});

check("readWebgpuKernelSpecs decodes the kernel spec written by Python", () => {
  const specs = readWebgpuKernelSpecs(modelBytes);
  assert.equal(specs.size, 1);
  const spec = specs.get(manifest.nodeName);
  assert.equal(spec.entry_point, manifest.entryPoint);
  assert.deepEqual(spec.dispatch, manifest.dispatch);
  assert.deepEqual(spec.bindings, manifest.bindings);
  assert.equal(typeof spec.wgsl, "string");
  assert.ok(spec.wgsl.includes("@compute"));
});

check("a node with no metadata_props is simply absent from the map", () => {
  // The fixture's graph has exactly one node (Add) -- nothing else to
  // assert here beyond what the first check already covers, but this
  // documents the contract explicitly: readNodeMetadataProps.size counts
  // only nodes that have at least one metadata_props entry.
  const perNode = readNodeMetadataProps(modelBytes);
  for (const [name, props] of perNode) {
    assert.ok(props.size > 0, `node ${name} present with an empty metadata map`);
  }
});

console.log(`\nonnx node metadata: ${passed} checks passed`);
