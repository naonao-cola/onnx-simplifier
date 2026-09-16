// Unit test for onnx_metadata_writer.mjs's attachWebgpuKernelSpec: does it
// actually write a real, readable-back metadata entry, and does it leave
// everything else in the file alone? No browser, no GPU, no Python --
// pure-JS, against a real committed .onnx fixture
// (webgpu_tinygrad_conv3d.onnx, which already carries one attached kernel
// spec on its flagged Conv node, from make_webgpu_tinygrad_codegen_fixture.py).
//
// "Everything else round-trips byte-identical" is checked at the level the
// module's own docstring promises: every OTHER node's raw serialized bytes
// (via the module's own readAllFields, exported as _internal for exactly
// this) are compared before/after, so this isn't just "the JSON I asked for
// reads back" -- it's "nothing else moved".
//
// Usage:
//   node test/onnx_metadata_writer.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { attachWebgpuKernelSpec, WEBGPU_KERNEL_METADATA_KEY, _internal } from "../onnx_metadata_writer.mjs";
import { readWebgpuKernelSpecs } from "../onnx_node_metadata.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = new Uint8Array(readFileSync(join(HERE, "webgpu_tinygrad_conv3d.onnx")));

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

// Every node's raw bytes, keyed by name, via the module's own field walker --
// dogfooding attachWebgpuKernelSpec's own parsing so "did this node's bytes
// change" is checked the same way the module itself sees the file.
function nodeBytesByName(modelBytes) {
  const modelFields = _internal.readAllFields(modelBytes);
  const graphField = modelFields.find((f) => f.field === 7 && f.wireType === 2);
  const graphFields = _internal.readAllFields(graphField.value);
  const result = new Map();
  for (const f of graphFields) {
    if (f.field !== 1 || f.wireType !== 2) continue;
    const nodeFields = _internal.readAllFields(f.value);
    const nameField = nodeFields.find((nf) => nf.field === 3 && nf.wireType === 2);
    const name = nameField ? new TextDecoder().decode(nameField.value) : "";
    result.set(name, f.value);
  }
  return result;
}

function main() {
  const originalSpecs = readWebgpuKernelSpecs(FIXTURE);
  const [nodeName, originalSpec] = [...originalSpecs.entries()][0];

  check("fixture has exactly one node with an attached kernel spec to start", () => {
    assert.equal(originalSpecs.size, 1);
    assert.ok(nodeName);
  });

  const newSpec = {
    steps: [
      {
        wgsl: "@compute @workgroup_size(1) fn tuned_kernel() {}",
        entry_point: "tuned_kernel",
        dispatch: [4, 1, 1],
        bindings: [],
      },
    ],
    intermediates: {},
  };
  const updated = attachWebgpuKernelSpec(FIXTURE, nodeName, newSpec);

  check("the target node's spec reads back exactly as attached", () => {
    const specs = readWebgpuKernelSpecs(updated);
    assert.deepEqual(specs.get(nodeName), newSpec);
    assert.notDeepEqual(specs.get(nodeName), originalSpec);
  });

  check("attaching replaces the old entry rather than appending a duplicate", () => {
    // A duplicate WEBGPU_KERNEL_METADATA_KEY entry would still deep-equal
    // newSpec via JSON.parse of whichever one readWebgpuKernelSpecs's own
    // Map happens to keep, so check at the raw metadata_props level instead.
    const before = nodeBytesByName(FIXTURE).get(nodeName);
    const after = nodeBytesByName(updated).get(nodeName);
    assert.notDeepEqual(before, after);
    const afterFields = _internal.readAllFields(after);
    const metadataEntries = afterFields.filter((f) => f.field === 9 && f.wireType === 2);
    const kernelKeyEntries = metadataEntries.filter((f) => {
      const entryFields = _internal.readAllFields(f.value);
      const key = entryFields.find((ef) => ef.field === 1 && ef.wireType === 2);
      return key && new TextDecoder().decode(key.value) === WEBGPU_KERNEL_METADATA_KEY;
    });
    assert.equal(kernelKeyEntries.length, 1);
  });

  check("every other node's raw bytes are untouched", () => {
    const before = nodeBytesByName(FIXTURE);
    const after = nodeBytesByName(updated);
    assert.equal(after.size, before.size, "node count changed");
    for (const [name, bytesBefore] of before) {
      if (name === nodeName) continue;
      const bytesAfter = after.get(name);
      assert.ok(bytesAfter, `node ${JSON.stringify(name)} disappeared`);
      assert.deepEqual(Array.from(bytesAfter), Array.from(bytesBefore), `node ${JSON.stringify(name)} changed`);
    }
  });

  check("the rest of the model (outside graph.node) is untouched", () => {
    // graph.node is field 1 inside graph (field 7 inside the model); every
    // OTHER field at either level should be byte-identical, in order.
    const beforeModel = _internal.readAllFields(FIXTURE).filter((f) => !(f.field === 7 && f.wireType === 2));
    const afterModel = _internal.readAllFields(updated).filter((f) => !(f.field === 7 && f.wireType === 2));
    assert.equal(afterModel.length, beforeModel.length);
    for (let i = 0; i < beforeModel.length; i++) {
      assert.equal(afterModel[i].field, beforeModel[i].field);
      assert.equal(afterModel[i].wireType, beforeModel[i].wireType);
      if (beforeModel[i].wireType === 0) {
        assert.equal(afterModel[i].value, beforeModel[i].value);
      } else {
        assert.deepEqual(Array.from(afterModel[i].value), Array.from(beforeModel[i].value));
      }
    }

    const beforeGraph = _internal.readAllFields(
      _internal.readAllFields(FIXTURE).find((f) => f.field === 7 && f.wireType === 2).value,
    );
    const afterGraph = _internal.readAllFields(
      _internal.readAllFields(updated).find((f) => f.field === 7 && f.wireType === 2).value,
    );
    const beforeGraphOther = beforeGraph.filter((f) => !(f.field === 1 && f.wireType === 2));
    const afterGraphOther = afterGraph.filter((f) => !(f.field === 1 && f.wireType === 2));
    assert.equal(afterGraphOther.length, beforeGraphOther.length);
    for (let i = 0; i < beforeGraphOther.length; i++) {
      assert.deepEqual(Array.from(afterGraphOther[i].value ?? []), Array.from(beforeGraphOther[i].value ?? []));
    }
  });

  check("attaching twice in a row still leaves exactly one entry (idempotent under re-tuning)", () => {
    const twice = attachWebgpuKernelSpec(attachWebgpuKernelSpec(FIXTURE, nodeName, newSpec), nodeName, originalSpec);
    assert.deepEqual(readWebgpuKernelSpecs(twice).get(nodeName), originalSpec);
    const afterFields = _internal.readAllFields(nodeBytesByName(twice).get(nodeName));
    const kernelKeyEntries = afterFields.filter((f) => {
      if (f.field !== 9 || f.wireType !== 2) return false;
      const entryFields = _internal.readAllFields(f.value);
      const key = entryFields.find((ef) => ef.field === 1 && ef.wireType === 2);
      return key && new TextDecoder().decode(key.value) === WEBGPU_KERNEL_METADATA_KEY;
    });
    assert.equal(kernelKeyEntries.length, 1);
  });

  check("attaching to a nonexistent node raises clearly", () => {
    assert.throws(() => attachWebgpuKernelSpec(FIXTURE, "no_such_node", newSpec), /no node named/);
  });

  console.log(`\nonnx metadata writer: ${passed} checks passed`);
}

main();
