// MAC-metrics test for the "Run inference" panel.
//
// Verifies that macs.mjs reads the onnxsim MAC/FLOP annotations (onnxsim
// PR #527) out of the fixture model's metadata_props, and that the throughput
// helper turns a model's FLOPs + a latency into GFLOP/s. This drives the exact
// same macs.mjs the browser panel uses, so a green run means the metadata path
// works end to end. Regenerate the annotated fixture with test/make_fixture.py.
//
//   node test/macs.test.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";
import { readAnnotations, perOpSummary, throughput, humanNum } from "../macs.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const model = new Uint8Array(readFileSync(join(HERE, "model.onnx")));

const ann = readAnnotations(model);
console.log("model metrics:", ann.model);
for (const n of ann.nodes) console.log(`  ${n.opType.padEnd(8)} ${String(n.name).padEnd(6)} macs=${n.macs}`);

// The fixture is MatMul([1,4]x[4,3]) -> Add -> Relu: 1*3*4 = 12 MACs.
assert.equal(ann.annotated, true, "fixture must carry onnxsim.* metadata_props");
assert.equal(ann.model.macs, "12", "model.macs");
assert.equal(ann.model.flops, "24", "model.flops");

const matmul = ann.nodes.find((n) => n.opType === "MatMul");
assert.ok(matmul, "MatMul node present");
assert.equal(matmul.macs, 12, "MatMul node macs");
assert.equal(ann.nodes.find((n) => n.opType === "Relu").macs, 0, "Relu macs = 0");

const summary = perOpSummary(ann.nodes);
assert.equal(summary.totalMacs, 12, "summed node MACs == model total");
assert.equal(summary.rows[0].opType, "MatMul", "MatMul dominates the MAC total");

// Throughput: 24 FLOPs over a 2 ms average = 24 / 0.002 = 12000 FLOP/s.
const tp = throughput(ann.model, 2.0);
assert.ok(tp, "throughput computed for a concrete FLOP count");
assert.ok(Math.abs(tp.gflops - 24 / 0.002 / 1e9) < 1e-12, "GFLOP/s");
assert.equal(throughput({ macs: "512*batch" }, 2.0), null, "symbolic FLOPs -> no throughput");

// Formatter sanity.
assert.equal(humanNum(1622336), "1.6M");

console.log(`\nMACs ${ann.model.macs} (${humanNum(+ann.model.macs)}), MatMul dominates; throughput helper OK`);
console.log("PASS: macs.mjs reads onnxsim metadata_props and computes throughput");
