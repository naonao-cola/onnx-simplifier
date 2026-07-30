// Node verification for onnx-macs.js: confirms the metadata_props reader pulls
// the onnxsim (PR #527) MAC/FLOP annotations out of the model correctly, and
// that onnxruntime-web's WASM backend actually runs the same model. This is a
// developer check for the browser demo in index.html, not part of the demo.
//
//   npm install onnxruntime-web
//   node test_node.mjs [model.annotated.onnx]

import fs from "node:fs";
import { readAnnotations, perOpSummary, runInference } from "./onnx-macs.js";

const modelPath = process.argv[2] || "sample_model.annotated.onnx";
const bytes = new Uint8Array(fs.readFileSync(modelPath));

// 1. Parse the annotations.
const ann = readAnnotations(bytes);
console.log("== model-level onnxsim metrics ==");
console.log(ann.model);
console.log("\n== per-node ==");
for (const n of ann.nodes) {
  console.log(`  ${n.opType.padEnd(18)} ${String(n.name).padEnd(12)} macs=${n.macs}`);
}
const summary = perOpSummary(ann.nodes);
console.log("\n== per-op MAC summary ==");
for (const r of summary.rows) console.log(`  ${r.opType.padEnd(18)} x${r.count}  macs=${r.macs}`);
console.log(`  TOTAL (summed from nodes) = ${summary.totalMacs}`);
console.log("\n== feed inputs ==", ann.feedInputs);

// 2. Assertions against the known demo model.
let failures = 0;
function check(name, got, want) {
  const ok = String(got) === String(want);
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}: got=${got} want=${want}`);
  if (!ok) failures++;
}
if (modelPath.includes("sample_model")) {
  console.log("\n== assertions (demo model) ==");
  check("model.macs", ann.model.macs, "1622336");
  check("model.flops", ann.model.flops, "3244672");
  check("node total == model.macs", summary.totalMacs, 1622336);
  check("conv1 macs", ann.nodes.find((n) => n.name === "conv1")?.macs, 442368);
  check("conv2 macs", ann.nodes.find((n) => n.name === "conv2")?.macs, 1179648);
  check("classifier(Gemm) macs", ann.nodes.find((n) => n.name === "classifier")?.macs, 320);
}

// 3. Real onnxruntime-web (WASM) inference over the same bytes.
console.log("\n== onnxruntime-web (wasm) inference ==");
try {
  const ort = await import("onnxruntime-web");
  const ortmod = ort.default ?? ort;
  const { avgMs, outputNames } = await runInference(ortmod, bytes, { runs: 20 });
  const macs = Number(ann.model.macs);
  const flops = macs * 2;
  const gflops = flops / (avgMs / 1000) / 1e9;
  console.log(`  outputs: ${outputNames}`);
  console.log(`  avg inference: ${avgMs.toFixed(3)} ms`);
  console.log(`  MACs (from PR #527 annotation): ${macs.toLocaleString()}`);
  console.log(`  achieved: ${gflops.toFixed(2)} GFLOP/s`);
} catch (e) {
  console.log("  inference step error:", e.message);
  failures++;
}

console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : failures + " CHECK(S) FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
