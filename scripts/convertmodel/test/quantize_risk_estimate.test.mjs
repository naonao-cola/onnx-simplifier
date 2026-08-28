// Unit tests for the pure helpers behind the "Check quantization risk" button
// in the "Quantize a model" panel: the WASM-call wrapper's error handling and
// the HTML summary renderer. No DOM or a real WASM module needed -- the
// button's click-handler glue (initQuantizeRiskPanel) is browser-only,
// exactly like quantize_ui.mjs's own init function.
//
// Usage:
//   node test/quantize_risk_estimate.test.mjs

import assert from "node:assert/strict";
import {
  estimateQuantizationRisk,
  renderQuantizationRisk,
} from "../quantize_risk_estimate.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

const fakeBytes = new Uint8Array([1, 2, 3, 4]);

check("estimateQuantizationRisk passes the model bytes' buffer through and returns the runtime's result", () => {
  let receivedBuf = null;
  const fakeResult = { riskLevel: "safe" };
  const runtime = {
    onnxsim_estimate_quantization_drop(buf) {
      receivedBuf = buf;
      return fakeResult;
    },
  };
  const result = estimateQuantizationRisk(runtime, fakeBytes);
  assert.equal(result, fakeResult);
  assert.ok(receivedBuf instanceof ArrayBuffer, "an ArrayBuffer was passed to the WASM call");
});

check("estimateQuantizationRisk throws when the WASM call returns null (parse failure)", () => {
  const runtime = { onnxsim_estimate_quantization_drop: () => null };
  assert.throws(
    () => estimateQuantizationRisk(runtime, fakeBytes),
    /quantization risk estimate failed/,
  );
});

function baseEstimate(overrides) {
  return {
    riskLevel: "safe",
    totalNodesAnalyzed: 3,
    unsafeNodes: [],
    outlierRiskNodes: [],
    worstOutlierRatio: NaN,
    estimatedRelativeError: 0.001234,
    ...overrides,
  };
}

check("renders a safe result with the estimated error and no node lists", () => {
  const html = renderQuantizationRisk(baseEstimate());
  assert.ok(html.includes("safe"), "risk level shown");
  assert.ok(html.includes("3 MatMul/Gemm/Conv/Attention"), "node count shown");
  assert.ok(html.includes("0.12%"), "estimated relative error formatted as a percent");
  assert.ok(!html.includes("Unsafe (accumulator overflow)"), "no unsafe-node row when unsafeNodes is empty");
  assert.ok(!html.includes("Outlier-risk nodes"), "no outlier-node row when outlierRiskNodes is empty");
});

check("renders the worst outlier ratio when finite", () => {
  const html = renderQuantizationRisk(
    baseEstimate({ riskLevel: "degraded", worstOutlierRatio: 250.4, outlierRiskNodes: ["Conv_1"] }),
  );
  assert.ok(html.includes("degraded"), "risk level shown");
  assert.ok(html.includes("250.4"), "outlier ratio shown");
  assert.ok(html.includes("Conv_1"), "outlier node name listed");
});

check("renders the unsafe-node list and an n/a error when NaN", () => {
  const html = renderQuantizationRisk(
    baseEstimate({
      riskLevel: "unsafe",
      estimatedRelativeError: NaN,
      unsafeNodes: ["MatMul_0", "MatMul_1"],
    }),
  );
  assert.ok(html.includes("unsafe"), "risk level shown");
  assert.ok(html.includes("n/a"), "NaN error rendered as n/a, not literal NaN%");
  assert.ok(html.includes("MatMul_0") && html.includes("MatMul_1"), "unsafe node names listed");
});

check("truncates a long node list with a '+N more' suffix", () => {
  const many = Array.from({ length: 9 }, (_, i) => `Conv_${i}`);
  const html = renderQuantizationRisk(baseEstimate({ riskLevel: "degraded", outlierRiskNodes: many }));
  assert.ok(html.includes("Conv_0"), "first node shown");
  assert.ok(html.includes("+3 more"), "truncation suffix for the remaining 3 (9 - 6 shown)");
  assert.ok(!html.includes("Conv_8"), "9th node not individually listed");
});

check("HTML-escapes node names", () => {
  const html = renderQuantizationRisk(
    baseEstimate({ riskLevel: "unsafe", unsafeNodes: ['<script>alert(1)</script>'] }),
  );
  assert.ok(!html.includes("<script>alert"), "raw node name not injected verbatim");
  assert.ok(html.includes("&lt;script&gt;"), "node name HTML-escaped");
});

console.log(`PASS: ${passed} checks`);
