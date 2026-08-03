// Unit tests for cdn.mjs — the single source of truth for the converter page's
// external CDN runtime dependencies. These guard the invariants that made the
// old triplicated constants risky: the wasm binaries and JS glue must share one
// versioned onnxruntime-web directory, and every declared endpoint must be an
// https URL on the expected origin.
//
// Usage:
//   node test/cdn.test.mjs

import assert from "node:assert/strict";
import {
  ORT_VERSION,
  ORT_BASE,
  HF_HUB_VERSION,
  HF_HUB_ESM,
  PERFETTO_ORIGIN,
  PERFETTO_EMBED_SRC,
} from "../cdn.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("ORT_BASE points at the pinned onnxruntime-web version's dist/", () => {
  assert.equal(
    ORT_BASE,
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
  );
  // The glue and the .wasm are pulled from this same directory, so the version
  // must appear in the base — the drift the old copies risked.
  assert.ok(ORT_BASE.includes(ORT_VERSION));
  assert.ok(ORT_BASE.endsWith("/dist/"));
});

check("HF_HUB_ESM points at the pinned @huggingface/hub version", () => {
  assert.equal(HF_HUB_ESM, `https://esm.sh/@huggingface/hub@${HF_HUB_VERSION}`);
  assert.ok(HF_HUB_ESM.includes(HF_HUB_VERSION));
});

check("PERFETTO_EMBED_SRC is the embedded-mode URL on PERFETTO_ORIGIN", () => {
  assert.ok(PERFETTO_EMBED_SRC.startsWith(PERFETTO_ORIGIN + "/"));
  assert.ok(PERFETTO_EMBED_SRC.includes("mode=embedded"));
});

check("every declared endpoint is an https URL on its expected origin", () => {
  const expect = [
    [ORT_BASE, "cdn.jsdelivr.net"],
    [HF_HUB_ESM, "esm.sh"],
    [PERFETTO_ORIGIN, "ui.perfetto.dev"],
    [PERFETTO_EMBED_SRC, "ui.perfetto.dev"],
  ];
  for (const [value, host] of expect) {
    const url = new URL(value);
    assert.equal(url.protocol, "https:", `${value} must be https`);
    assert.equal(url.host, host, `${value} must be on ${host}`);
  }
});

console.log(`PASS: ${passed} checks`);
