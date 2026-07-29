// Unit test for the Netron URL builder used by the converter page's
// "Visualize with Netron" panel. Pure logic, no network or browser needed.
//
// Usage:
//   node test/netron.test.mjs

import assert from "node:assert/strict";
import {
  NETRON_BASE,
  NETRON_INLINE_MAX,
  buildNetronUrl,
  canEmbedInline,
} from "../netron.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

const DATA_URL = "data:application/octet-stream;base64,QUJD"; // "ABC"

check("builds a netron URL with the data URL in the hash", () => {
  const url = buildNetronUrl(DATA_URL, "model.onnx");
  assert.ok(url.startsWith(NETRON_BASE), "starts with the Netron base URL");
  const [head, hash] = url.split("#");
  // The (possibly large) data URL must live in the hash, untouched, so its
  // base64 padding and +/ characters survive.
  assert.equal(hash, DATA_URL);
  // The file name rides along as the `identifier` query param for format
  // detection.
  assert.ok(head.includes("identifier=model.onnx"));
});

check("percent-encodes the identifier file name", () => {
  const url = buildNetronUrl(DATA_URL, "my model.onnx");
  assert.ok(url.includes("identifier=my%20model.onnx"));
  assert.ok(url.endsWith(`#${DATA_URL}`), "data URL still intact in the hash");
});

check("falls back to a default identifier", () => {
  const url = buildNetronUrl(DATA_URL, "");
  assert.ok(url.includes("identifier=model.onnx"));
});

check("rejects non-data URLs", () => {
  assert.throws(() => buildNetronUrl("https://example.com/model.onnx", "x"));
  assert.throws(() => buildNetronUrl(undefined, "x"));
});

check("canEmbedInline respects the size threshold", () => {
  assert.equal(canEmbedInline(DATA_URL), true);
  const big = "data:application/octet-stream;base64," + "A".repeat(NETRON_INLINE_MAX);
  assert.equal(canEmbedInline(big), false);
  assert.equal(canEmbedInline(undefined), false);
});

console.log(`PASS: ${passed} checks`);
