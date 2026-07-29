// Unit test for the Netron URL builder used by the converter page's
// "Visualize with Netron" panel. Pure logic, no network or browser needed.
//
// Usage:
//   node test/netron.test.mjs

import assert from "node:assert/strict";
import {
  NETRON_BASE,
  NETRON_URL_MAX,
  BROWSER_URL_MAX,
  buildNetronUrl,
  netronUrlFits,
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

check("netronUrlFits gates on the browser URL cap", () => {
  // The safe threshold must stay under the browser's hard navigation limit,
  // otherwise the URLs we hand out get blocked as `about:blank#blocked`.
  assert.ok(NETRON_URL_MAX < BROWSER_URL_MAX);

  assert.equal(netronUrlFits(DATA_URL, "model.onnx"), true);

  // A data URL whose *whole* built URL lands just under the cap fits; one just
  // over it does not. The `identifier` prefix counts against the same budget,
  // so the check is on buildNetronUrl(...).length, not the data URL alone.
  const prefix = "data:application/octet-stream;base64,";
  const fill = (total) => prefix + "A".repeat(total - prefix.length);
  const overhead = buildNetronUrl(prefix, "model.onnx").length - prefix.length;

  const justFits = fill(NETRON_URL_MAX - overhead);
  const justOver = fill(NETRON_URL_MAX - overhead + 1);
  assert.equal(buildNetronUrl(justFits, "model.onnx").length, NETRON_URL_MAX);
  assert.equal(netronUrlFits(justFits, "model.onnx"), true);
  assert.equal(netronUrlFits(justOver, "model.onnx"), false);

  assert.equal(netronUrlFits(undefined, "model.onnx"), false);
  assert.equal(netronUrlFits("https://example.com/m.onnx", "m"), false);
});

console.log(`PASS: ${passed} checks`);
