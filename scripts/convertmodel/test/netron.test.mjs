// Unit test for the pure helpers behind the converter page's "Visualize with
// Netron" panel. No network or browser needed.
//
// Usage:
//   node test/netron.test.mjs

import assert from "node:assert/strict";
import {
  NETRON_EMBED_URL,
  NETRON_PATH,
  NETRON_PING,
  NETRON_PONG,
  toArrayBuffer,
  dataUrlToArrayBuffer,
  buildModelMessage,
} from "../netron.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("embed URL points at the self-hosted Netron in embedded mode", () => {
  assert.ok(NETRON_EMBED_URL.startsWith(NETRON_PATH));
  assert.ok(NETRON_EMBED_URL.includes("embed"));
  // Handshake strings must match the Netron host.
  assert.equal(NETRON_PING, "PING");
  assert.equal(NETRON_PONG, "PONG");
});

check("toArrayBuffer copies a typed array to an exact standalone buffer", () => {
  const src = new Uint8Array([1, 2, 3, 4]);
  const buf = toArrayBuffer(src);
  assert.ok(buf instanceof ArrayBuffer);
  assert.deepEqual(Array.from(new Uint8Array(buf)), [1, 2, 3, 4]);
  // A copy, not a view: mutating the source must not affect the result.
  src[0] = 99;
  assert.equal(new Uint8Array(buf)[0], 1);
});

check("toArrayBuffer copies only a view's slice of a larger buffer", () => {
  const backing = new Uint8Array([10, 11, 12, 13, 14, 15]);
  const view = new Uint8Array(backing.buffer, 2, 3); // [12, 13, 14]
  const buf = toArrayBuffer(view);
  assert.equal(buf.byteLength, 3);
  assert.deepEqual(Array.from(new Uint8Array(buf)), [12, 13, 14]);
});

check("toArrayBuffer clones an ArrayBuffer", () => {
  const src = new Uint8Array([5, 6, 7]).buffer;
  const buf = toArrayBuffer(src);
  assert.notEqual(buf, src);
  assert.deepEqual(Array.from(new Uint8Array(buf)), [5, 6, 7]);
});

check("toArrayBuffer rejects non-buffer input", () => {
  assert.throws(() => toArrayBuffer("nope"));
  assert.throws(() => toArrayBuffer(undefined));
});

check("dataUrlToArrayBuffer decodes base64 model bytes", () => {
  const buf = dataUrlToArrayBuffer("data:application/octet-stream;base64,QUJD"); // "ABC"
  assert.deepEqual(Array.from(new Uint8Array(buf)), [65, 66, 67]);
});

check("dataUrlToArrayBuffer rejects non-data URLs", () => {
  assert.throws(() => dataUrlToArrayBuffer("https://example.com/model.onnx"));
  assert.throws(() => dataUrlToArrayBuffer(undefined));
});

check("buildModelMessage wraps the buffer with an identifier", () => {
  const buffer = new Uint8Array([1, 2, 3]).buffer;
  const message = buildModelMessage(buffer, "my model.onnx");
  assert.equal(message.netron.buffer, buffer);
  assert.equal(message.netron.name, "my model.onnx");
  assert.equal(message.netron.identifier, "my model.onnx");
});

check("buildModelMessage falls back to a default identifier", () => {
  const message = buildModelMessage(new ArrayBuffer(0), "");
  assert.equal(message.netron.identifier, "model.onnx");
});

check("buildModelMessage requires an ArrayBuffer", () => {
  assert.throws(() => buildModelMessage(new Uint8Array([1]), "x"));
});

console.log(`PASS: ${passed} checks`);
