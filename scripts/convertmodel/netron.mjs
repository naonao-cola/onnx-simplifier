// Shared helpers for handing a model to Netron straight from the browser, with
// no server round trip. Kept dependency-free and pure so the Node test can
// exercise the exact logic the page uses.
//
// The converter self-hosts a build of Netron (see the deploy workflow, which
// copies it to ./netron/) that speaks a postMessage embedding protocol modeled
// on Perfetto's. Rather than packing the model into the URL — which the browser
// caps at ~2 MB and rejects with `about:blank#blocked` for anything larger — we
// load Netron in a frame/window and post the raw bytes into it:
//
//   1. Post the string "PING" until Netron replies "PONG" (it is then ready).
//   2. Post `{ netron: { buffer, name, identifier } }` with the model bytes.
//   3. Netron replies `{ netron: { status: "success" | "error", ... } }`.
//
// This carries any size, since nothing goes through a URL, and nothing is
// uploaded — the bytes stay in-process between same-origin windows.

// Where the self-hosted Netron lives, relative to the converter page. The
// `embed` flag puts Netron in embedded mode (no update/consent dialogs, no
// telemetry) so it is ready to be driven by the protocol.
export const NETRON_PATH = "./netron/";
export const NETRON_EMBED_URL = `${NETRON_PATH}?embed`;

// The readiness handshake strings, matching the Netron host.
export const NETRON_PING = "PING";
export const NETRON_PONG = "PONG";

// Normalize model bytes to a standalone ArrayBuffer holding exactly those
// bytes. Accepts an ArrayBuffer or any typed-array / DataView view, and always
// returns a fresh copy so the source is never detached when the buffer is
// posted (and a view into a larger buffer never leaks its neighbours).
export function toArrayBuffer(data) {
  if (data instanceof ArrayBuffer) {
    return data.slice(0);
  }
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  throw new Error("Expected an ArrayBuffer or a typed array.");
}

// Decode a `data:...;base64,...` URL (as produced by the converter worker) into
// an ArrayBuffer of the model bytes.
export function dataUrlToArrayBuffer(dataUrl) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) {
    throw new Error("dataUrlToArrayBuffer expects a data: URL");
  }
  const comma = dataUrl.indexOf(",");
  if (comma === -1) {
    throw new Error("Malformed data: URL");
  }
  const binary = atob(dataUrl.slice(comma + 1));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

// Build the message that hands `buffer` (an ArrayBuffer) to the embedded Netron.
// `name` labels the model so Netron picks the right parser from its extension.
export function buildModelMessage(buffer, name) {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new Error("buildModelMessage expects an ArrayBuffer");
  }
  const identifier = name || "model.onnx";
  return { netron: { buffer, name: identifier, identifier } };
}
