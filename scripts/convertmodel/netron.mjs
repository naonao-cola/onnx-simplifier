// Shared helpers for handing a model to Netron (https://netron.app) straight
// from the browser, with no server round trip. Kept dependency-free and pure so
// the Node test can exercise the exact URL-building logic the page uses.
//
// How Netron opens a model from a URL (see netron's browser host `start()`):
//   * the model URL is read from the location *hash* (falling back to the
//     `url` query parameter);
//   * the `identifier` query parameter is used for format detection, so it must
//     carry the file name (e.g. "model.onnx");
//   * `data:` URLs are fetched verbatim — Netron skips its usual cache-busting
//     query for them — so an in-memory model encoded as a data URL loads
//     directly, cross-origin, without uploading anything.
//
// Putting the (potentially large) data URL in the hash rather than the query
// string keeps the base64 padding/`+`/`/` characters intact: the hash is used
// raw, while the query string would be run through URLSearchParams.

export const NETRON_BASE = "https://netron.app/";

// Above this data-URL length we skip the inline <iframe> and only offer an
// "open in a new tab" link: browsers refuse to navigate to extremely long URLs,
// and embedding a multi-megabyte frame src is a poor experience anyway.
export const NETRON_INLINE_MAX = 6 * 1024 * 1024;

// Build the Netron URL that opens `dataUrl` (a `data:...;base64,...` string) and
// labels it `name` so Netron picks the right parser from the extension.
export function buildNetronUrl(dataUrl, name) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) {
    throw new Error("buildNetronUrl expects a data: URL");
  }
  const identifier = encodeURIComponent(name || "model.onnx");
  return `${NETRON_BASE}?identifier=${identifier}#${dataUrl}`;
}

// Whether a data URL is small enough to embed inline in an <iframe>.
export function canEmbedInline(dataUrl) {
  return typeof dataUrl === "string" && dataUrl.length <= NETRON_INLINE_MAX;
}
