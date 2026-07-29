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

// Chromium refuses to *navigate* to a URL longer than url::kMaxURLChars
// (2 MiB) and shows `about:blank#blocked` instead. Because we hand the whole
// model to Netron as a base64 `data:` URL packed into the URL hash, that cap is
// the real ceiling on both the inline <iframe> src and the "open in a new tab"
// link. Base64 inflates the model by ~4/3, so anything past ~1.5 MiB overflows
// it. Other engines have their own limits (Firefox is far higher, older Safari
// lower); 2 MiB is the safe common denominator that matches the block we see.
export const BROWSER_URL_MAX = 2 * 1024 * 1024;

// Keep headroom below the hard cap for the `?identifier=<name>#` prefix and for
// engines with a slightly lower limit, so we never hand out a URL right at the
// edge that the browser then blocks.
export const NETRON_URL_MAX = BROWSER_URL_MAX - 8 * 1024;

// Build the Netron URL that opens `dataUrl` (a `data:...;base64,...` string) and
// labels it `name` so Netron picks the right parser from the extension.
export function buildNetronUrl(dataUrl, name) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) {
    throw new Error("buildNetronUrl expects a data: URL");
  }
  const identifier = encodeURIComponent(name || "model.onnx");
  return `${NETRON_BASE}?identifier=${identifier}#${dataUrl}`;
}

// Whether the Netron URL for this model fits under the browser's navigation
// limit. When it does, both the inline <iframe> and the "open in a new tab"
// link work; when it doesn't, the browser blocks the navigation
// (`about:blank#blocked`), so the caller must fall back to a local download.
// The check is on the *whole* built URL, not just the data URL, since the
// `identifier` prefix counts against the same cap.
export function netronUrlFits(dataUrl, name) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) return false;
  return buildNetronUrl(dataUrl, name).length <= NETRON_URL_MAX;
}
