// Small browser helpers for saving data to a file via a temporary object URL.
// Shared by the converter page's "Download log" buttons and the Netron panes'
// "download model" buttons so the click-to-download logic lives in one place.

// Trigger a browser download of `blob` under `filename`.
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || "download";
  a.click();
  // Give the click a tick to start before revoking the URL.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Save a text string (e.g. console log output) as a file.
export function downloadText(text, filename) {
  downloadBlob(new Blob([text ?? ""], { type: "text/plain" }), filename);
}

// Save raw bytes (an ArrayBuffer or a typed-array view) as a file, e.g. a model.
export function downloadBytes(data, filename) {
  downloadBlob(new Blob([data], { type: "application/octet-stream" }), filename);
}
