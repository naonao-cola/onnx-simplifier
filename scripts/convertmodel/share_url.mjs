// "Share link" button: build a URL that reproduces the current converter
// configuration and copy it to the clipboard.
//
// The input source (model / graph / backend) and the processor already get
// reflected into the address bar live (syncInputUrl / syncUrlParam in
// query_params.mjs), but the other conversion options — constant fold, shape
// inference, tensor size threshold, target opset — are only *read* from the URL
// on load, not written back when the user changes a control. buildShareSearch
// closes that gap: it overlays every option's current value onto the live query
// string, so the copied link carries the whole configuration.
//
// Clicking Share also updates the address bar (history.replaceState) to the same
// full URL, so a manual copy of the location bar matches what was copied.

import { buildShareSearch } from "./query_params.mjs";

const btn = document.getElementById("share-link-button");
const statusEl = document.getElementById("share-link-status");

// The absolute, shareable URL for the current configuration. Built from the page
// URL (minus any existing query/hash) plus the freshly composed query string,
// preserving the fragment. Works for file:// as well as served pages.
function currentShareUrl() {
  const search = buildShareSearch(window.location.search, document);
  const base = window.location.href.split(/[?#]/)[0];
  return base + search + window.location.hash;
}

// Copy `text` to the clipboard, preferring the async Clipboard API and falling
// back to a hidden-textarea + execCommand for older / insecure-context browsers.
// Returns whether the copy is believed to have succeeded.
async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    // Keep it off-screen and non-disruptive while still selectable.
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    ta.setAttribute("readonly", "");
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

let clearTimer = null;
function flash(msg) {
  if (!statusEl) return;
  statusEl.textContent = msg;
  if (clearTimer) clearTimeout(clearTimer);
  clearTimer = setTimeout(() => {
    statusEl.textContent = "";
    clearTimer = null;
  }, 4000);
}

if (btn) {
  btn.addEventListener("click", async () => {
    const url = currentShareUrl();
    // Reflect the full configuration in the address bar too, so copying the
    // location bar by hand yields the same link. Best-effort.
    try {
      window.history.replaceState(null, "", url);
    } catch {
      // ignore — a failed URL update must not break the copy.
    }
    const ok = await copyText(url);
    flash(ok ? "link copied to clipboard ✓" : "copy failed — select the address bar and copy it");
  });
}
