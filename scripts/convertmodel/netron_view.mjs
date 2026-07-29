// Browser glue for the "Visualize with Netron" panel on the converter page.
//
// Shows the model graph before and after simplify/optimize side by side, each
// rendered by a self-hosted Netron (see ./netron/) in an <iframe>. The model is
// posted into Netron as raw bytes over the embedding protocol (see netron.mjs),
// so nothing leaves the page and there is no model-size limit.
//
// The "before" pane is driven here by watching the file input directly. The
// "after" pane is driven by the converter worker code, which calls the
// `window.netronShowAfter(dataUrl, name)` hook this module installs once a
// conversion finishes.

import {
  NETRON_EMBED_URL,
  NETRON_PING,
  NETRON_PONG,
  toArrayBuffer,
  dataUrlToArrayBuffer,
  buildModelMessage,
} from "./netron.mjs";

// Read a File as an ArrayBuffer.
function fileToArrayBuffer(file) {
  return file.arrayBuffer();
}

// Netron is served from the converter's own origin, so we only ever talk to a
// same-origin window and can pin the target origin.
const NETRON_ORIGIN = window.location.origin;

// Drive one Netron window (an <iframe>'s contentWindow or a popup) through the
// readiness handshake and hand it the model. Returns a function that tears the
// handshake down; call it before starting another one on the same window.
function driveNetron(target, buffer, name, onStatus) {
  let settled = false;
  const message = buildModelMessage(buffer, name);
  const onMessage = (e) => {
    if (e.source !== target) {
      return;
    }
    if (e.data === NETRON_PONG) {
      // Ready — hand over the model. Post a copy (no transfer) so the caller's
      // bytes stay usable for the other pane / the inference panel.
      target.postMessage(message, NETRON_ORIGIN);
    } else if (e.data && typeof e.data === "object" && e.data.netron && e.data.netron.status) {
      teardown();
      if (onStatus) {
        onStatus(e.data.netron);
      }
    }
  };
  const ping = setInterval(() => {
    // Netron ignores pings until it is ready, so keep pinging.
    try {
      target.postMessage(NETRON_PING, NETRON_ORIGIN);
    } catch {
      // The window may be gone (popup closed); stop trying.
      teardown();
    }
  }, 50);
  // Stop pinging even if Netron never answers, so we don't ping forever.
  const timeout = setTimeout(teardown, 20000);
  function teardown() {
    if (settled) {
      return;
    }
    settled = true;
    clearInterval(ping);
    clearTimeout(timeout);
    window.removeEventListener("message", onMessage);
  }
  window.addEventListener("message", onMessage);
  return teardown;
}

// Per-pane state: the current model bytes (for the "open in a new tab" button),
// a teardown for the in-flight inline handshake, and how far the frame's Netron
// has loaded. The embedded Netron stays live across conversions — it accepts a
// fresh model on each new handshake — so the frame is loaded once and then just
// re-driven, rather than reloaded per model.
const panes = {
  before: { buffer: null, name: "", teardown: null, loading: false, loaded: false },
  after: { buffer: null, name: "", teardown: null, loading: false, loaded: false },
};

// Point one pane ("before" | "after") at a model, embedding it inline and
// arming the "open in a new tab" button. `buffer` is an ArrayBuffer owned by
// this pane (a fresh copy), safe to post repeatedly.
function renderPane(which, buffer, name) {
  const frame = document.getElementById(`netron-${which}-frame`);
  const open = document.getElementById(`netron-${which}-open`);
  const note = document.getElementById(`netron-${which}-note`);
  if (!frame) {
    return;
  }

  const pane = panes[which];
  pane.buffer = buffer;
  pane.name = name;
  if (pane.teardown) {
    pane.teardown();
    pane.teardown = null;
  }
  if (note) {
    note.textContent = "";
  }
  frame.style.display = "";

  // Hand the pane's current model to the Netron already loaded in the frame.
  const drive = () => {
    pane.teardown = driveNetron(frame.contentWindow, pane.buffer, pane.name, (status) => {
      pane.teardown = null;
      if (status.status === "error" && note) {
        note.textContent = `Netron could not open the model: ${status.message || "unknown error"}`;
      }
    });
  };

  if (pane.loaded) {
    drive();
  } else if (!pane.loading) {
    // First model for this pane: load Netron once, then drive whatever the
    // latest model is (a newer one may have arrived while it was loading).
    pane.loading = true;
    frame.onload = () => {
      pane.loaded = true;
      drive();
    };
    frame.src = NETRON_EMBED_URL;
  }
  // If still loading (not yet loaded), the onload handler above will drive the
  // latest pane.buffer once Netron is ready.

  if (open) {
    open.style.display = "";
    open.onclick = () => openInNewTab(which);
  }
}

// Open the pane's current model in a new Netron tab. Must run from a user
// gesture (the button click) so the popup is not blocked.
function openInNewTab(which) {
  const pane = panes[which];
  if (!pane.buffer) {
    return;
  }
  // Note: no `noopener` — we need the returned handle to post the model to.
  const win = window.open(NETRON_EMBED_URL, "_blank");
  if (!win) {
    const note = document.getElementById(`netron-${which}-note`);
    if (note) {
      note.textContent = "Pop-up blocked — allow pop-ups to open Netron in a new tab.";
    }
    return;
  }
  driveNetron(win, pane.buffer, pane.name);
}

function initNetronPanel() {
  const fileInput = document.getElementById("file-input");
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) {
        return;
      }
      fileToArrayBuffer(file)
        .then((buffer) => renderPane("before", toArrayBuffer(buffer), file.name))
        .catch((err) => console.error("netron (before):", err));
    });
  }

  // Called by the converter worker glue in index.html when a result is ready.
  window.netronShowAfter = (dataUrl, name) => {
    try {
      renderPane("after", dataUrlToArrayBuffer(dataUrl), name);
    } catch (err) {
      console.error("netron (after):", err);
    }
  };
}

initNetronPanel();
