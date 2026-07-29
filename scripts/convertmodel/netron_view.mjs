// Browser glue for the "Visualize with Netron" panel on the converter page.
//
// Shows the model graph before and after simplify/optimize side by side, each
// rendered by Netron (https://netron.app) in an <iframe>. The model is passed
// as an in-browser data URL (see netron.mjs), so nothing leaves the page.
//
// The "before" pane is driven here by watching the file input directly. The
// "after" pane is driven by the converter worker code, which calls the
// `window.netronShowAfter(dataUrl, name)` hook this module installs once a
// conversion finishes.

import { buildNetronUrl, netronUrlFits } from "./netron.mjs";

// Read a File as a `data:...;base64,...` URL — exactly the shape Netron wants.
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("read failed"));
    reader.readAsDataURL(file);
  });
}

// Turn a `data:` URL back into a Blob so we can offer a short-lived object-URL
// download whose length is independent of the model size (unlike the data URL
// itself, which the browser refuses to navigate to past ~2 MB).
async function dataUrlToBlob(dataUrl) {
  return (await fetch(dataUrl)).blob();
}

// Point one pane ("before" | "after") at a model. When the model is small
// enough that Netron's data-URL fits under the browser's URL cap, both the
// inline <iframe> and the "open in a new tab" link work. Otherwise the browser
// would block the over-long navigation (`about:blank#blocked`), so we hide the
// frame and offer a local download to drag into netron.app — no upload, any
// size.
function renderPane(which, dataUrl, name) {
  const link = document.getElementById(`netron-${which}-link`);
  const frame = document.getElementById(`netron-${which}-frame`);
  const note = document.getElementById(`netron-${which}-note`);
  if (!link || !frame || !note) return;

  if (netronUrlFits(dataUrl, name)) {
    const url = buildNetronUrl(dataUrl, name);
    link.href = url;
    link.removeAttribute("download");
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `open ${name} in a new tab ↗`;
    link.style.display = "";
    frame.src = url;
    frame.style.display = "";
    note.textContent = "";
    return;
  }

  // Too large for a Netron URL: fall back to a download the user drops into
  // netron.app themselves.
  frame.removeAttribute("src");
  frame.style.display = "none";
  link.style.display = "none";
  dataUrlToBlob(dataUrl)
    .then((blob) => {
      link.href = URL.createObjectURL(blob);
      link.download = name;
      link.removeAttribute("target");
      link.removeAttribute("rel");
      link.textContent = `download ${name} ↓`;
      link.style.display = "";
    })
    .catch((err) => console.error(`netron (${which}) download:`, err));
  note.innerHTML =
    "Too large to open in Netron from a link — browsers block URLs over " +
    "~2 MB. Download the model above, then drop it into " +
    '<a href="https://netron.app" target="_blank" rel="noopener">netron.app</a>.';
}

function initNetronPanel() {
  const fileInput = document.getElementById("file-input");
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      fileToDataURL(file)
        .then((dataUrl) => renderPane("before", dataUrl, file.name))
        .catch((err) => console.error("netron (before):", err));
    });
  }

  // Called by the converter worker glue in index.html when a result is ready.
  window.netronShowAfter = (dataUrl, name) => {
    try {
      renderPane("after", dataUrl, name);
    } catch (err) {
      console.error("netron (after):", err);
    }
  };
}

initNetronPanel();
