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

import { buildNetronUrl, canEmbedInline } from "./netron.mjs";

// Read a File as a `data:...;base64,...` URL — exactly the shape Netron wants.
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("read failed"));
    reader.readAsDataURL(file);
  });
}

// Point one pane ("before" | "after") at a model. Always fills in the
// open-in-new-tab link; embeds the inline frame only when the model is small
// enough to navigate to.
function renderPane(which, dataUrl, name) {
  const link = document.getElementById(`netron-${which}-link`);
  const frame = document.getElementById(`netron-${which}-frame`);
  const note = document.getElementById(`netron-${which}-note`);
  if (!link || !frame || !note) return;

  const url = buildNetronUrl(dataUrl, name);
  link.href = url;
  link.textContent = `open ${name} in a new tab ↗`;
  link.style.display = "";

  if (canEmbedInline(dataUrl)) {
    frame.src = url;
    frame.style.display = "";
    note.textContent = "";
  } else {
    frame.removeAttribute("src");
    frame.style.display = "none";
    note.textContent =
      "Model is too large to embed inline — use the link above to open it in Netron.";
  }
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
