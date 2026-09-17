import { compileToKmodel } from "./ncc_wasm.mjs";

const onnxInput = document.getElementById("ncc-onnx-input");
const targetSelect = document.getElementById("ncc-target");
const compileButton = document.getElementById("ncc-compile-button");
const statusEl = document.getElementById("ncc-status");
const logEl = document.getElementById("ncc-log");

const firmwareInput = document.getElementById("firmware-input");
const flashPresetSelect = document.getElementById("flash-preset");
const flashAddressInput = document.getElementById("flash-address");

compileButton.addEventListener("click", async () => {
  const file = onnxInput.files && onnxInput.files[0];
  if (!file) {
    statusEl.textContent = "pick an ONNX file first";
    return;
  }

  compileButton.disabled = true;
  logEl.textContent = "";
  statusEl.textContent = "compiling...";
  try {
    const onnxBytes = new Uint8Array(await file.arrayBuffer());
    const target = targetSelect.value;
    const kmodelBytes = await compileToKmodel(onnxBytes, {
      target,
      onLog: (line) => {
        logEl.textContent += line + "\n";
        logEl.scrollTop = logEl.scrollHeight;
      },
    });

    const kmodelName = file.name.replace(/\.onnx$/i, "") + ".kmodel";
    const kmodelFile = new File([kmodelBytes], kmodelName, { type: "application/octet-stream" });

    // Hand the result straight to the flasher below: a real FileList can't
    // be constructed directly, but DataTransfer's is assignable to a file
    // input's .files (standard technique, works in this page's target
    // browsers -- Chrome/Edge, same as Web Serial itself requires).
    const dt = new DataTransfer();
    dt.items.add(kmodelFile);
    firmwareInput.files = dt.files;
    firmwareInput.dispatchEvent(new Event("change"));

    if (flashPresetSelect) {
      flashPresetSelect.value = "0x00C00000";
      flashPresetSelect.dispatchEvent(new Event("change"));
    } else if (flashAddressInput) {
      flashAddressInput.value = "0x00C00000";
    }

    statusEl.textContent = `done: ${kmodelBytes.length.toLocaleString()} bytes -- loaded into the flasher below (model, 0x00C00000)`;
  } catch (e) {
    statusEl.textContent = "compile failed: " + (e && e.message ? e.message : e);
  } finally {
    compileButton.disabled = false;
  }
});
