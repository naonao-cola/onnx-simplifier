# onnx-cardputer-flash

Browser demo: take an ONNX model straight from Hugging Face, simplify/quantize
it with onnxsim, convert it for TFLite Micro, and flash the resulting
firmware onto an ESP32-S3 board (M5Stack Cardputer) — no native app, no
server, no driver install, using
[Web Serial](https://developer.chrome.com/docs/capabilities/serial) instead
of a USB-DFU/WebUSB path (see this tool's design discussion for why Web
Serial was picked over WebUSB for this class of board).

## Why Cardputer, and why Web Serial

- **Web Serial, not WebUSB**: ESP32's ROM bootloader speaks a UART-over-USB
  ISP protocol (through the board's USB-serial chip), not raw USB DFU — that
  is exactly what Web Serial is for, and it is a *solved* problem thanks to
  Espressif's own [`esptool-js`](https://github.com/espressif/esptool-js),
  which this tool's flashing panel uses directly. A K210 board (M5StickV,
  Sipeed Maix Amigo) is the same story in principle but has no existing
  Web Serial port of its ISP protocol — a real follow-up, not attempted here.
- **Cardputer**: an ESP32-S3 board, so it gets `esptool-js`'s flashing story
  for free. It has no camera, so the model class this recipe targets is
  audio keyword-spotting (it has a built-in mic) rather than vision — see
  `firmware/README.md`.

## Pipeline and what's verified

| Step | Where | Status |
|---|---|---|
| Fetch + simplify/quantize an HF model | the existing [onnxsim model converter](../convertmodel/index.html) — this tool doesn't repeat that UI | already shipped, unrelated to this addition |
| ONNX → TFLite (int8) | `scripts/onnx_to_tflite_micro.py` (`convert_to_tflite`, wraps `onnx2tf`) | reviewed, not run in CI (needs onnx2tf + TensorFlow, not installed anywhere else in this repo) |
| TFLite → C header | `scripts/onnx_to_tflite_micro.py` (`emit_c_header`) | unit-tested, see `tests/` |
| Firmware build (TFLite Micro + model) | `firmware/README.md` (recipe, not a checked-in project) | reviewed against upstream docs, not built/run — no hardware here |
| Flash over Web Serial | `web/flasher.mjs` (Espressif's `esptool-js`) | loads and runs its UI logic cleanly in a browser (checked headless); **not** exercised against a real board — no ESP32-S3 attached to this environment |

Nothing here claims to be hardware-verified end to end — that step needs a
real Cardputer and a person with the browser permission prompt in front of
them. Try it and report back if any of the `esptool-js` call shapes are
stale.

## Using it

1. Simplify/quantize your model with the
   [onnxsim converter](../convertmodel/index.html), download the result.
2. `pip install onnx2tf` (pulls in TensorFlow), then:
   ```sh
   python3 scripts/onnx_to_tflite_micro.py your_model.onnx model_data.h
   ```
3. Follow `firmware/README.md` to bake `model_data.h` into a TFLite Micro
   sketch and produce one merged `.bin`.
4. Serve `web/` locally (`python3 -m http.server` from `web/`) and open it
   in Chrome/Edge — Connect, pick the `.bin`, Flash.

## Not done yet / follow-ups

- A Sipeed Maix Amigo / M5StickV (Kendryte K210) target: same board family,
  a real camera (Amigo) and better on-device inference (KPU). The Web
  Serial flasher this needed didn't exist anywhere, so it's now built —
  see [`../onnx-k210-flash/`](../onnx-k210-flash/README.md) (a from-scratch
  port of `kflash.py`'s ISP protocol). Still needs an onnx→kmodel
  conversion script (nncase, not TFLite Micro) and a firmware recipe,
  mirroring this tool's `scripts/`/`firmware/` — not done yet.
- No CI coverage — nothing here can run without either a real board or a
  TensorFlow install this repo doesn't otherwise carry.
- Vision models are out of scope for Cardputer specifically (no camera); an
  ESP32-S3 board with a camera module would reuse everything except
  `firmware/README.md`'s mic-specific wiring.
