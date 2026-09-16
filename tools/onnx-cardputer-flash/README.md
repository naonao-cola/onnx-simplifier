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
| ONNX → TFLite (int8) | `scripts/onnx_to_tflite_micro.py` (`convert_to_tflite`, wraps `onnx2tf -oiqt`) | **run for real against a real HF model — found a real bug**: `-oiqt` requires a Float32 ONNX input, but a model that's already ONNX-quantized (QuantizeLinear/DequantizeLinear baked in, `uint8` input) makes onnx2tf error out before producing anything. `--no-int8` (plain Float32 output, skips `-oiqt`) works around it; the script doesn't yet detect/handle already-quantized inputs itself — see `firmware/runtime/README.md`'s "Status", finding #3 |
| TFLite → C header | `scripts/onnx_to_tflite_micro.py` (`emit_c_header`) | unit-tested, see `tests/` |
| Firmware (generic TFLite Micro runtime) | [`firmware/runtime/`](firmware/runtime/README.md) — a real PlatformIO project, not just a recipe | **compiled, flashed, and booted on a real M5Stack Cardputer.** Real hardware found and fixed a `qio`→`dio` flash-mode bug (finding #1); found (unfixed, upstream-library) a crash on legacy-quantized `DepthwiseConv` models (finding #2); found and precisely root-caused (unfixed, upstream-library) that this model's `Transpose` op runs on `uint8` data, which this TFLM version's `Transpose` kernel doesn't support (finding #4); and found and fixed a `Serial`-routing bug that was hiding *all* app/TFLM diagnostic output, including finding #4's own error text (finding #5) — see that README's "Status" for all five |
| Flash over Web Serial | `web/flasher.mjs` (Espressif's `esptool-js`) | loads and runs its UI logic cleanly in a browser (checked headless); the underlying ISP protocol was exercised for real via `esptool` directly against `/dev/ttyACM0` (same USB-Serial/JTAG port Web Serial would use) — `flasher.mjs`'s own browser-side call shapes are still **not** exercised through an actual Chrome Web Serial session |

Real hardware confirmed: flashing, booting, mmap'ing the model partition,
and (for a Float32 model) `AllocateTensors()` all work end to end, and
`Serial`/on-device diagnostics are now real and readable (finding #5). Not
yet confirmed: a *correct* inference result -- the one real model tried so
far loads but its `Transpose` op fails on `uint8` input, a precise, real
TFLM library gap (finding #4), and no int8-quantized model has cleared
`AllocateTensors()` at all (blocked on findings #2 and #3). The browser UI
itself doing the flashing (vs. `esptool` CLI standing in for it) is also
still unverified. See `firmware/runtime/README.md`'s "Status" for the full,
numbered findings.

## Using it

1. **Once per board:** serve `web/` locally (`python3 -m http.server` from
   `web/`), open it in Chrome/Edge, Connect, pick
   `firmware/runtime/prebuilt/cardputer-runtime.bin`, flash at `0x0`.
2. Simplify/quantize your model with the
   [onnxsim converter](../convertmodel/index.html) (the candidate-model
   table on this tool's own page has one-click links), download the result.
3. `pip install onnx2tf` (pulls in TensorFlow), then:
   ```sh
   python3 scripts/onnx_to_tflite_micro.py your_model.onnx model.tflite
   ```
   (a `.tflite` output path writes the plain converted model; any other
   extension writes a C header instead — the runtime firmware reads a
   plain `.tflite` file from its flash partition, not a C array).
4. Flash *that* `.tflite` file's raw bytes at `0x310000`, same page, same
   Connect session — no rebuild, no PlatformIO, step 1 doesn't repeat.
5. Reset the board (or power-cycle) — it prints what it loaded over serial
   (115200 baud) and on its own screen.

Prefer one self-contained binary per model instead? `firmware/README.md`'s
older recipe bakes the model directly into the firmware as a C array.

## Not done yet / follow-ups

- **A model whose graph doesn't feed `uint8` data into `Transpose`** would
  sidestep finding #4 (root-caused, not fixed -- the gap is in vendored
  `Chirale_TensorFLowLite`, not this repo). Either a model whose converted
  graph doesn't have a `Transpose` immediately on the raw quantized input,
  or a genuine per-channel int8-quantized model (which needs finding #3
  fixed first) would be the next real test of whether `Invoke()` can
  produce a *correct* result on this hardware, not just avoid crashing.
- **Handle already-quantized ONNX inputs in `onnx_to_tflite_micro.py`**
  (finding #3): `convert_to_tflite()` always passes onnx2tf's `-oiqt`,
  which needs a Float32 input and errors out on a model that's already
  ONNX-quantized (QuantizeLinear/DequantizeLinear baked in). Detecting that
  case (e.g. checking the ONNX graph's input dtype, or catching onnx2tf's
  specific error) and skipping `-oiqt` automatically would make the script
  work uniformly across both already-quantized and still-float HF models.
- A Sipeed Maix Amigo / M5StickV (Kendryte K210) target: same board family,
  a real camera (Amigo) and better on-device inference (KPU). The Web
  Serial flasher this needed didn't exist anywhere, so it's now built —
  see [`../onnx-k210-flash/`](../onnx-k210-flash/README.md) (a from-scratch
  port of `kflash.py`'s ISP protocol). Its onnx→kmodel conversion
  (nncase, not TFLite Micro) is done and run for real against a real HF
  model; still needs a firmware recipe (MaixPy or bare-metal K210 SDK) to
  actually load a kmodel on-device, mirroring this tool's `firmware/`.
- No CI coverage — nothing here can run without either a real board or a
  TensorFlow install this repo doesn't otherwise carry.
- Vision models are out of scope for Cardputer specifically (no camera); an
  ESP32-S3 board with a camera module would reuse everything except
  `firmware/README.md`'s mic-specific wiring.
