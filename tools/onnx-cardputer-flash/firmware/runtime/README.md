# Cardputer generic TFLite Micro runtime (prebuilt)

`prebuilt/cardputer-runtime.bin` is a **compiled, merged, flashable image**
— flash it once at address `0x0` and it never needs rebuilding again.
Swapping models afterward is a second Web Serial write of a `.tflite`
file's bytes at a fixed address (`0x310000`), using the same flasher UI,
no PlatformIO/compiler involved.

## Status

**Compiled, flashed, and booted on a real M5Stack Cardputer.** Built with
PlatformIO against `esp32-s3-devkitc-1` + `framework = arduino`, linking
`m5stack/M5Cardputer@1.1.1` and `spaziochirale/Chirale_TensorFLowLite@2.0.0`.
Actual output:

```
RAM:   49.9% (163640 / 327680 bytes)
Flash: 25.0% (784865 / 3145728 bytes, the "factory" app partition)
```

**Real-hardware finding #1 (fixed): the `esp32-s3-devkitc-1` board profile's
default flash mode (`qio`) doesn't work on the Cardputer's actual flash
chip.** Flashed as `qio`, the ROM bootloader loops forever
(`ets_loader.c 78` → `TG0WDT_SYS_RST` → repeat) and never reaches app code
-- it never gets past the ROM's own SPI flash read. Flashed as `dio`
instead, the ROM loads the second-stage bootloader and app correctly. Fixed
by `board_build.flash_mode = dio` in `platformio.ini`; `prebuilt/` and the
rebuild recipe below are updated to match.

**Real-hardware finding #2 (real bug, unfixed here): a legacy-quantized
model's `DepthwiseConv` filter crashes `AllocateTensors()`.** Flashing
Google's official `micro_speech` TinyConv reference model (from the
`tensorflow-Micro-Speech-TinyConv-SpeechCommands-uint8-onnx` HF repo's own
`source/model.tflite`, as a shortcut to get *some* real `.tflite` onto the
board fast -- *not* run through this tool's own conversion) at `0x310000`
causes a `Guru Meditation Error (StoreProhibited)` crash loop. Symbolized
with `xtensa-esp32s3-elf-addr2line` against a matching build: the crash is
`tflite::PopulateConvolutionQuantizationParams`
(`Chirale_TensorFLowLite/src/tensorflow/lite/kernels/kernel_util.cpp:212`),
called from `CalculateOpDataDepthwiseConv` while preparing the DepthwiseConv
op inside `AllocateTensors()`. That function's per-axis-only overload does
`reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)`
and immediately dereferences `->scale->size` with **no null check** (the
other overload right below it does `TF_LITE_ENSURE(context,
affine_quantization)` first) -- it assumes every DepthwiseConv filter has
per-channel affine quantization metadata attached. That old reference model
predates per-channel affine quantization (legacy scalar scale/zero_point
only, no `TfLiteAffineQuantization` struct), so `filter->quantization.params`
is null and the dereference crashes. This is a real gap in the vendored
`Chirale_TensorFLowLite` library, not something to patch in this repo (it's
a separate PlatformIO dependency, not vendored here).

**Real-hardware finding #3 (real bug, this tool's own script): `onnx2tf
-oiqt` rejects an already-quantized ONNX input.** Running this tool's own
`scripts/onnx_to_tflite_micro.py` (`convert_to_tflite()`, which always
passes `-oiqt` by default) against that same HF repo's `model.onnx` fails:
`-oiqt` is onnx2tf's own float->int8 calibration flow, which requires a
Float32 graph input; this particular ONNX model is *already*
uint8-quantized (QuantizeLinear/DequantizeLinear baked in from its original
export), so its input is `uint8`, not `float32`, and onnx2tf errors out
before producing anything. `--no-int8` (skip `-oiqt`, keep onnx2tf's plain
Float32 output) works around it -- see finding #4 below. This
script doesn't currently detect or handle already-quantized ONNX inputs;
treat `-oiqt` as verified only for models that are still Float32 going in.

**Real-hardware finding #4 (real bug, unfixed here, but precisely
root-caused): `Transpose` doesn't support `uint8` in this TFLM version.**
The Float32 `.tflite` produced by the `--no-int8` workaround above (71232
bytes, from the same HF model) flashed at `0x310000`, booted with no crash,
and cleared `AllocateTensors()` for real on hardware -- real proof the
flash-partition-mmap -> `TFL3` detection -> `tflite::GetModel()` ->
`AllOpsResolver` -> `MicroInterpreter::AllocateTensors()` chain works end to
end (float32 kernels don't call `PopulateConvolutionQuantizationParams`, so
this doesn't retest finding #2). But `interpreter->Invoke()` over a zeroed
input (`main.cpp`'s deliberately minimal sanity check) fails every time,
printing this exact pair of lines (see finding #5 for how these became
visible at all):

```
Type UINT8 is currently not supported by Transpose. Only float32 and int8 is supported
Node TRANSPOSE (number 0) failed to invoke with status 1
```

This model's graph (per onnx2tf's own conversion log) is
`Reshape -> Transpose -> ... -> DequantizeLinear -> Conv/MatMul (float) ->
... -> QuantizeLinear -> output` -- the graph's true external input is
`uint8` (this "float32" export keeps the original ONNX QuantizeLinear/
DequantizeLinear nodes as real ops rather than folding them into tensor
metadata; only the *internal* compute is float32), and `Transpose` is the
very first op, running on that still-`uint8` data before `DequantizeLinear`
ever converts it. `Chirale_TensorFLowLite`'s vendored TFLM `Transpose`
kernel (`tensorflow/lite/micro/kernels/transpose.cpp`) only implements
`float32` and `int8` -- not `uint8` -- so it fails cleanly rather than
computing a wrong result. This is a real op/dtype coverage gap in the
vendored library, not something to patch in this repo. A per-channel
int8-quantized model (fixing finding #3) also still hasn't cleared
`AllocateTensors()` on real hardware, so neither quantized nor float
inference has produced a *correct* result end to end yet -- only that the
pipeline can load and attempt to run a real model without crashing, and
that a specific, real cause blocks this specific model's actual inference.

**Real-hardware finding #5 (fixed): `Serial` -- and `Chirale_TensorFLowLite`'s
own internal error reporting -- silently went to a disconnected pin.**
Every serial capture attempt in this session (both remote, via a Python
`pyserial` script driving the board's DTR/RTS directly, and local, via
`screen`/`pio device monitor` on the same machine) came back with nothing
beyond the ROM boot banner -- no app output at all, success or failure,
even though the Cardputer's *screen* clearly showed real `printLine()` text
the whole time. Root cause: `esp32-s3-devkitc-1`'s board profile defines
`ARDUINO_USB_MODE=1` but not `ARDUINO_USB_CDC_ON_BOOT`, and
Arduino-ESP32's `HardwareSerial.cpp`/`HWCDC.cpp` both gate on that second
flag -- without it, `Serial` is `HardwareSerial(0)` (UART0's physical GPIO
pins), and the native USB-Serial/JTAG peripheral (the *only* port actually
wired to the host on this board, and the same one the ROM bootloader and
`esptool` both use) is instead exposed under a *different* name,
`USBSerial`, that nothing in this firmware or `Chirale_TensorFLowLite`
references. `Chirale_TensorFLowLite`'s own `DebugLog()` (and therefore
every `MicroPrintf` diagnostic TFLM itself emits on a kernel failure, like
finding #4's exact error text above) also writes to `Serial`, so this
silently swallowed the *one* piece of diagnostic output that would have
explained finding #4 immediately. Fixed by adding
`build_flags = -DARDUINO_USB_CDC_ON_BOOT=1` to `platformio.ini`, which
makes `Serial` alias the same USB-Serial/JTAG peripheral everything else
already uses -- confirmed working by capturing finding #4's exact error
text over that same port after the fix.

**Real-hardware finding #6: correct, real int8 inference confirmed end to
end.** Findings #2-#4 all trace back to using the *wrong kind* of test
model -- the HF repo's `-uint8-onnx` variant is already ONNX-quantized with
the legacy scheme findings #2-#4 don't handle. The same HF org also
publishes an `-fp32-onnx` variant of the identical architecture
(`ketiswp/tensorflow-Micro-Speech-TinyConv-SpeechCommands-fp32-onnx`) --
genuinely Float32, letting `onnx2tf -oiqt` do real calibration instead of
choking on already-quantized input (finding #3) or producing a raw-`uint8`
`Transpose` (finding #4). `-oiqt`'s *default* calibration path is broken
for any non-image model, and arguably for every model right now: it always
downloads a fixed `20x128x128x3` ImageNet-shaped `.npy` calibration file
and loads it with `np.load()` **without** `allow_pickle=True`
(`onnx2tf/utils/common_functions.py`'s `download_test_image_data()`),
which fails outright on current numpy -- a real bug in onnx2tf itself, not
this repo, and unrelated to model shape. Worked around by supplying real
calibration data via onnx2tf's own `-cind` flag directly (this script
doesn't expose it yet -- see the main `README.md`'s "Not done yet /
follow-ups"):

```sh
python3 -c "
import numpy as np
rng = np.random.default_rng(0)
np.save('calib_input.npy', rng.random((20, 40, 1, 49), dtype=np.float32))
"
onnx2tf -i model_fp32.onnx -o out -oiqt \
    -cind Reshape_2__0 calib_input.npy "[0.0]" "[1.0]"
```

(Note the calibration tensor's input-op name is onnx2tf's *converted* TF
name, `Reshape_2__0`, not the ONNX graph's own `Reshape_2:0` -- passing the
latter fails with a `KeyError` deep in TF's calibrator.) The resulting
`*_full_integer_quant.tflite` (21464 bytes) has real per-tensor int8
quantization on both I/O boundaries (`scale=0.0039, zero_point=-128`) and
produces genuinely different outputs for different inputs on the desktop
interpreter -- unlike finding #2's crash or finding #4's clean failure,
this model flashed at `0x310000`, booted, and **`Invoke()` returned
`kTfLiteOk`, repeatedly, confirmed live over serial** (thanks to finding
#5's fix) with output `type=9` (`kTfLiteInt8`, matching the model's real
output dtype). This is the first fully successful real-hardware, real-model,
correct-quantization-path inference this tool has produced end to end.

**Real-hardware finding #7: the Cardputer's own microphone now feeds real
audio into a matching model.** `src/main.cpp` checks whether the loaded
model's input is exactly `int8`, single-input, 1960 bytes (`40 x 49` --
this model family's shape); if so, it records one second of real audio via
`M5Cardputer.Mic`, runs it through TF's own `microfrontend` library
(already vendored inside `Chirale_TensorFLowLite` -- FFT, mel filterbank,
noise reduction, PCAN gain control, log, configured with the exact
16kHz/30ms-window/20ms-step/40-channel settings this model family's own
`micro_model_settings.h` uses), and quantizes the result with the same
`value_scale=256, value_div=666` conversion as TF's own
`feature_provider.cc` (a re-implementation of that specific,
training-pipeline-derived scaling, not a generic one). Confirmed working
end to end on real hardware: `mic ready -- feeding real audio into the
model below`, then repeated `test inference: OK (46 ms)` with real, varying
output values (e.g. `out[0] dequant: 0.2500,0.2461,0.2461,0.2539` for
ambient background noise -- a genuine, if unremarkable, live prediction,
not a placeholder). Any model whose input doesn't match this exact
shape/type still falls back to the original zeroed-input sanity check --
this is a narrow, model-family-specific special case, not a general "runs
any model's real preprocessing" claim (see the file's own header comment).
Inference latency (`millis()` around `Invoke()`) and every output tensor's
actual values (raw quantized + dequantized) are now printed on every cycle,
not just once at boot.

The `tensor_arena` size (100KB default) is untested against a model that
actually needs meaningfully more than this one did -- `AllocateTensors()`
fails loudly and tells you to raise it if a model needs more, but no model
has gotten that far yet.

## How it works

`src/main.cpp`:

1. `esp_partition_find_first()` + `esp_partition_mmap()` map the "model"
   partition (`partitions.csv`, offset `0x310000`, 1MB) directly into the
   CPU's address space — **not** copied into RAM. The Cardputer's ESP32-S3
   has ~320KB of SRAM total; several candidate models (see the main
   `README.md`'s table) are themselves hundreds of KB, so a copy wouldn't
   fit regardless of arena size.
2. Checks the mapped bytes for the TFLite flatbuffer identifier (`TFL3` at
   byte offset 4) before touching them — right after flashing this
   runtime, the partition is erased flash (`0xFF` bytes), and the board
   should say "no model flashed yet" rather than crash on garbage.
3. `tflite::GetModel()` + `tflite::AllOpsResolver` + `tflite::MicroInterpreter`
   load and initialize the model, then `AllocateTensors()`.
4. If the model's input matches the 40-channel/49-frame int8 audio-frontend
   shape (see finding #7), records real microphone audio and extracts
   features into it every cycle; otherwise zeroes it (proves `Invoke()`
   runs, says nothing about accuracy). Either way, every cycle prints
   `Invoke()`'s status, latency, and the actual output values -- not just
   "OK" and a shape dump. Repeats every 2 seconds in `loop()`, both so a
   mic-driven model gets a fresh real prediction each cycle and so a host
   reconnecting after this board's native-USB reset-time disconnect isn't
   limited to one narrow window right after boot.

## Partition layout (`partitions.csv`)

| Partition | Offset | Size | Purpose |
|---|---|---|---|
| `nvs` | `0x9000` | `0x5000` | Arduino/ESP-IDF default |
| `phy_init` | `0xe000` | `0x1000` | Arduino/ESP-IDF default |
| `factory` (app) | `0x10000` | `0x300000` (3MB) | this runtime's compiled code |
| `model` (data) | `0x310000` | `0x100000` (1MB) | swappable — flash a `.tflite` here |

A model must fit in 1MB (every candidate in the main README's table is
≤530KB) and its activation memory must fit the 100KB `tensor_arena`
(unverified per-model — `AllocateTensors()` will say so if it doesn't).

## Using it

1. Flash `prebuilt/cardputer-runtime.bin` at address `0x0` with
   `onnx-cardputer-flash`'s Web Serial flasher (`../../web/index.html`) —
   exactly like flashing any other merged image.
2. Convert a model (`../../scripts/onnx_to_tflite_micro.py`) and flash
   *just* the resulting `.tflite` file's bytes at address `0x310000`, same
   flasher, same page. No rebuild.
3. Open a serial monitor (115200 baud) or look at the Cardputer's screen —
   it prints what it loaded and, every 2 seconds, a fresh inference's
   status, latency, and output values. For a model matching the 40x49 int8
   audio-frontend shape (finding #7), that's a real prediction from the
   built-in mic; otherwise it's a zeroed-input sanity check.

## Rebuilding it yourself

```sh
pip install platformio   # if you don't have it
cd tools/onnx-cardputer-flash/firmware/runtime
pio run
python3 -m esptool --chip esp32s3 merge_bin -o cardputer-runtime.bin \
    --flash_mode dio --flash_freq 80m --flash_size 8MB \
    0x0     .pio/build/cardputer/bootloader.bin \
    0x8000  .pio/build/cardputer/partitions.bin \
    0x10000 .pio/build/cardputer/firmware.bin
```

`--flash_mode dio`, not the board profile's default `qio` -- see "Status"
above (real-hardware finding #1). `board_build.flash_mode = dio` in
`platformio.ini` makes `pio run` itself build the right thing; the `merge_bin`
step needs the same flag repeated since it stamps its own image header.

`sha256sum` of the committed `prebuilt/cardputer-runtime.bin`:
`9229e9d52009189088f84ab3479ba715812d503d67a8e33a877531fd4851ac98`
