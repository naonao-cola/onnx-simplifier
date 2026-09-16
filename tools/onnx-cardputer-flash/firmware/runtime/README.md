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
RAM:   40.2% (131672 / 327680 bytes)
Flash: 24.2% (761105 / 3145728 bytes, the "factory" app partition)
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
Float32 output) works around it -- see "Confirmed end-to-end" below. This
script doesn't currently detect or handle already-quantized ONNX inputs;
treat `-oiqt` as verified only for models that are still Float32 going in.

**Real-hardware finding #4 (real bug, unfixed here): the sanity `Invoke()`
call fails even after a clean `AllocateTensors()`.** The Float32 `.tflite`
produced by the `--no-int8` workaround above (71232 bytes, from the same HF
model) flashed at `0x310000`, booted with no crash, and the Cardputer's own
screen showed `model loaded ok` plus its input/output shape dump -- real
proof the flash-partition-mmap -> `TFL3` detection -> `tflite::GetModel()`
-> `AllOpsResolver` -> `MicroInterpreter::AllocateTensors()` chain works
end to end on real hardware (float32 kernels don't call
`PopulateConvolutionQuantizationParams`, so this doesn't retest finding #2).
But the very next step, `interpreter->Invoke()` over a zeroed input
(`main.cpp`'s deliberately minimal sanity check), printed `test inference:
FAILED` -- `Invoke()` returned a non-`kTfLiteOk` status. Not yet
root-caused: candidates are the all-zero input being out of a range some op
in this graph doesn't handle (this model's real input is MFCC audio
features, not raw zeros), or a real kernel bug for one of this graph's ops
in this TFLM version's float32 path. A per-channel int8-quantized model
(fixing finding #3) also still hasn't cleared `AllocateTensors()` on real
hardware, so neither quantized nor float inference has been proven to
produce a *correct* result end to end yet -- only that the pipeline can
load and attempt to run a real model without crashing.

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
   load and initialize the model, `AllocateTensors()`, then run **one**
   inference over a zeroed input and print each tensor's shape/type plus
   the output. This proves the load-from-flash-partition path executes a
   real model end to end; it says nothing about the model's accuracy, and
   deliberately doesn't include any model-specific input preprocessing
   (MFCC framing for a keyword-spotting model, etc.) — that's real,
   per-model work that belongs in a follow-up sketch built against one
   chosen model, not something a "generic" runtime can do generically.

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
   it prints what it loaded and the sanity-inference result.

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
`e22990fe84bd37dd7399f47dec896b28a8f96a5b4ca6999b28703114cbdc9b24`
