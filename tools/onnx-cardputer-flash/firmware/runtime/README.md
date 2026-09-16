# Cardputer generic TFLite Micro runtime (prebuilt)

`prebuilt/cardputer-runtime.bin` is a **compiled, merged, flashable image**
— flash it once at address `0x0` and it never needs rebuilding again.
Swapping models afterward is a second Web Serial write of a `.tflite`
file's bytes at a fixed address (`0x310000`), using the same flasher UI,
no PlatformIO/compiler involved.

## Status

**Compiled and linked for real** (this isn't a recipe someone still has to
get working): built with PlatformIO against `esp32-s3-devkitc-1` +
`framework = arduino`, linking `m5stack/M5Cardputer@1.1.1` and
`spaziochirale/Chirale_TensorFLowLite@2.0.0`. Actual output:

```
RAM:   40.2% (131672 / 327680 bytes)
Flash: 24.2% (760961 / 3145728 bytes, the "factory" app partition)
```

**Not run on real hardware.** No Cardputer is attached to the environment
this was built in — the ELF links, sizes fit, and the merge step produces
a well-formed image, but nobody has powered on a board with it yet. If you
try it, the most likely failure points are: the DTR/RTS flashing sequence
(`onnx-k210-flash` had a real bug here that only showed up against source;
this ESP32 side reuses `esptool-js`, which is much better-trodden ground),
the `tensor_arena` size (100KB default — `AllocateTensors()` fails loudly
and tells you to raise it if a model needs more), and the mmap flash-mode
match (`qio`, the `esp32-s3-devkitc-1` board profile's actual default —
verified from that board's own PlatformIO manifest, not assumed).

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
    --flash_mode qio --flash_freq 80m --flash_size 8MB \
    0x0     .pio/build/cardputer/bootloader.bin \
    0x8000  .pio/build/cardputer/partitions.bin \
    0x10000 .pio/build/cardputer/firmware.bin
```

`sha256sum` of the committed `prebuilt/cardputer-runtime.bin`:
`fd8c4cd46b850fcf63e0a6fe630230a0b02233f8ddac15be5e25a6eaa2cc6585`
