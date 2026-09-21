# M5StickV camera demo

Live camera -> a `.kmodel` running on the K210's KPU -> the M5StickV's LCD.
The same idea as `../runtime` (flash the firmware once at `0x0`, swap models by
flashing a `.kmodel` at `0x00C00000`), but it drives the camera and screen, so
it runs *your* model on *live frames* instead of a zeroed input.

**M5StickV only, and only units with the OV7740 camera** (M5Stack shipped some
with a GC0328; the demo says so and stops if the sensor isn't an OV7740).
kendryte-standalone-sdk has no driver for this board's camera, LCD or power
chip, so `src/m5stickv.c` supplies them, taken from Sipeed's MaixPy-v1 -- see
`../../THIRD_PARTY_NOTICES.md`.

## Build

```sh
./build.sh [WORKDIR]      # -> WORKDIR/onnx-k210-camera.bin
```

Same toolchain/SDK/nncase 1.9.0 setup as `../runtime/build.sh` (this wraps it).

## Try it

`prebuilt/` has the firmware (`onnx-k210-camera.bin`, built by `build.sh`) and
the demo model (`edge.kmodel`, with its `edge.onnx`), so you can skip the
build and the model step: flash the `.bin` at `0x0`, then `edge.kmodel` at
`0x00C00000`. To rebuild the model yourself:

```sh
# 1. a model: a fixed-weight Sobel edge detector (3x3 conv, ReLU, 1x1 conv)
python3.10 -m venv .venv && .venv/bin/pip install nncase==1.9.0.20230322 numpy onnx
.venv/bin/python ../../scripts/make_edge_demo_model.py out/      # -> out/edge.kmodel

# 2. flash the firmware at 0x0, then the model at 0x00C00000 (kfpkg with
#    "address": "0x00C00000" as a string and "sha256Prefix": false -- see
#    ../runtime/README.md)
kflash -B goE -b 115200 -p /dev/ttyUSB0 onnx-k210-camera.bin
```

Buttons: **A** cycles the view (camera / model output / split -- left half
camera, right half the model's output), **B** cycles the output gain.

The model must take one float32 `1x3xHxW` input in `[0,1]` (what
`onnx_to_kmodel.py` produces by default); the 240x135 camera frame is resized
to `HxW`. A single-channel float32 spatial output (`1x1xhxw`) is shown as a
greyscale picture; a short float32 vector (e.g. classification logits) is
printed as top-5 on the UART with a confidence bar on the LCD.

## Checking it without looking at the screen

UART (115200) prints fps and timings once a second. `scripts/grab.py` asks the
running demo for one frame over UART and saves it as PNGs -- the camera image,
the DVP's RGB "AI" planes the model is fed, and the model output -- and, given
`--onnx`, compares the device's output with onnxruntime run on the device's
*exact* input:

```sh
python3 scripts/grab.py --out grab/ --onnx out/edge.onnx
```

Opening the serial port resets the M5StickV (DTR/RTS), so the script waits
for the demo to come back up before asking.

## Status -- what's actually been verified

Run on a real M5StickV (OV7740), 2026-09-22, judged by eye on the LCD plus the
UART console:

- AXP192 init, OV7740 detection, LCD, live camera picture (colours and
  orientation confirmed by the person holding the board), and the Sobel edge
  model running on live frames (~15 ms per inference, 13 fps end to end) with
  its edge map shown beside the camera. Also a 32x32 classifier at 2.1 ms.
- **Not verified:** the edge model's numeric output against onnxruntime.
  `scripts/grab.py` (frame dump over UART, with an `--onnx` comparison) is
  written but was never got working: the M5StickV's USB link kept dropping on
  the test machine whenever the demo sent bulk UART data, so treat the script
  as untested.
- Flashing was done with `kflash` (`-B goE -b 115200`). The in-browser
  flasher (`../../web/`) has **not** been tried with this board; see
  `../../README.md` finding #8 for its open `FLASH_WRITE` problem.

Things that cost time (kept so they don't again):

- **The camera is on hardware I2C2, not the DVP's SCCB block.** MaixPy's
  default sensor config for this board has `i2c_num = 2`; talking to the
  sensor through `dvp_sccb_*` never got an ACK from any address. Pins: SCL
  41, SDA 40, 7-bit address `0x21`; read = write the register, stop, then a
  separate read.
- **The LCD pins are 18-22** (RST 21, DCX 20, CS 22, SCLK 19, MOSI 18) on
  SPI0 chip-select 3 -- MaixPy's generic config template lists 36-39, which
  is a different board.
- **UART RX needs `fpioa_set_function(4, FUNC_UARTHS_RX)`**; TX works from
  the boot defaults, RX doesn't. Also mapping IO5 to `UARTHS_TX` *silenced
  the console* -- leave TX alone.
- **`sysctl_set_spi0_dvp_data(1)` is required.** It connects the shared
  SPI0/DVP data bus to the pins. Without it the frame interrupts still fire
  (~26 fps) but every pixel is zero: a black picture and a model that never sees
  anything (its top-5 came out identical every frame -- a good tell).
- **The sensor emits YUV422, and only the DVP's AI output is RGB.** The
  "display" buffer stays YUV: shown as RGB565 it is a grey, green-tinted
  picture (first tried, with an R/B swap, before checking channel means of the
  two buffers). The LCD frame is therefore built from the AI planes.
- **USB is fragile on this board:** the FT232 dropped off the bus several times
  (`-110`/`-32` in dmesg), including mid-flash -- a partial flash then boots to
  "boot failed with exit code 233" on the boot ROM; just reflash. Use a good
  cable and a port directly on the machine. The AXP192 auto-powers the board
  whenever USB is present, and holding the power key past its 4 s shutdown
  time (set in `m5_power_init`) turns it straight back on.
- If the picture is upside-down or mirrored, try another `M5_LCD_MADCTL`
  (`0x20`, `0xA0`, `0xE0`; default `0x60`).
- The demo draws more current than the plain runtime (backlight, camera, LCD);
  a marginal USB cable/port shows up as FT232 `-110` timeouts in `dmesg`.
