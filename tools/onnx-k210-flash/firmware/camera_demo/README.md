# K210 camera demo (M5StickV, Maix Cube)

Live camera -> a `.kmodel` running on the K210's KPU -> the board's LCD.
The same idea as `../runtime` (flash the firmware once at `0x0`, swap models by
flashing a `.kmodel` at `0x00C00000`), but it drives the camera and screen, so
it runs *your* model on *live frames* instead of a zeroed input.

**M5StickV works end to end. Maix Cube does not yet -- its screen shows the
wrong colour** (see "Cube status" below); everything else on Cube (power,
camera, buttons) is confirmed working on real hardware. Both boards need an
OV7740 camera (M5Stack shipped some M5StickV units with a GC0328 instead; the
demo detects the sensor and stops if it isn't an OV7740).
kendryte-standalone-sdk has no driver for either board's camera, LCD or power
chip, so `src/board_m5stickv.c` / `src/board_cube.c` supply them (the OV7740
camera driver, `src/camera_ov7740.c`, is shared), taken from Sipeed's
MaixPy-v1 and Maixduino -- see `../../THIRD_PARTY_NOTICES.md`.

## Build

```sh
BOARD=m5stickv ./build.sh [WORKDIR]   # default; -> WORKDIR/onnx-k210-camera-m5stickv.bin
BOARD=cube     ./build.sh [WORKDIR]   #        -> WORKDIR/onnx-k210-camera-cube.bin
```

Same toolchain/SDK/nncase 1.9.0 setup as `../runtime/build.sh` (this wraps it).

## Try it

`prebuilt/` has the M5StickV firmware (`onnx-k210-camera-m5stickv.bin`, built
by `BOARD=m5stickv ./build.sh`) and the demo model (`edge.kmodel`, with its
`edge.onnx`), so you can skip the build and the model step on that board:
flash the `.bin` at `0x0`, then `edge.kmodel` at `0x00C00000`. There is no
Cube prebuilt yet -- see "Cube status" below. To rebuild the model yourself:

```sh
# 1. a model: a fixed-weight Sobel edge detector (3x3 conv, ReLU, 1x1 conv)
python3.10 -m venv .venv && .venv/bin/pip install nncase==1.9.0.20230322 numpy onnx
.venv/bin/python ../../scripts/make_edge_demo_model.py out/      # -> out/edge.kmodel

# 2. flash the firmware at 0x0, then the model at 0x00C00000 (kfpkg with
#    "address": "0x00C00000" as a string and "sha256Prefix": false -- see
#    ../runtime/README.md)
kflash -B goE -b 115200 -p /dev/ttyUSB0 onnx-k210-camera-m5stickv.bin
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

## Cube status -- screen not yet correct

Run on a real Maix Cube (OV7740), 2026-09-22, judged by eye on the LCD plus the
UART console:

**Working, confirmed on hardware:** AXP173 power init, OV7740 camera detected
and capturing, KEY1/KEY2 buttons, and the LCD backlight (a plain K210 GPIO,
separate from the panel data path) toggles visibly.

**Not working: the LCD picture is the wrong colour.** The board's own
schematic (`Maix-Cube-2757(Schematic).pdf`, downloaded from dl.sipeed.com)
shows the panel wired as a genuine 8-bit parallel ("8080"-style) bus, not
simple serial SPI -- confirmed independently by the M1n module's own
datasheet, which lists the same 8 data lines. Earlier attempts driving it
with a single SPI wire (tried on several different candidate pins) never
produced anything but noise, consistent with that: 1 of 8 required data bits
is never enough. Switching to genuine octal SPI framing (`SPI_FF_OCTAL`,
matching Maixduino's own `cores/arduino/st7789.c` command sequence exactly)
was the fix for *that* -- a full-screen solid fill now reaches the panel and
displays evenly. What's still wrong is the colour itself: a solid-red fill
came out pink/purple, and a later attempt at solid green wasn't visibly
different from earlier broken output. That's a channel/byte-order bug in the
pixel path, not a structural one, but it isn't tracked down yet -- see
`src/board_cube.c`'s `board_lcd_show()` for the leading suspect (the 2-pixels-
per-32-bit-word packing MaixPy-v1's `mcu_lcd_draw_picture()` uses, which this
code attempts to replicate but hasn't been re-verified against real
hardware after the switch to octal framing).

**The genuinely new finding, worth keeping regardless of how the colour bug
resolves:** the panel's 8 data lines (labelled `SPI0_D0`..`SPI0_D7` in every
Sipeed source that mentions them at all -- this schematic, the module
datasheet) are never given a raw K210 IO pin number anywhere in Sipeed's
public documentation, and neither MaixPy-v1's `py_lcd.c` nor Maixduino's
`st7789.c`/`pins_arduino.h` ever call `fpioa_set_function()` for them, on
this board or Sipeed's other octal-LCD board (TWatch). Checked and ruled out
as the explanation: a leftover factory `config.json` recoverable from flash
(scanned 1.4MB-15MB of this board's own flash for any recoverable text,
found nothing), and Sipeed's GitHub release binaries (no Cube-specific
firmware variant is published at all, unlike M5StickV/Amigo/TWatch). The
working theory, confirmed useful on hardware: these are not ordinary
FPIOA-routable pins, but fixed hardware pads whose routing is switched by the
single `sysctl_set_spi0_dvp_data()` bit -- shared with the DVP camera's own
8-bit data bus. `board_lcd_init()` calls it with `0`; the camera path needs
`1`. Neither value was confirmed to give the *right* colour, but both give a
full, evenly-filled screen rather than noise, which is what actually settled
that no per-pin FPIOA mapping is needed here at all.

Two other things worth recording:

- **`SPI0_MOSI` (pin 28) is a real, board-labelled pin** (the schematic, the
  M1n module datasheet, and Maixduino's `pins_arduino.h` all agree on it) --
  but it's for the SD card / Grove SPI header, not the LCD. Trying it as an
  LCD data line (an earlier, since-abandoned attempt) was based on it being
  the only "SPI0_MOSI"-named pin around, and didn't work, consistent with
  the above.
- **AXP173 power write is real and needed** -- confirmed by Sipeed's own
  wiki text: "MaixCube needs to configure the power chip before using the
  LCD, otherwise the screen will be blurred." The exact three register
  values in `board_power_init()` are copied from MaixPy-v1's Amigo board
  (same PMU chip, different panel/rails) and are not yet confirmed correct
  for the Cube specifically.

## M5StickV status -- verified, working

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
  time (set in `board_power_init`) turns it straight back on.
- If the picture is upside-down or mirrored, try another `BOARD_LCD_MADCTL`
  (`0x20`, `0xA0`, `0xE0`; default `0x60`).
- The demo draws more current than the plain runtime (backlight, camera, LCD);
  a marginal USB cable/port shows up as FT232 `-110` timeouts in `dmesg`.
