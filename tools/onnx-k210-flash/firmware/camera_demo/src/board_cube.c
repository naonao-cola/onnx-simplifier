// Sipeed Maix Cube board support: the 240x240 IPS LCD on octal SPI0 (a
// genuine 8-bit parallel "8080"-style bus, confirmed from the board's own
// schematic -- see below), the AXP173 power chip, and the two KEY buttons.
// See board.h and ../README.md's Cube status section: the LCD is not yet
// producing a correct picture, but real progress is recorded here rather
// than left in a scratch file -- see the comments below for exactly what's
// confirmed and what's still open.
//
// Pin numbers and the ST7789 sequence are cross-checked against three
// independent Sipeed sources (never copied blindly from one): MaixPy-v1
// (components/boards/config/cube.config.json, components/drivers/lcd,
// py_lcd.c's DEV_CUBE_IPS_240x240), Maixduino's own board variant
// (variants/sipeed_maix_cube/pins_arduino.h, cores/arduino/st7789.c) --
// which independently agrees on every control pin -- and the board's own
// schematic (Maix-Cube-2757(Schematic).pdf, dl.sipeed.com) plus the M1n
// module's datasheet. See ../../../THIRD_PARTY_NOTICES.md.

#ifdef BOARD_CUBE

#include "board.h"

#include <stddef.h>
#include <stdio.h>

#include "dmac.h"
#include "fpioa.h"
#include "gpiohs.h"
#include "i2c.h"
#include "sleep.h"
#include "spi.h"
#include "sysctl.h"

// -------------------------------------------------------------- power ----

#define PMU_ADDR 0x34 // AXP173 on I2C1 (SCL 30, SDA 31) -- confirmed by Sipeed's own wiki
#define PMU_PIN_SCL 30
#define PMU_PIN_SDA 31

int board_power_init(void) {
  // IO banks 6/7 (IO 36..47: the LCD control pins and the camera) are 1.8V on
  // this board, as in MaixPy.
  sysctl_set_power_mode(SYSCTL_POWER_BANK6, SYSCTL_POWER_V18);
  sysctl_set_power_mode(SYSCTL_POWER_BANK7, SYSCTL_POWER_V18);

  fpioa_set_function(PMU_PIN_SCL, FUNC_I2C1_SCLK);
  fpioa_set_function(PMU_PIN_SDA, FUNC_I2C1_SDA);
  i2c_init(I2C_DEVICE_1, PMU_ADDR, 7, 100000);

  // Sipeed's own wiki: "MaixCube needs to configure the power chip before
  // using the LCD, otherwise the screen will be blurred; in MaixPy firmware
  // this is configured automatically" -- i.e. this write is real and
  // necessary, not a guess. The exact three register values are copied
  // as-is from MaixPy-v1's Amigo board (projects/maixpy_amigo_ips/
  // builtin_py/_boot.py), which uses the same AXP173; NOT yet confirmed
  // these are the right values for the Cube specifically (Amigo's LCD is a
  // different panel on different rails) -- worth revisiting if the picture
  // is still wrong once the pixel-format bug below is fixed.
  static const uint8_t regs[][2] = {{0x27, 0x20}, {0x28, 0x0C}, {0x36, 0xCC}};
  for (size_t i = 0; i < 3; i++) {
    uint8_t b[2] = {regs[i][0], regs[i][1]};
    i2c_send_data(I2C_DEVICE_1, b, 2);
  }
  return 0;
}

// ------------------------------------------------------------------ LCD ----

#define LCD_PIN_CS 36
#define LCD_PIN_RST 37
#define LCD_PIN_DCX 38
#define LCD_PIN_CLK 39 // named LCD_WR in cube.config.json (an 8080-bus name); used here as plain SPI0_SCLK
#define LCD_PIN_BL 17 // backlight, plain GPIO -- confirmed working (toggles visibly on real hardware)
#define LCD_RST_GPIOHS 30
#define LCD_DCX_GPIOHS 31
#define LCD_BL_GPIOHS 2
#define LCD_SPI SPI_DEVICE_0
#define LCD_CS SPI_CHIP_SELECT_3
#define LCD_DMA DMAC_CHANNEL2
#define LCD_FREQ 15000000 // MaixPy's default for this LCD type

// MADCTL and the panel offset inside the ST7789's 240x320 RAM. MaixPy's
// DEV_CUBE_IPS_240x240 starts from direction 0 with no offset (an offset of 80
// applies only to the directions that swap X/Y).
#ifndef BOARD_LCD_MADCTL
#define BOARD_LCD_MADCTL 0x00
#endif
#ifndef LCD_OFFSET_X
#define LCD_OFFSET_X 0
#endif
#ifndef LCD_OFFSET_Y
#define LCD_OFFSET_Y 0
#endif

// This panel is wired as a genuine 8-bit parallel bus (DB0..DB7, CS, DC, WR,
// RD, RESET -- confirmed straight from the board's own schematic: the LCD
// FPC connector section is explicitly labelled "8bit MCU LCD"), not simple
// serial SPI. That's why an earlier attempt at driving it with a single SPI
// data wire (any one of several candidate pins) never produced a correct
// picture regardless of which pin was tried: with only 1 of 8 required data
// bits present, the panel never receives a complete, valid byte.
//
// The 8 data lines are labelled "SPI0_D0".."SPI0_D7" in every Sipeed source
// that mentions them (this schematic, the M1n module's own datasheet) --
// but unlike every other signal on this board, they are never given a raw
// K210 IO pin number anywhere in Sipeed's public documentation, and neither
// MaixPy-v1's py_lcd.c nor Maixduino's st7789.c/pins_arduino.h (both
// checked directly) ever call fpioa_set_function() for them, on this board
// or any other Sipeed board using the same octal LCD path (TWatch). That
// consistent absence is itself the evidence for the theory this code acts
// on: these 8 lines are not ordinary FPIOA-routable IO pins at all, but
// fixed hardware pads whose routing is switched by the single
// sysctl_set_spi0_dvp_data() bit (shared with the DVP camera's own 8-bit
// data bus) -- see sysctl.h's spi_dvp_data_enable field. Confirmed on real
// hardware: with genuine octal SPI framing (SPI_FF_OCTAL, matching
// Maixduino's cores/arduino/st7789.c tft_write_command/tft_write_word
// exactly) and no fpioa_set_function() call for the data lines at all, a
// full-screen fill reaches the panel and displays evenly -- proving this
// path, not a per-pin FPIOA mapping, is what's needed.
//
// NOT YET RESOLVED: the fill colour comes out wrong (seen on real hardware:
// a uniform pink/purple instead of the intended solid colour, and a later
// attempt at solid green was not visibly different from the broken states
// before it) -- a channel/byte-order bug in the pixel path, not a
// structural one, but not tracked down yet. Also unconfirmed: which value
// of sysctl_set_spi0_dvp_data() is actually correct for the LCD (0 is used
// below; on real hardware neither 0 nor 1 was confirmed to give a correct
// colour, though both produce a full, evenly-filled screen rather than
// noise). See ../README.md's Cube status section before spending more time
// here -- the next thing to check is the pixel packing: mcu_lcd_draw_picture
// in MaixPy-v1's lcd_mcu.c packs two RGB565 pixels per 32-bit word with a
// specific byte-swap-and-reorder (SWAP_16(p[1]), SWAP_16(p[0])) that this
// code does not yet replicate exactly for board_lcd_show()'s bulk path
// below (only fpioa/omission of camera use during LCD test was checked).
static void lcd_cmd(uint8_t c) {
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_LOW);
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_OCTAL, 8, 0);
  spi_init_non_standard(LCD_SPI, 8, 0, 0, SPI_AITM_AS_FRAME_FORMAT);
  spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, &c, 1, SPI_TRANS_CHAR);
}

static void lcd_data8(const uint8_t *d, uint32_t n) {
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_OCTAL, 8, 0);
  spi_init_non_standard(LCD_SPI, 8, 0, 0, SPI_AITM_AS_FRAME_FORMAT);
  spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, d, n, SPI_TRANS_CHAR);
}

static void lcd_reg(uint8_t c, const uint8_t *d, uint32_t n) {
  lcd_cmd(c);
  if (n) lcd_data8(d, n);
}

#define LCD_REG(c, ...) \
  do { static const uint8_t d_[] = {__VA_ARGS__}; lcd_reg(c, d_, sizeof(d_)); } while (0)

void board_lcd_init(void) {
  fpioa_set_function(LCD_PIN_RST, FUNC_GPIOHS0 + LCD_RST_GPIOHS);
  fpioa_set_function(LCD_PIN_DCX, FUNC_GPIOHS0 + LCD_DCX_GPIOHS);
  fpioa_set_function(LCD_PIN_CS, FUNC_SPI0_SS0 + 3);
  fpioa_set_function(LCD_PIN_CLK, FUNC_SPI0_SCLK);
  fpioa_set_function(LCD_PIN_BL, FUNC_GPIOHS0 + LCD_BL_GPIOHS);
  gpiohs_set_drive_mode(LCD_BL_GPIOHS, GPIO_DM_OUTPUT);
  gpiohs_set_pin(LCD_BL_GPIOHS, GPIO_PV_HIGH);

  // See the big comment above: no fpioa_set_function() for the 8 data
  // lines. sysctl_set_spi0_dvp_data(0) picks which of the two peripherals
  // (this LCD vs. the DVP camera) the shared data pads are currently
  // routed to; call it again with 1 before using the camera.
  sysctl_set_spi0_dvp_data(0);

  gpiohs_set_drive_mode(LCD_DCX_GPIOHS, GPIO_DM_OUTPUT);
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  gpiohs_set_drive_mode(LCD_RST_GPIOHS, GPIO_DM_OUTPUT);
  gpiohs_set_pin(LCD_RST_GPIOHS, GPIO_PV_LOW);
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_OCTAL, 8, 0);
  spi_set_clk_rate(LCD_SPI, LCD_FREQ);
  msleep(50);
  gpiohs_set_pin(LCD_RST_GPIOHS, GPIO_PV_HIGH);
  msleep(50);

  lcd_cmd(0x01); // software reset
  msleep(50);
  lcd_cmd(0x11); // sleep out
  msleep(120);
  LCD_REG(0x3A, 0x55); // pixel format
  msleep(10);
  LCD_REG(0x36, BOARD_LCD_MADCTL);
  lcd_cmd(0x21); // display inversion on (IPS panel; cube.config.json has invert: 1)
  msleep(10);
  lcd_cmd(0x13); // normal display mode
  msleep(10);
  lcd_cmd(0x29); // display on
}

static void lcd_set_area(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2) {
  x1 += LCD_OFFSET_X; x2 += LCD_OFFSET_X;
  y1 += LCD_OFFSET_Y; y2 += LCD_OFFSET_Y;
  uint8_t d[4] = {x1 >> 8, x1 & 0xFF, x2 >> 8, x2 & 0xFF};
  lcd_reg(0x2A, d, 4);
  d[0] = y1 >> 8; d[1] = y1 & 0xFF; d[2] = y2 >> 8; d[3] = y2 & 0xFF;
  lcd_reg(0x2B, d, 4);
  lcd_cmd(0x2C); // memory write
}

static inline uint16_t swap16(uint16_t v) { return (uint16_t)((v >> 8) | (v << 8)); }

void board_lcd_show(const uint16_t *pixels) {
  lcd_set_area(0, 0, BOARD_LCD_W - 1, BOARD_LCD_H - 1);
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  // 32-bit frames, two RGB565 pixels packed per word: matches
  // MaixPy-v1's mcu_lcd_draw_picture()/Maixduino's tft_write_word exactly
  // (the byte-swap is required there; NOT yet re-verified as correct here
  // -- see the big comment above this file's LCD section, this is the next
  // thing to check against real hardware).
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_OCTAL, 32, 0);
  spi_init_non_standard(LCD_SPI, 0, 32, 0, SPI_AITM_AS_FRAME_FORMAT);
  size_t n = (size_t)BOARD_LCD_W * BOARD_LCD_H;
  for (size_t i = 0; i + 1 < n; i += 2) {
    uint32_t word = (uint32_t)swap16(pixels[i + 1]) | ((uint32_t)swap16(pixels[i]) << 16);
    spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, &word, 1, SPI_TRANS_INT);
  }
}

// -------------------------------------------------------------- buttons ----

#define BTN_A_PIN 10 // KEY1
#define BTN_B_PIN 11 // KEY2
#define BTN_A_GPIOHS 0
#define BTN_B_GPIOHS 1

void board_buttons_init(void) {
  fpioa_set_function(BTN_A_PIN, FUNC_GPIOHS0 + BTN_A_GPIOHS);
  fpioa_set_function(BTN_B_PIN, FUNC_GPIOHS0 + BTN_B_GPIOHS);
  gpiohs_set_drive_mode(BTN_A_GPIOHS, GPIO_DM_INPUT_PULL_UP);
  gpiohs_set_drive_mode(BTN_B_GPIOHS, GPIO_DM_INPUT_PULL_UP);
}

int board_button_a(void) { return gpiohs_get_pin(BTN_A_GPIOHS) == GPIO_PV_LOW; }
int board_button_b(void) { return gpiohs_get_pin(BTN_B_GPIOHS) == GPIO_PV_LOW; }

#endif // BOARD_CUBE
