// M5StickV board support: AXP192 power rails, the 135x240 ST7789 LCD on
// standard SPI0, and the two front buttons. See board.h. Pin numbers, register
// values and command sequences are from Sipeed's MaixPy-v1 (the firmware the
// board ships with) -- see ../../../THIRD_PARTY_NOTICES.md.

#ifdef BOARD_M5STICKV

#include "board.h"

#include <stddef.h>
#include <stdio.h>

#include "dmac.h"
#include "dvp.h"
#include "fpioa.h"
#include "gpiohs.h"
#include "i2c.h"
#include "iomem.h"
#include "plic.h"
#include "sleep.h"
#include "spi.h"
#include "sysctl.h"

// ---------------------------------------------------------------- power ----

#define AXP192_ADDR 0x34
#define AXP192_PIN_SCL 28
#define AXP192_PIN_SDA 29
#define VBUS_GPIOHS 26 // gpiohs index used for the "VBUS as input" control pin

static int axp_write(uint8_t reg, uint8_t val) {
  uint8_t b[2] = {reg, val};
  return i2c_send_data(I2C_DEVICE_0, b, 2);
}

int board_power_init(void) {
  sysctl_set_power_mode(SYSCTL_POWER_BANK3, SYSCTL_POWER_V33);
  fpioa_set_function(AXP192_PIN_SCL, FUNC_I2C0_SCLK);
  fpioa_set_function(AXP192_PIN_SDA, FUNC_I2C0_SDA);
  i2c_init(I2C_DEVICE_0, AXP192_ADDR, 7, 400000);

  // Probe: register 0x03 is the AXP192 chip id.
  uint8_t reg = 0x03, id = 0;
  if (i2c_recv_data(I2C_DEVICE_0, &reg, 1, &id, 1) != 0) return -1;
  printf("AXP192 chip id 0x%02x\n", id);

  static const uint8_t seq[][2] = {
      {0x23, 0x08}, // DC-DC2 (K210 core) 0.9V
      {0x33, 0xC1}, // 190mA charge current
      {0x36, 0x6C}, // 4s power-key shutdown
      {0x91, 0xF0}, // GPIO0 LDO 3.3V: LCD backlight
      {0x90, 0x02}, // GPIO0 in LDO mode
      {0x28, 0xF0}, // LDO2 3.3V / LDO3 1.8V (camera + LCD rails)
      {0x27, 0x2C}, // DC-DC3 1.8V
      {0x12, 0xFF}, // enable all outputs and EXTEN
      {0x23, 0x08}, // DC-DC2 0.9V
      {0x31, 0x03}, // battery cutoff 3.2V
      {0x39, 0xFC}, // temperature protection off (no sensor fitted)
  };
  for (size_t i = 0; i < sizeof(seq) / sizeof(seq[0]); i++) {
    if (axp_write(seq[i][0], seq[i][1]) != 0) return -2;
  }
  {
    static const uint8_t regs[] = {0x12, 0x23, 0x27, 0x28, 0x90, 0x91};
    printf("AXP192 readback:");
    for (size_t i = 0; i < sizeof(regs); i++) {
      uint8_t r = regs[i], v = 0xEE;
      i2c_recv_data(I2C_DEVICE_0, &r, 1, &v, 1);
      printf(" [0x%02x]=0x%02x", r, v);
    }
    printf("\n");
  }
  // BAT -> 5V boost -> VBUS: don't use VBUS as an input.
  fpioa_set_function(23, FUNC_GPIOHS0 + VBUS_GPIOHS);
  gpiohs_set_drive_mode(VBUS_GPIOHS, GPIO_DM_OUTPUT);
  gpiohs_set_pin(VBUS_GPIOHS, GPIO_PV_HIGH);
  msleep(20);
  return 0;
}

// ------------------------------------------------------------------ LCD ----

#define LCD_PIN_RST 21
#define LCD_PIN_DCX 20
#define LCD_PIN_CS 22
#define LCD_PIN_SCLK 19
#define LCD_PIN_MOSI 18
#define LCD_RST_GPIOHS 30
#define LCD_DCX_GPIOHS 31
#define LCD_SPI SPI_DEVICE_0
#define LCD_CS SPI_CHIP_SELECT_3
#define LCD_DMA DMAC_CHANNEL2
#define LCD_FREQ 40000000

// MADCTL for landscape. MaixPy's lcd.rotation(2) is DIR_YX_LRUD (0x60); if the
// picture comes out upside down or mirrored on a particular unit, the other
// candidates are 0x20 / 0xA0 / 0xE0.
#ifndef BOARD_LCD_MADCTL
#define BOARD_LCD_MADCTL 0x60
#endif
// The 135x240 panel sits at an offset inside the ST7789's 240x320 RAM;
// these are the landscape (MV=1) values.
#define LCD_OFFSET_X 40
#define LCD_OFFSET_Y 52

static void lcd_cmd(uint8_t c) {
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_LOW);
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_STANDARD, 8, 0);
  spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, &c, 1, SPI_TRANS_CHAR);
}

static void lcd_data(const uint8_t *d, uint32_t n) {
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_STANDARD, 8, 0);
  spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, d, n, SPI_TRANS_CHAR);
}

static void lcd_reg(uint8_t c, const uint8_t *d, uint32_t n) {
  lcd_cmd(c);
  if (n) lcd_data(d, n);
}

#define LCD_REG(c, ...) \
  do { static const uint8_t d_[] = {__VA_ARGS__}; lcd_reg(c, d_, sizeof(d_)); } while (0)

void board_lcd_init(void) {
  fpioa_set_function(LCD_PIN_RST, FUNC_GPIOHS0 + LCD_RST_GPIOHS);
  fpioa_set_function(LCD_PIN_DCX, FUNC_GPIOHS0 + LCD_DCX_GPIOHS);
  fpioa_set_function(LCD_PIN_CS, FUNC_SPI0_SS0 + 3);
  fpioa_set_function(LCD_PIN_SCLK, FUNC_SPI0_SCLK);
  fpioa_set_function(LCD_PIN_MOSI, FUNC_SPI0_D0);

  gpiohs_set_drive_mode(LCD_DCX_GPIOHS, GPIO_DM_OUTPUT);
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  gpiohs_set_drive_mode(LCD_RST_GPIOHS, GPIO_DM_OUTPUT);
  gpiohs_set_pin(LCD_RST_GPIOHS, GPIO_PV_LOW);
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_STANDARD, 8, 0);
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
  lcd_cmd(0x21); // display inversion on (this panel is inverted)
  msleep(10);
  lcd_cmd(0x13); // normal display mode
  msleep(10);
  lcd_cmd(0x29); // display on

  // M5StickV panel tuning (projects/maixpy_m5stickv/builtin_py/_boot.py).
  LCD_REG(0x3A, 0x05);
  LCD_REG(0xB2, 0x05, 0x05, 0x00, 0x33, 0x33);
  LCD_REG(0xB7, 0x23);
  LCD_REG(0xBB, 0x22);
  LCD_REG(0xC0, 0x2C);
  LCD_REG(0xC2, 0x01);
  LCD_REG(0xC3, 0x13);
  LCD_REG(0xC4, 0x20);
  LCD_REG(0xC6, 0x0F);
  LCD_REG(0xD0, 0xA4, 0xA1);
  LCD_REG(0xD6, 0xA1);
  LCD_REG(0xE0, 0x23, 0x70, 0x06, 0x0C, 0x08, 0x09, 0x27, 0x2E, 0x34, 0x46, 0x37, 0x13, 0x13, 0x25, 0x2A);
  LCD_REG(0xE1, 0x70, 0x04, 0x08, 0x09, 0x07, 0x03, 0x2C, 0x42, 0x42, 0x38, 0x14, 0x14, 0x27, 0x2C);
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

void board_lcd_show(const uint16_t *pixels) {
  lcd_set_area(0, 0, BOARD_LCD_W - 1, BOARD_LCD_H - 1);
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  // 16-bit frames go out MSB first, i.e. RGB565 high byte first, as the ST7789 wants.
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_STANDARD, 16, 0);
  spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, pixels, (size_t)BOARD_LCD_W * BOARD_LCD_H, SPI_TRANS_SHORT);
}

// -------------------------------------------------------------- buttons ----

#define BTN_A_PIN 36
#define BTN_B_PIN 37
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

#endif // BOARD_M5STICKV
