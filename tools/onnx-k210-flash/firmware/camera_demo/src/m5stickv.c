// See m5stickv.h. Pin numbers, AXP192 register values, OV7740 register table
// and ST7789 command sequence come from Sipeed's MaixPy-v1
// (components/boards/m5stick, components/micropython/port/src/omv/ov7740.c and
// sensor.c, components/drivers/lcd, projects/maixpy_m5stickv/builtin_py) --
// Apache 2.0 / MIT (the OV7740 table is OpenMV's), see THIRD_PARTY_NOTICES.md.

#include "m5stickv.h"

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

int m5_power_init(void) {
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
#ifndef M5_LCD_MADCTL
#define M5_LCD_MADCTL 0x60
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

void m5_lcd_init(void) {
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
  LCD_REG(0x36, M5_LCD_MADCTL);
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

void m5_lcd_show(const uint16_t *pixels) {
  lcd_set_area(0, 0, M5_LCD_W - 1, M5_LCD_H - 1);
  gpiohs_set_pin(LCD_DCX_GPIOHS, GPIO_PV_HIGH);
  // 16-bit frames go out MSB first, i.e. RGB565 high byte first, as the ST7789 wants.
  spi_init(LCD_SPI, SPI_WORK_MODE_0, SPI_FF_STANDARD, 16, 0);
  spi_send_data_normal_dma(LCD_DMA, LCD_SPI, LCD_CS, pixels, (size_t)M5_LCD_W * M5_LCD_H, SPI_TRANS_SHORT);
}

// --------------------------------------------------------------- camera ----

#define CAM_PIN_PCLK 47
#define CAM_PIN_XCLK 46
#define CAM_PIN_HREF 45
#define CAM_PIN_PWDN 44
#define CAM_PIN_VSYNC 43
#define CAM_PIN_RST 42
// MaixPy's default sensor config for this board has i2c_num = 2: the camera
// is programmed through hardware I2C2 on these pins, not the DVP's own SCCB
// block (which never got an ACK from the sensor here).
#define CAM_PIN_SCL 41
#define CAM_PIN_SDA 40
#define CAM_I2C I2C_DEVICE_2

#define OV7740_ADDR 0x21 // 7-bit address (SCCB id 0x42)
#define OV7740_ID 0x7742

// OpenMV's default OV7740 setup (QVGA 320x240, 60 fps), via MaixPy.
static const uint8_t ov7740_regs[][2] = {
    {0x47, 0x02}, {0x17, 0x27}, {0x04, 0x40}, {0x1B, 0x81}, {0x29, 0x17}, {0x5F, 0x03}, {0x3A, 0x09},
    {0x33, 0x44}, {0x68, 0x1A}, {0x14, 0x38}, {0x5F, 0x04}, {0x64, 0x00}, {0x67, 0x90}, {0x27, 0x80},
    {0x45, 0x41}, {0x4B, 0x40}, {0x36, 0x2f}, {0x11, 0x00}, {0x36, 0x3f}, {0x12, 0x00}, {0x17, 0x25},
    {0x18, 0xa0}, {0x1a, 0xf0}, {0x31, 0x50}, {0x32, 0x78}, {0x82, 0x3f}, {0x85, 0x08}, {0x86, 0x02},
    {0x87, 0x01}, {0xd5, 0x10}, {0x0d, 0x34}, {0x19, 0x03}, {0x2b, 0xf8}, {0x2c, 0x01}, {0x53, 0x00},
    {0x89, 0x30}, {0x8d, 0x30}, {0x8f, 0x85}, {0x93, 0x30}, {0x95, 0x85}, {0x99, 0x30}, {0x9b, 0x85},
    {0xac, 0x6E}, {0xbe, 0xff}, {0xbf, 0x00}, {0x38, 0x14}, {0xe9, 0x00}, {0x3D, 0x08}, {0x3E, 0x80},
    {0x3F, 0x40}, {0x40, 0x7F}, {0x41, 0x6A}, {0x42, 0x29}, {0x49, 0x64}, {0x4A, 0xA1}, {0x4E, 0x13},
    {0x4D, 0x50}, {0x44, 0x58}, {0x4C, 0x1A}, {0x4E, 0x14}, {0x38, 0x11}, {0x84, 0x70},
};

static uint8_t *g_raw;      // DVP display output: YUV422, unused but the DVP wants somewhere to write it
static uint8_t *g_ai;       // DVP AI output: R, G, B planes
static volatile int g_frame_done;

static int dvp_irq(void *ctx) {
  (void)ctx;
  if (dvp_get_interrupt(DVP_STS_FRAME_FINISH)) {
    dvp_clear_interrupt(DVP_STS_FRAME_START | DVP_STS_FRAME_FINISH);
    g_frame_done = 1;
  } else {
    // Frame start: convert this frame only if the previous one has been consumed.
    if (g_frame_done == 0) dvp_start_convert();
    dvp_clear_interrupt(DVP_STS_FRAME_START);
  }
  return 0;
}

// SCCB-style register access over I2C: a read is a write of the register
// number, a stop, then a separate read.
static int ov_read(uint8_t reg, uint8_t *val) {
  if (i2c_send_data(CAM_I2C, &reg, 1) != 0) return -1;
  return i2c_recv_data(CAM_I2C, NULL, 0, val, 1);
}

static uint16_t ov_id(void) {
  uint8_t hi = 0, lo = 0;
  if (ov_read(0x0A, &hi) != 0 || ov_read(0x0B, &lo) != 0) return 0;
  return (uint16_t)((hi << 8) | lo);
}

static void ov_write(uint8_t reg, uint8_t val) {
  uint8_t b[2] = {reg, val};
  i2c_send_data(CAM_I2C, b, 2);
  msleep(5);
}

void *m5_alloc_dma(uint32_t bytes) { return iomem_malloc(bytes); }

uint16_t m5_camera_init(void) {
  sysctl_set_power_mode(SYSCTL_POWER_BANK6, SYSCTL_POWER_V18);
  sysctl_set_power_mode(SYSCTL_POWER_BANK7, SYSCTL_POWER_V18);
  fpioa_set_function(CAM_PIN_PCLK, FUNC_CMOS_PCLK);
  fpioa_set_function(CAM_PIN_XCLK, FUNC_CMOS_XCLK);
  fpioa_set_function(CAM_PIN_HREF, FUNC_CMOS_HREF);
  fpioa_set_function(CAM_PIN_PWDN, FUNC_CMOS_PWDN);
  fpioa_set_function(CAM_PIN_VSYNC, FUNC_CMOS_VSYNC);
  fpioa_set_function(CAM_PIN_RST, FUNC_CMOS_RST);
  fpioa_set_function(CAM_PIN_SCL, FUNC_I2C2_SCLK);
  fpioa_set_function(CAM_PIN_SDA, FUNC_I2C2_SDA);
  i2c_init(CAM_I2C, OV7740_ADDR, 7, 100000);

  // The K210 shares one 8-bit data bus between SPI0 and the DVP; this switch
  // connects it to the pins. MaixPy and the SDK's camera demos set it at boot.
  // Without it the frame interrupts still fire (sync lines work) but every
  // pixel reads as zero: a black picture and a model that never sees anything.
  sysctl_set_spi0_dvp_data(1);
  msleep(300); // let the camera rails settle
  dvp_init(8); // also pulses RESET/PWDN
  dvp_set_xclk_rate(12000000);
  dvp->cmos_cfg |= DVP_CMOS_POWER_DOWN;
  msleep(10);
  dvp->cmos_cfg &= ~DVP_CMOS_POWER_DOWN;
  msleep(10);

  uint16_t id = ov_id();
  if (id != OV7740_ID) {
    // Second chance: pulse RESET.
    dvp->cmos_cfg &= ~DVP_CMOS_RESET;
    msleep(10);
    dvp->cmos_cfg |= DVP_CMOS_RESET;
    msleep(30);
    id = ov_id();
    if (id != OV7740_ID) {
      return id;
    }
  }

  dvp_set_xclk_rate(22000000);
  msleep(10);
  ov_write(0x12, 0x80); // register reset
  msleep(2);
  for (size_t i = 0; i < sizeof(ov7740_regs) / sizeof(ov7740_regs[0]); i++) {
    ov_write(ov7740_regs[i][0], ov7740_regs[i][1]);
  }

  g_raw = iomem_malloc(M5_CAM_W * M5_CAM_H * 2);
  g_ai = iomem_malloc(M5_CAM_W * M5_CAM_H * 3);
  if (!g_raw || !g_ai) return 0;

  dvp_set_image_format(DVP_CFG_YUV_FORMAT); // the sensor emits YUV422; the DVP converts it to RGB for the AI output
  dvp_disable_burst();
  dvp_disable_auto();
  dvp_set_output_enable(DVP_OUTPUT_AI, 1);
  dvp_set_output_enable(DVP_OUTPUT_DISPLAY, 1);
  dvp_set_image_size(M5_CAM_W, M5_CAM_H);
  dvp_set_ai_addr((uint32_t)g_ai, (uint32_t)(g_ai + M5_CAM_W * M5_CAM_H),
                  (uint32_t)(g_ai + 2 * M5_CAM_W * M5_CAM_H));
  dvp_set_display_addr((uint32_t)g_raw);

  dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 0);
  plic_set_priority(IRQN_DVP_INTERRUPT, 2);
  plic_irq_register(IRQN_DVP_INTERRUPT, dvp_irq, NULL);
  plic_irq_disable(IRQN_DVP_INTERRUPT);
  dvp_clear_interrupt(DVP_STS_FRAME_START | DVP_STS_FRAME_FINISH);
  dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 1);
  plic_irq_enable(IRQN_DVP_INTERRUPT);
  g_frame_done = 1; // nothing pending until the first snapshot
  return id;
}

int m5_camera_snapshot(uint32_t timeout_ms) {
  g_frame_done = 0;
  uint64_t deadline = sysctl_get_time_us() + (uint64_t)timeout_ms * 1000;
  while (!g_frame_done) {
    if (sysctl_get_time_us() > deadline) return -1;
  }
  return 0;
}

const uint8_t *m5_camera_ai_planes(void) { return g_ai; }

// -------------------------------------------------------------- buttons ----

#define BTN_A_PIN 36
#define BTN_B_PIN 37
#define BTN_A_GPIOHS 0
#define BTN_B_GPIOHS 1

void m5_buttons_init(void) {
  fpioa_set_function(BTN_A_PIN, FUNC_GPIOHS0 + BTN_A_GPIOHS);
  fpioa_set_function(BTN_B_PIN, FUNC_GPIOHS0 + BTN_B_GPIOHS);
  gpiohs_set_drive_mode(BTN_A_GPIOHS, GPIO_DM_INPUT_PULL_UP);
  gpiohs_set_drive_mode(BTN_B_GPIOHS, GPIO_DM_INPUT_PULL_UP);
}

int m5_button_a(void) { return gpiohs_get_pin(BTN_A_GPIOHS) == GPIO_PV_LOW; }
int m5_button_b(void) { return gpiohs_get_pin(BTN_B_GPIOHS) == GPIO_PV_LOW; }

// ------------------------------------------------------- display helpers ----

void m5_ai_to_rgb565(const uint8_t *planes, uint16_t *out) {
  const uint8_t *r = planes, *g = planes + M5_CAM_W * M5_CAM_H, *b = planes + 2 * M5_CAM_W * M5_CAM_H;
  for (size_t i = 0; i < (size_t)M5_CAM_W * M5_CAM_H; i++) {
    out[i] = (uint16_t)(((r[i] >> 3) << 11) | ((g[i] >> 2) << 5) | (b[i] >> 3));
  }
}
