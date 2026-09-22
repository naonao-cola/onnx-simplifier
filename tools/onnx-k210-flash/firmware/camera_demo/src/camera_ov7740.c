// OV7740 camera + DVP capture, shared by every board this demo supports (the
// M5StickV and the Maix Cube use the same sensor on the same pins). Register
// table and setup are from Sipeed's MaixPy-v1 / OpenMV -- see
// ../../../THIRD_PARTY_NOTICES.md.

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

void *dma_alloc(uint32_t bytes) { return iomem_malloc(bytes); }

uint16_t cam_init(void) {
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

  g_raw = iomem_malloc(CAM_W * CAM_H * 2);
  g_ai = iomem_malloc(CAM_W * CAM_H * 3);
  if (!g_raw || !g_ai) return 0;

  dvp_set_image_format(DVP_CFG_YUV_FORMAT); // the sensor emits YUV422; the DVP converts it to RGB for the AI output
  dvp_disable_burst();
  dvp_disable_auto();
  dvp_set_output_enable(DVP_OUTPUT_AI, 1);
  dvp_set_output_enable(DVP_OUTPUT_DISPLAY, 1);
  dvp_set_image_size(CAM_W, CAM_H);
  dvp_set_ai_addr((uint32_t)g_ai, (uint32_t)(g_ai + CAM_W * CAM_H),
                  (uint32_t)(g_ai + 2 * CAM_W * CAM_H));
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

int cam_snapshot(uint32_t timeout_ms) {
  g_frame_done = 0;
  uint64_t deadline = sysctl_get_time_us() + (uint64_t)timeout_ms * 1000;
  while (!g_frame_done) {
    if (sysctl_get_time_us() > deadline) return -1;
  }
  return 0;
}

const uint8_t *cam_ai_planes(void) { return g_ai; }

void cam_ai_to_rgb565(const uint8_t *planes, uint16_t *out) {
  const uint8_t *r = planes, *g = planes + CAM_W * CAM_H, *b = planes + 2 * CAM_W * CAM_H;
  for (size_t i = 0; i < (size_t)CAM_W * CAM_H; i++) {
    out[i] = (uint16_t)(((r[i] >> 3) << 11) | ((g[i] >> 2) << 5) | (b[i] >> 3));
  }
}
