// M5StickV board support for the camera demo: AXP192 power rails, the OV7740
// camera on the K210's DVP port, and the 135x240 ST7789 LCD on SPI0.
//
// kendryte-standalone-sdk has no driver for any of this (its camera/LCD demos
// target other boards), so the pin map, the AXP192 register sequence, the
// OV7740 register table and the ST7789 setup are taken from Sipeed's
// MaixPy-v1 (the firmware M5StickV ships with) -- see ../README.md and
// ../../../THIRD_PARTY_NOTICES.md.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Landscape LCD, and the DVP window: the sensor outputs 320x240 and the DVP
// crops the top-left W x H, so camera pixels map 1:1 onto LCD pixels.
#define M5_LCD_W 240
#define M5_LCD_H 135
#define M5_CAM_W M5_LCD_W
#define M5_CAM_H M5_LCD_H

// Returns 0 on success. Must run first: it powers the LCD backlight and the
// camera/LCD rails.
int m5_power_init(void);
void m5_lcd_init(void);

// Pushes a host-order (R<<11|G<<5|B) RGB565 frame of M5_LCD_W x M5_LCD_H.
// `pixels` must be in iomem (non-cached) memory -- SPI DMA reads it.
void m5_lcd_show(const uint16_t *pixels);

// Returns the sensor's chip id (0x7742 for OV7740), or 0 if none answered.
uint16_t m5_camera_init(void);
// Blocks until a frame has been captured into the DVP's buffers (or the
// timeout passes; returns -1). The buffers then stay untouched until the next
// m5_camera_snapshot() call.
int m5_camera_snapshot(uint32_t timeout_ms);
// The DVP's RGB "AI" output: three consecutive planes R, G, B, each
// M5_CAM_W*M5_CAM_H bytes, 0..255. The OV7740 emits YUV422; the DVP converts
// it to RGB only for this output (the "display" buffer stays YUV, so don't
// show it as RGB565 -- that gives a grey, green-tinted picture).
const uint8_t *m5_camera_ai_planes(void);

// Builds a host-order RGB565 LCD frame (`out` as for m5_lcd_show) from the AI
// planes -- the picture and the model then share one data source.
void m5_ai_to_rgb565(const uint8_t *planes, uint16_t *out);

// Allocates a non-cached buffer (iomem); NULL on failure.
void *m5_alloc_dma(uint32_t bytes);

// Front buttons, active-low GPIOs; returns 1 while pressed.
void m5_buttons_init(void);
int m5_button_a(void);
int m5_button_b(void);

#ifdef __cplusplus
}
#endif
