// Board and camera interface for the camera demo. One firmware per board,
// chosen at build time (BOARD=m5stickv|cube, see build.sh); the OV7740 camera
// code is shared, everything else (power chip, LCD, buttons) is per board.
#pragma once

#include <stdint.h>

#if defined(BOARD_M5STICKV)
#define BOARD_NAME "M5StickV"
#define BOARD_LCD_W 240
#define BOARD_LCD_H 135
#elif defined(BOARD_CUBE)
#define BOARD_NAME "Maix Cube"
#define BOARD_LCD_W 240
#define BOARD_LCD_H 240
#else
#error "define BOARD_M5STICKV or BOARD_CUBE (build.sh does this)"
#endif

// The DVP window: the sensor outputs 320x240 and the DVP crops the top-left
// CAM_W x CAM_H, so camera pixels map 1:1 onto LCD pixels.
#define CAM_W BOARD_LCD_W
#define CAM_H BOARD_LCD_H

#ifdef __cplusplus
extern "C" {
#endif

// ---- per board ----

// Returns 0 on success. Runs first: powers the rails / backlight the LCD and
// camera need.
int board_power_init(void);
void board_lcd_init(void);
// Pushes a host-order (R<<11|G<<5|B) RGB565 frame of BOARD_LCD_W x BOARD_LCD_H.
// `pixels` must be in iomem (non-cached) memory -- SPI DMA reads it.
void board_lcd_show(const uint16_t *pixels);
// Two front buttons, active-low GPIOs; returns 1 while pressed.
void board_buttons_init(void);
int board_button_a(void);
int board_button_b(void);

// ---- OV7740 camera (shared) ----

// Returns the sensor's chip id (0x7742 for OV7740), or 0 if none answered.
uint16_t cam_init(void);
// Blocks until a frame has been captured into the DVP's buffers (or the
// timeout passes; returns -1). The buffers then stay untouched until the next
// cam_snapshot() call.
int cam_snapshot(uint32_t timeout_ms);
// The DVP's RGB "AI" output: three consecutive planes R, G, B, each
// CAM_W*CAM_H bytes, 0..255. The OV7740 emits YUV422; the DVP converts it to
// RGB only for this output (the "display" buffer stays YUV, so don't show it
// as RGB565 -- that gives a grey, green-tinted picture).
const uint8_t *cam_ai_planes(void);
// Builds a host-order RGB565 LCD frame (`out` as for board_lcd_show) from the
// AI planes -- the picture and the model then share one data source.
void cam_ai_to_rgb565(const uint8_t *planes, uint16_t *out);

// Allocates a non-cached buffer (iomem); NULL on failure.
void *dma_alloc(uint32_t bytes);

#ifdef __cplusplus
}
#endif
