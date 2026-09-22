// M5StickV camera demo: live camera -> the .kmodel flashed at 0x00C00000 (KPU)
// -> LCD, on top of the same nncase 1.9.0 runtime as ../../runtime.
//
// Flash layout is the runtime's: this firmware at 0x0, a kmodel at 0x00C00000.
// The model must take a float32 1x3xHxW input in [0,1] (the default of
// ../../../scripts/onnx_to_kmodel.py). The camera frame is 240x135 (the LCD's
// size), resized nearest-neighbour to HxW. Output handling is generic:
//   - a single-channel spatial map (1x1xhxw, float32) is shown as a greyscale
//     picture -- see ../examples/make_edge_model.py for one;
//   - a short vector (classification logits) is reported as top-5 on UART,
//     with a confidence bar on the LCD.
//
// Buttons: A cycles the view (camera / model output / split), B cycles the
// output gain. UART (115200): prints stats once a second; send 'v' / 'g' to do
// what A / B do, and 'd' to dump the current frame, model input and output
// (see ../scripts/grab.py).

#include <cmath>
#include <cstdio>
#include <cstring>

#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

#include "dmac.h"
#include "fpioa.h"
#include "plic.h"
#include "sysctl.h"
#include "uarths.h"
extern "C" {
#include "board.h"
#include "w25qxx.h"
}

using namespace nncase;
using namespace nncase::runtime;

namespace {

constexpr uint32_t kModelFlashAddress = 0x00C00000;
constexpr size_t kModelBufferSize = 2 * 1024 * 1024;
constexpr size_t kMaxInputFloats = 3 * 240 * 240;
constexpr int kW = CAM_W, kH = CAM_H;

alignas(256) uint8_t g_model_buffer[kModelBufferSize];
float g_input[kMaxInputFloats];

enum View { kCamera, kModel, kSplit, kViewCount };
int g_view = kSplit;
int g_gain_idx = 1;
const float kGains[] = {0.5f, 1.f, 2.f, 4.f, 8.f};
constexpr int kGainCount = sizeof(kGains) / sizeof(kGains[0]);

uint16_t *g_fb;      // what goes to the LCD (iomem)
uint16_t *g_cam_fb;  // camera frame converted to RGB565 (iomem)
uint8_t g_map[kW * kH]; // model output as 0..255, resampled to the LCD

uint64_t now_us() { return sysctl_get_time_us(); }

void uart_put(const void *p, size_t n) { uarths_send_data((const uint8_t *)p, n); }

// Frame dump for host-side checking: 8-byte magic, u32 kind, u32 length, payload.
void dump_chunk(uint32_t kind, const void *data, uint32_t len) {
  static const uint8_t magic[8] = {0xAA, 0x55, 'K', '2', '1', '0', 'D', 'M'};
  uart_put(magic, 8);
  uart_put(&kind, 4);
  uart_put(&len, 4);
  uart_put(data, len);
}

void draw_gray(const uint8_t *g, uint16_t *out, int x0, int x1) {
  for (int y = 0; y < kH; y++) {
    for (int x = x0; x < x1; x++) {
      uint8_t v = g[y * kW + x];
      out[y * kW + x] = (uint16_t)(((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3));
    }
  }
}

struct Model {
  interpreter interp;
  bool loaded = false;
  int in_h = 0, in_w = 0;
};

// Loads the kmodel from flash. Returns false (with a message) if there is none.
bool load_model(Model &m) {
  w25qxx_init(3, 0);
  w25qxx_enable_quad_mode();
  w25qxx_read_data(kModelFlashAddress, g_model_buffer, kModelBufferSize, W25QXX_QUAD_FAST);
  uint32_t id;
  memcpy(&id, g_model_buffer, 4);
  if (id != 'KMDL') {
    printf("no model at 0x%08x (id 0x%08x) -- camera only\n", kModelFlashAddress, id);
    return false;
  }
  auto r = m.interp.load_model({reinterpret_cast<const gsl::byte *>(g_model_buffer), kModelBufferSize});
  if (!r.is_ok()) {
    printf("model load failed: %s\n", r.unwrap_err().message().c_str());
    return false;
  }
  auto &shape = m.interp.input_shape(0);
  auto type = m.interp.input_desc(0).datatype;
  if (m.interp.inputs_size() != 1 || shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || type != dt_float32 ||
      shape[2] * shape[3] * 3 > kMaxInputFloats) {
    printf("unsupported model input (need one float32 1x3xHxW, H*W <= %d)\n", (int)(kMaxInputFloats / 3));
    return false;
  }
  m.in_h = (int)shape[2];
  m.in_w = (int)shape[3];
  printf("model: input 1x3x%dx%d float32, %zu output(s)\n", m.in_h, m.in_w, m.interp.outputs_size());
  return true;
}

// AI planes (R,G,B, kW*kH each) -> nearest-neighbour resized float CHW in [0,1].
void make_input(const Model &m, const uint8_t *planes) {
  const float k = 1.f / 255.f;
  for (int c = 0; c < 3; c++) {
    const uint8_t *src = planes + c * kW * kH;
    float *dst = g_input + c * m.in_h * m.in_w;
    for (int y = 0; y < m.in_h; y++) {
      const uint8_t *row = src + (y * kH / m.in_h) * kW;
      for (int x = 0; x < m.in_w; x++) dst[y * m.in_w + x] = row[x * kW / m.in_w] * k;
    }
  }
}

struct Output {
  const float *data = nullptr;
  size_t n = 0;
  runtime_shape_t shape;
};

// Runs the model on g_input. On success `out` points into the runtime's own
// output buffer, valid until the next run.
bool run_model(Model &m, Output &out) {
  auto &shape = m.interp.input_shape(0);
  size_t nbytes = (size_t)m.in_h * m.in_w * 3 * sizeof(float);
  auto t = hrt::create(dt_float32, shape, {reinterpret_cast<gsl::byte *>(g_input), nbytes}, false, hrt::pool_shared);
  if (!t.is_ok()) return false;
  auto &in = t.unwrap();
  if (!hrt::sync(in, hrt::sync_write_back).is_ok()) return false;
  if (!m.interp.input_tensor(0, in).is_ok()) return false;
  if (!m.interp.run().is_ok()) return false;
  auto ot = m.interp.output_tensor(0);
  if (!ot.is_ok()) return false;
  auto host = ot.unwrap().as_host();
  if (!host.is_ok()) return false;
  auto map = hrt::map(host.unwrap(), hrt::map_read);
  if (!map.is_ok()) return false;
  auto buf = map.unwrap().buffer();
  if (m.interp.output_desc(0).datatype != dt_float32) return false;
  out.data = reinterpret_cast<const float *>(buf.data());
  out.n = buf.size_bytes() / sizeof(float);
  out.shape = m.interp.output_shape(0);
  return true;
}

bool is_spatial_map(const Output &o) {
  return o.shape.size() == 4 && o.shape[1] == 1 && o.shape[2] > 1 && o.shape[3] > 1;
}

// Spatial output -> g_map (0..255, LCD-sized). Scales by a slowly-decaying
// running maximum so the picture keeps its contrast as the scene changes.
void map_to_gray(const Output &o) {
  static float run_max = 1e-3f;
  int oh = (int)o.shape[2], ow = (int)o.shape[3];
  float mx = 1e-6f;
  for (size_t i = 0; i < o.n; i++) mx = fmaxf(mx, o.data[i]);
  run_max = fmaxf(run_max * 0.95f, mx);
  float s = 255.f * kGains[g_gain_idx] / run_max;
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      float v = o.data[(y * oh / kH) * ow + (x * ow / kW)] * s;
      g_map[y * kW + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  }
}

// Top-5 of a logits/probability vector. Returns the softmax probability of the top-1.
float top5(const Output &o, int *idx) {
  float mx = o.data[0];
  for (size_t i = 1; i < o.n; i++) mx = fmaxf(mx, o.data[i]);
  float sum = 0;
  for (size_t i = 0; i < o.n; i++) sum += expf(o.data[i] - mx);
  bool used[4096] = {false};
  float best_p = 0;
  for (int k = 0; k < 5; k++) {
    int bi = -1;
    for (size_t i = 0; i < o.n; i++)
      if (!used[i] && (bi < 0 || o.data[i] > o.data[bi])) bi = (int)i;
    used[bi] = true;
    idx[k] = bi;
    if (k == 0) best_p = expf(o.data[bi] - mx) / sum;
  }
  return best_p;
}

void handle_uart(bool &want_dump) {
  uint8_t c;
  while (uarths_receive_data(&c, 1) == 1) {
    if (c == 'v') g_view = (g_view + 1) % kViewCount;
    else if (c == 'g') g_gain_idx = (g_gain_idx + 1) % kGainCount;
    else if (c == 'd') want_dump = true;
  }
}

} // namespace

int main() {
  sysctl_pll_set_freq(SYSCTL_PLL0, 800000000UL);
  sysctl_pll_set_freq(SYSCTL_PLL1, 400000000UL); // KPU clock source, see ../../runtime/src/main.cpp
  sysctl_pll_set_freq(SYSCTL_PLL2, 45158400UL);
  // TX works from the boot default; RX needs mapping. (Also mapping IO5 to
  // UARTHS_TX silenced the console on this board, so leave it alone.)
  fpioa_set_function(4, FUNC_UARTHS_RX);
  uarths_init();
  plic_init();
  sysctl_enable_irq();
  dmac_init();
  sysctl_clock_enable(SYSCTL_CLOCK_AI);

  printf("onnx-k210-flash camera demo (%s)\n", BOARD_NAME);
  if (board_power_init() != 0) printf("power init failed -- carrying on\n");
  board_buttons_init();

  g_fb = (uint16_t *)dma_alloc(kW * kH * 2);
  g_cam_fb = (uint16_t *)dma_alloc(kW * kH * 2);
  if (!g_fb || !g_cam_fb) {
    printf("out of iomem\n");
    while (1) {}
  }
  board_lcd_init();
  // Test pattern first, so a working LCD is visible even if the camera isn't.
  for (int y = 0; y < kH; y++)
    for (int x = 0; x < kW; x++) g_fb[y * kW + x] = (uint16_t)(((x * 31 / kW) << 11) | ((y * 63 / kH) << 5) | 0x0F);
  board_lcd_show(g_fb);

  uint16_t cam = cam_init();
  if (cam != 0x7742) {
    printf("camera: expected OV7740 (0x7742), got 0x%04x -- only OV7740 boards are supported\n", cam);
    while (1) {}
  }
  printf("camera: OV7740 ok\n");

  static Model model;
  model.loaded = load_model(model);
  if (!model.loaded) g_view = kCamera;

  bool prev_a = false, prev_b = false;
  uint64_t t_stat = now_us();
  int frames = 0;
  uint64_t infer_us = 0, cam_us = 0;

  while (1) {
    uint64_t t0 = now_us();
    if (cam_snapshot(500) != 0) {
      printf("camera: frame timeout\n");
      continue;
    }
    uint64_t t1 = now_us();
    cam_us += t1 - t0;

    const uint8_t *planes = cam_ai_planes();
    cam_ai_to_rgb565(planes, g_cam_fb);

    Output out;
    bool have_out = false;
    if (model.loaded) {
      make_input(model, planes);
      uint64_t t2 = now_us();
      have_out = run_model(model, out);
      infer_us += now_us() - t2;
      if (!have_out) printf("inference failed\n");
    }

    bool spatial = have_out && is_spatial_map(out);
    if (spatial) map_to_gray(out);

    switch (spatial ? g_view : kCamera) {
      case kCamera:
        memcpy(g_fb, g_cam_fb, kW * kH * 2);
        break;
      case kModel:
        draw_gray(g_map, g_fb, 0, kW);
        break;
      case kSplit:
        memcpy(g_fb, g_cam_fb, kW * kH * 2);
        draw_gray(g_map, g_fb, kW / 2, kW);
        break;
    }
    if (have_out && !spatial && out.n <= 4096) {
      static int frame_no = 0;
      int idx[5];
      float p = top5(out, idx);
      int bar = (int)(p * kW);
      for (int y = kH - 6; y < kH; y++)
        for (int x = 0; x < kW; x++) g_fb[y * kW + x] = x < bar ? 0x07E0 : 0x0000;
      if (++frame_no % 30 == 0)
        printf("top5: %d %d %d %d %d  (p=%.3f)\n", idx[0], idx[1], idx[2], idx[3], idx[4], (double)p);
    }
    board_lcd_show(g_fb);
    frames++;

    bool a = board_button_a(), b = board_button_b();
    if (a && !prev_a) g_view = (g_view + 1) % kViewCount;
    if (b && !prev_b) g_gain_idx = (g_gain_idx + 1) % kGainCount;
    prev_a = a;
    prev_b = b;

    bool want_dump = false;
    handle_uart(want_dump);
    if (want_dump) {
      // The DVP buffers stay put until the next snapshot, so this is one consistent frame.
      dump_chunk(1, planes, kW * kH * 3);
      if (model.loaded) {
        dump_chunk(2, g_input, (uint32_t)(model.in_h * model.in_w * 3 * sizeof(float)));
        if (have_out) {
          uint32_t hdr[5] = {(uint32_t)out.shape.size(), 0, 0, 0, 0};
          for (size_t i = 0; i < out.shape.size() && i < 4; i++) hdr[1 + i] = (uint32_t)out.shape[i];
          dump_chunk(3, hdr, sizeof(hdr));
          dump_chunk(4, out.data, (uint32_t)(out.n * sizeof(float)));
        }
      }
    }

    uint64_t now = now_us();
    if (now - t_stat >= 1000000) {
      printf("view=%d gain=%.1f  %d fps  cam wait %lu us  infer %lu us\n", g_view, (double)kGains[g_gain_idx], frames,
             (unsigned long)(frames ? cam_us / frames : 0), (unsigned long)(frames ? infer_us / frames : 0));
      frames = 0;
      cam_us = infer_us = 0;
      t_stat = now;
    }
  }
  return 0;
}
