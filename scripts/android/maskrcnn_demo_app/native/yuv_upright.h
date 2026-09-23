// Camera2 YUV_420_888 -> gravity-up RGB, scaled (nearest neighbour) to fw x fh, for the demo's
// single-session engines (yolo_engine.cpp, sam_engine.cpp). rot (0/90/180/270) is the clockwise
// rotation that makes the sensor frame upright. YUV -> RGB is JFIF full-range BT.601 (what camera
// YUV_420_888 is), fixed point, the same as maskrcnn_engine.cpp's quant_yuv. put(oy, ox, r, g, b)
// is called once per output pixel, rows split over `threads` threads.
#pragma once
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace demo {

struct YuvPlanes {
  const uint8_t *y, *u, *v;
  int ys, uvs, uvps, w, h;  // row strides, uv pixel stride, sensor frame size
};

template <class F>
void par_rows(long n, int threads, F f) {
  std::vector<std::thread> th;
  const long step = (n + threads - 1) / threads;
  for (int t = 1; t < threads; ++t) {
    long a = t * step, b = std::min(n, a + step);
    if (a < b) th.emplace_back([=] { f(a, b); });
  }
  f(0, std::min(n, step));
  for (auto& x : th) x.join();
}

// Upright size of the sensor frame.
inline void upright_dims(const YuvPlanes& P, int rot, int* rw, int* rh) {
  *rw = (rot % 180) ? P.h : P.w;
  *rh = (rot % 180) ? P.w : P.h;
}

template <class Put>
void yuv_upright(const YuvPlanes& P, int rot, int fw, int fh, int threads, Put put) {
  int RW, RH;
  upright_dims(P, rot, &RW, &RH);
  std::vector<int> rxs(fw), rys(fh);
  for (int x = 0; x < fw; ++x) rxs[x] = std::min(RW - 1, (int)(((long)x * RW + RW / 2) / fw));
  for (int y = 0; y < fh; ++y) rys[y] = std::min(RH - 1, (int)(((long)y * RH + RH / 2) / fh));
  par_rows(fh, threads, [&](long a, long b) {
    for (long oy = a; oy < b; ++oy) {
      const int ry = rys[oy];
      for (int ox = 0; ox < fw; ++ox) {
        const int rx = rxs[ox];
        int sx, sy;
        switch (rot) {
          case 90: sx = ry; sy = P.h - 1 - rx; break;
          case 180: sx = P.w - 1 - rx; sy = P.h - 1 - ry; break;
          case 270: sx = P.w - 1 - ry; sy = rx; break;
          default: sx = rx; sy = ry;
        }
        const int yy = P.y[sy * P.ys + sx];
        const int ci = (sy >> 1) * P.uvs + (sx >> 1) * P.uvps;
        const int uu = P.u[ci] - 128, vv = P.v[ci] - 128;
        // x1024 fixed point: 1.402, 0.344136, 0.714136, 1.772
        int r = yy + ((1436 * vv + 512) >> 10);
        int g = yy - ((352 * uu + 731 * vv + 512) >> 10);
        int bb = yy + ((1815 * uu + 512) >> 10);
        r = r < 0 ? 0 : (r > 255 ? 255 : r);
        g = g < 0 ? 0 : (g > 255 ? 255 : g);
        bb = bb < 0 ? 0 : (bb > 255 ? 255 : bb);
        put((int)oy, ox, (uint8_t)r, (uint8_t)g, (uint8_t)bb);
      }
    }
  });
}

}  // namespace demo
