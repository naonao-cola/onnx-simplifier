/* Rotated-box BEV NMS and circle NMS for Fast-BEV / Fast-BEV++ decoding (host and phone).
 *
 * Boxes are (x, y, w, h, r): center, x/y extents before rotation, angle. Both upstreams' kernels
 * (Fast-BEV mmdet3d/ops/iou3d iou3d_kernel.cu, and mmcv's box_iou_rotated used by BEVDet's
 * nms_bev) rotate the corners by -r about the center; so does this. IoU = inter / (a + b - inter)
 * with the exact convex polygon intersection (Sutherland-Hodgman). Input boxes must already be
 * sorted by descending score (as both upstreams do before their kernel); a box is suppressed by an
 * earlier kept box when IoU > thr (circle NMS: squared center distance <= thr). Returns the number
 * of kept indices written to keep[] (indices into the sorted input), at most max_keep.
 */
#include <math.h>

typedef struct { float x, y; } pt;

static int corners(const float *b, pt *out) {
  float c = cosf(b[4]), s = sinf(b[4]);
  float hx = b[2] * 0.5f, hy = b[3] * 0.5f;
  const float dx[4] = {-hx, hx, hx, -hx}, dy[4] = {-hy, -hy, hy, hy};
  for (int k = 0; k < 4; k++) {  /* rotate by -r: [[c, s], [-s, c]] */
    out[k].x = b[0] + dx[k] * c + dy[k] * s;
    out[k].y = b[1] - dx[k] * s + dy[k] * c;
  }
  return 4;
}

static float cross(pt o, pt a, pt b) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); }

static float area(const pt *p, int n) {
  float s = 0.f;
  for (int i = 0; i < n; i++) {
    pt a = p[i], b = p[(i + 1) % n];
    s += a.x * b.y - a.y * b.x;
  }
  return fabsf(s) * 0.5f;
}

/* clip polygon p (n) by the half-plane left of edge a->b (for a CCW clip polygon) */
static int clip(const pt *p, int n, pt a, pt b, float orient, pt *out) {
  int m = 0;
  for (int i = 0; i < n; i++) {
    pt cur = p[i], prv = p[(i + n - 1) % n];
    float dc = orient * cross(a, b, cur), dp = orient * cross(a, b, prv);
    if (dc >= 0) {
      if (dp < 0) {
        float t = dp / (dp - dc);
        out[m++] = (pt){prv.x + t * (cur.x - prv.x), prv.y + t * (cur.y - prv.y)};
      }
      out[m++] = cur;
    } else if (dp >= 0) {
      float t = dp / (dp - dc);
      out[m++] = (pt){prv.x + t * (cur.x - prv.x), prv.y + t * (cur.y - prv.y)};
    }
  }
  return m;
}

float rbox_iou(const float *a, const float *b) {
  /* exact early-out: boxes whose circumscribed circles don't meet cannot overlap */
  const float dx = a[0] - b[0], dy = a[1] - b[1];
  const float r = 0.5f * (sqrtf(a[2] * a[2] + a[3] * a[3]) + sqrtf(b[2] * b[2] + b[3] * b[3]));
  if (dx * dx + dy * dy > r * r * 1.0001f) return 0.f;
  pt pa[4], pb[4], buf1[16], buf2[16];
  corners(a, pa);
  corners(b, pb);
  float orient = cross(pb[0], pb[1], pb[2]) >= 0 ? 1.f : -1.f;
  int n = 4;
  for (int k = 0; k < 4; k++) buf1[k] = pa[k];
  for (int e = 0; e < 4 && n > 0; e++) {
    n = clip(buf1, n, pb[e], pb[(e + 1) % 4], orient, buf2);
    for (int k = 0; k < n; k++) buf1[k] = buf2[k];
  }
  float inter = n > 2 ? area(buf1, n) : 0.f;
  float sa = a[2] * a[3], sb = b[2] * b[3];
  float u = sa + sb - inter;
  return inter / (u > 1e-8f ? u : 1e-8f);
}

int nms_rotated(const float *boxes, int n, float thr, int max_keep, int *keep) {
  int nk = 0;
  for (int i = 0; i < n && nk < max_keep; i++) {
    int ok = 1;
    for (int k = 0; k < nk; k++)
      if (rbox_iou(boxes + 5 * keep[k], boxes + 5 * i) > thr) { ok = 0; break; }
    if (ok) keep[nk++] = i;
  }
  return nk;
}

int nms_circle(const float *xy, int n, float thr, int max_keep, int *keep) {
  int nk = 0;
  for (int i = 0; i < n && nk < max_keep; i++) {
    int ok = 1;
    for (int k = 0; k < nk; k++) {
      float dx = xy[2 * keep[k]] - xy[2 * i], dy = xy[2 * keep[k] + 1] - xy[2 * i + 1];
      if (dx * dx + dy * dy <= thr) { ok = 0; break; }
    }
    if (ok) keep[nk++] = i;
  }
  return nk;
}
