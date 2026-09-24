package org.onnxsim.maskrcnndemo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

import java.util.Arrays;

/**
 * A colored point cloud (MCC's frame: x right, y up, z toward the viewer), drawn as z-buffered square
 * splats into a half-resolution bitmap, turned by dragging (yaw about y, pitch about x) and zoomed by
 * pinching. It starts from the photo's viewpoint. A thumbnail (the photo with its mask) sits in the
 * top-left corner, the stats panel bottom-left.
 */
final class PointCloudView extends View {
    private float[] xyz = new float[0];
    private int[] argb = new int[0];
    private int n;
    private float cx, cy, cz, radius = 1, spacing = 0.1f;
    private float yaw, pitch, zoom = 1;
    private float lastX, lastY, lastSpan;
    private Bitmap buf, thumb;
    private int[] px;
    private float[] zb;
    private String stats = "";
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint statsBg = new Paint();
    private final Paint thumbPaint = new Paint(Paint.FILTER_BITMAP_FLAG);

    PointCloudView(Context c) {
        super(c);
        textPaint.setColor(Color.WHITE);
        textPaint.setTypeface(android.graphics.Typeface.MONOSPACE);
        statsBg.setColor(0xB0000000);
        setBackgroundColor(0xFF202428);
    }

    /** New cloud (xyz 3 per point, colors ARGB) with grid spacing `spacing`; resets the view. */
    void setPoints(float[] xyz, int[] argb, float spacing, Bitmap thumb, String stats) {
        this.xyz = xyz;
        this.argb = argb;
        this.n = argb.length;
        this.spacing = spacing;
        this.thumb = thumb;
        this.stats = stats;
        double sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < n; i++) {
            sx += xyz[3 * i];
            sy += xyz[3 * i + 1];
            sz += xyz[3 * i + 2];
        }
        cx = n > 0 ? (float) (sx / n) : 0;
        cy = n > 0 ? (float) (sy / n) : 0;
        cz = n > 0 ? (float) (sz / n) : 0;
        float r2 = 1e-6f;
        for (int i = 0; i < n; i++) {
            float dx = xyz[3 * i] - cx, dy = xyz[3 * i + 1] - cy, dz = xyz[3 * i + 2] - cz;
            r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
        }
        radius = (float) Math.sqrt(r2);
        yaw = pitch = 0;
        zoom = 1;
        postInvalidate();
    }

    void setStats(String s) {
        stats = s;
        postInvalidate();
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                lastX = e.getX();
                lastY = e.getY();
                return true;
            case MotionEvent.ACTION_POINTER_DOWN:
                lastSpan = span(e);
                return true;
            case MotionEvent.ACTION_POINTER_UP:  // the remaining finger continues the drag from where it is
                int k = e.getActionIndex() == 0 ? 1 : 0;
                lastX = e.getX(k);
                lastY = e.getY(k);
                return true;
            case MotionEvent.ACTION_MOVE:
                if (e.getPointerCount() >= 2) {
                    float s = span(e);
                    if (lastSpan > 0) zoom = Math.max(0.3f, Math.min(6f, zoom * s / lastSpan));
                    lastSpan = s;
                } else {
                    float k2 = (float) Math.PI / Math.min(getWidth(), getHeight());
                    yaw += (e.getX() - lastX) * k2;
                    pitch = Math.max(-1.5f, Math.min(1.5f, pitch + (e.getY() - lastY) * k2));
                    lastX = e.getX();
                    lastY = e.getY();
                }
                invalidate();
                return true;
            default:
                return true;
        }
    }

    private static float span(MotionEvent e) {
        return (float) Math.hypot(e.getX(0) - e.getX(1), e.getY(0) - e.getY(1));
    }

    @Override
    protected void onDraw(Canvas cv) {
        int w = Math.max(1, getWidth() / 2), h = Math.max(1, getHeight() / 2);
        if (buf == null || buf.getWidth() != w || buf.getHeight() != h) {
            buf = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            px = new int[w * h];
            zb = new float[w * h];
        }
        Arrays.fill(px, 0);
        Arrays.fill(zb, Float.MAX_VALUE);
        // camera on +z at 3 radii from the centroid, looking at it; the cloud fills ~70% of the short side
        float dist = 3 * radius, f = zoom * 0.35f * Math.min(w, h) * dist / radius;
        float cyw = (float) Math.cos(yaw), syw = (float) Math.sin(yaw);
        float cp = (float) Math.cos(pitch), sp = (float) Math.sin(pitch);
        for (int i = 0; i < n; i++) {
            float x = xyz[3 * i] - cx, y = xyz[3 * i + 1] - cy, z = xyz[3 * i + 2] - cz;
            float x1 = cyw * x + syw * z, z1 = -syw * x + cyw * z;  // yaw about y
            float y2 = cp * y - sp * z1, z2 = sp * y + cp * z1;      // pitch about x
            float depth = dist - z2;
            if (depth <= 0.05f * radius) continue;
            float sx = w / 2f + f * x1 / depth, sy = h / 2f - f * y2 / depth;
            int half = Math.max(0, (int) (0.5f * f * spacing / depth));
            // darker with depth, for a depth cue
            float shade = Math.max(0.45f, Math.min(1f, 1.15f - 0.35f * (depth - dist + radius) / radius));
            int c = argb[i];
            int col = 0xFF000000 | ((int) (((c >> 16) & 255) * shade) << 16) | ((int) (((c >> 8) & 255) * shade) << 8)
                    | (int) ((c & 255) * shade);
            int x0 = (int) sx - half, y0 = (int) sy - half;
            for (int yy = Math.max(0, y0); yy <= Math.min(h - 1, y0 + 2 * half); yy++)
                for (int xx = Math.max(0, x0); xx <= Math.min(w - 1, x0 + 2 * half); xx++) {
                    int o = yy * w + xx;
                    if (depth < zb[o]) {
                        zb[o] = depth;
                        px[o] = col;
                    }
                }
        }
        buf.setPixels(px, 0, w, 0, 0, w, h);
        cv.drawBitmap(buf, null, new RectF(0, 0, getWidth(), getHeight()), null);
        if (thumb != null) {
            float tw = getWidth() * 0.28f, th = tw * thumb.getHeight() / thumb.getWidth();
            float top = 136 * getResources().getDisplayMetrics().density;  // below the model and mode buttons
            cv.drawBitmap(thumb, null, new RectF(8, top, 8 + tw, top + th), thumbPaint);
        }
        textPaint.setTextSize(Math.min(30f, getWidth() / 45f));
        String[] lines = stats.split("\n");
        float lh = textPaint.getTextSize() * 1.2f, tw = 0;
        for (String l : lines) tw = Math.max(tw, textPaint.measureText(l));
        float top = getHeight() - 16 - lh * lines.length;
        cv.drawRect(8, top - 8, 24 + tw, getHeight() - 8, statsBg);
        for (int i = 0; i < lines.length; i++) cv.drawText(lines[i], 16, top + lh * (i + 1) - 6, textPaint);
    }
}
