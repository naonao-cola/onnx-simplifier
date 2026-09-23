package org.onnxsim.maskrcnndemo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.view.View;

/**
 * Draws the last processed frame, its detections (box, label + score, and for Mask R-CNN the 28x28
 * instance mask scaled into the box, alpha-blended in a per-class colour), and the FPS / latency
 * panel.
 */
final class OverlayView extends View {
    static final float SCORE_THRESH = 0.5f;
    private Engine.Result result;
    private String stats = "starting...";
    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBg = new Paint();
    private final Paint statsBg = new Paint();
    private final Paint maskPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final int[] maskPx = new int[784];
    // SAM mode: a mask bitmap drawn over the frame rect (0, 0, extraW, extraH) in frame pixels, and
    // the tap that produced it
    private Bitmap extra;
    private float extraW, extraH, markX = -1, markY = -1;
    private final Paint markPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    // the last frame's placement on screen: view = o + frame * sc
    private volatile float lastOx, lastOy, lastSc = 0;

    OverlayView(Context c) {
        super(c);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(30f);
        textPaint.setTypeface(android.graphics.Typeface.MONOSPACE);
        statsBg.setColor(0xB0000000);
        setBackgroundColor(Color.BLACK);
    }

    void update(Engine.Result r, String s) {
        result = r;
        stats = s;
        postInvalidate();
    }

    /** Frame + panel + an overlay bitmap covering frame pixels (0, 0, w, h) and a tap marker (x, y). */
    void update(Engine.Result r, String s, Bitmap overlayBmp, float w, float h, float x, float y) {
        extra = overlayBmp;
        extraW = w;
        extraH = h;
        markX = x;
        markY = y;
        update(r, s);
    }

    /** View coordinates -> the displayed frame's pixels, null if no frame yet or outside it. */
    float[] toFrame(float vx, float vy) {
        Engine.Result r = result;
        if (r == null || r.frame == null || lastSc == 0) return null;
        float fx = (vx - lastOx) / lastSc, fy = (vy - lastOy) / lastSc;
        if (fx < 0 || fy < 0 || fx >= r.frame.getWidth() || fy >= r.frame.getHeight()) return null;
        return new float[] {fx, fy};
    }

    void setStats(String s) {
        stats = s;
        postInvalidate();
    }

    static int color(int label) {
        float h = (label * 47) % 360;
        return Color.HSVToColor(new float[] {h, 0.85f, 1f});
    }

    @Override
    protected void onDraw(Canvas cv) {
        Engine.Result r = result;
        if (r != null && r.frame != null) {
            float fw = r.frame.getWidth(), fh = r.frame.getHeight();
            float sc = Math.min(getWidth() / fw, getHeight() / fh);
            float ox = (getWidth() - fw * sc) / 2, oy = (getHeight() - fh * sc) / 2;
            cv.drawBitmap(r.frame, null, new RectF(ox, oy, ox + fw * sc, oy + fh * sc), null);
            lastOx = ox;
            lastOy = oy;
            lastSc = sc;
            if (extra != null) cv.drawBitmap(extra, null, new RectF(ox, oy, ox + extraW * sc, oy + extraH * sc), maskPaint);
            if (markX >= 0) {
                markPaint.setColor(Color.YELLOW);
                cv.drawCircle(ox + markX * sc, oy + markY * sc, 14f, markPaint);
            }
            for (int i = 0; i < r.n; i++) {
                if (r.scores[i] < r.thresh) continue;
                int col = color(r.labels[i]);
                float x1 = ox + r.boxes[4 * i] * sc, y1 = oy + r.boxes[4 * i + 1] * sc;
                float x2 = ox + r.boxes[4 * i + 2] * sc, y2 = oy + r.boxes[4 * i + 3] * sc;
                if (r.masks.length > 0) {
                    int rgb = col & 0x00FFFFFF;
                    for (int k = 0; k < 784; k++) maskPx[k] = r.masks[784 * i + k] > 0.5f ? (0x80000000 | rgb) : 0;
                    // A fresh bitmap per detection: a hardware canvas records draws and uploads bitmap
                    // contents at render time, so reusing one mutable bitmap would paint every box with
                    // the last detection's mask.
                    Bitmap m = Bitmap.createBitmap(maskPx, 28, 28, Bitmap.Config.ARGB_8888);
                    cv.drawBitmap(m, null, new RectF(x1, y1, x2, y2), maskPaint);
                }
                boxPaint.setColor(col);
                cv.drawRect(x1, y1, x2, y2, boxPaint);
                String t = Coco.name(r.labels[i]) + String.format(" %.2f", r.scores[i]);
                float tw = textPaint.measureText(t);
                labelBg.setColor((col & 0x00FFFFFF) | 0xC0000000);
                cv.drawRect(x1, y1 - 34, x1 + tw + 8, y1, labelBg);
                cv.drawText(t, x1 + 4, y1 - 8, textPaint);
            }
        }
        textPaint.setTextSize(Math.min(30f, getWidth() / 45f));  // fits the stats lines in portrait too
        String[] lines = stats.split("\n");
        float lh = textPaint.getTextSize() * 1.2f, w = 0;
        for (String l : lines) w = Math.max(w, textPaint.measureText(l));
        // bottom-left, so it doesn't cover labels of boxes near the top edge
        float top = getHeight() - 16 - lh * lines.length;
        cv.drawRect(8, top - 8, 24 + w, getHeight() - 8, statsBg);
        for (int i = 0; i < lines.length; i++) cv.drawText(lines[i], 16, top + lh * (i + 1) - 6, textPaint);
    }
}
