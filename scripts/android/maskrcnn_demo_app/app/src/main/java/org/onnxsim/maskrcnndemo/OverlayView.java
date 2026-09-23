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
 * Draws the last processed frame, its detections (box, label + score, 28x28 instance mask scaled
 * into the box, alpha-blended in a per-class colour), and the FPS / latency panel.
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
    private final Bitmap maskBmp = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);
    private final int[] maskPx = new int[784];

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
            for (int i = 0; i < r.n; i++) {
                if (r.scores[i] < SCORE_THRESH) continue;
                int col = color(r.labels[i]);
                float x1 = ox + r.boxes[4 * i] * sc, y1 = oy + r.boxes[4 * i + 1] * sc;
                float x2 = ox + r.boxes[4 * i + 2] * sc, y2 = oy + r.boxes[4 * i + 3] * sc;
                int rgb = col & 0x00FFFFFF;
                for (int k = 0; k < 784; k++) maskPx[k] = r.masks[784 * i + k] > 0.5f ? (0x80000000 | rgb) : 0;
                maskBmp.setPixels(maskPx, 0, 28, 0, 0, 28, 28);
                cv.drawBitmap(maskBmp, null, new RectF(x1, y1, x2, y2), maskPaint);
                boxPaint.setColor(col);
                cv.drawRect(x1, y1, x2, y2, boxPaint);
                String t = Coco.name(r.labels[i]) + String.format(" %.2f", r.scores[i]);
                float tw = textPaint.measureText(t);
                labelBg.setColor((col & 0x00FFFFFF) | 0xC0000000);
                cv.drawRect(x1, y1 - 34, x1 + tw + 8, y1, labelBg);
                cv.drawText(t, x1 + 4, y1 - 8, textPaint);
            }
        }
        String[] lines = stats.split("\n");
        float lh = 36f, w = 0;
        for (String l : lines) w = Math.max(w, textPaint.measureText(l));
        cv.drawRect(8, 8, 24 + w, 16 + lh * lines.length, statsBg);
        for (int i = 0; i < lines.length; i++) cv.drawText(lines[i], 16, 8 + lh * (i + 1) - 6, textPaint);
    }
}
