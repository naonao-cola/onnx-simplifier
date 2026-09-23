package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Super resolution (its own process, see the manifest): an x4 SR model from ../vision_models/superres
 * on the HTP (sr_engine.cpp). Each frame: the camera's 16:9 crop at 1920x1080 (1080x1920 upright in
 * portrait) is the "original"; its 4x4 average (480x270, 1/16 the pixels -- what a game would render)
 * goes through the model; the screen shows the SR output right of a draggable divider and a
 * reference left of it: bicubic of the same low-res input (CPU), the original, or the low-res pixels.
 *   buttons  model (XLSR int8 default, QuickSRNet-M int8, XLSR fp16, Real-ESRGAN anime int8),
 *            reference (bicubic / original / low-res)
 *   extras   model <stem> (sr_xlsr_int8), mode images (the test images, center-cropped to 16:9),
 *            ref bicubic|original|lowres, split 0..1
 */
public class SrActivity extends MainActivity {
    private static final String TAG = "SrDemo";
    static final String[][] SR_MODELS = {{"XLSR int8", "sr_xlsr_int8"}, {"QuickSRNet-M int8", "sr_qsrm_int8"},
            {"XLSR fp16", "sr_xlsr_fp16"}, {"Real-ESRGAN int8", "sr_esrgan_int8"}};
    static final String[] REFS = {"bicubic", "original", "low-res"};
    private volatile int modelIdx, refIdx;
    private volatile float split = 0.5f;

    @Override
    String activityKey() {
        return "sr";
    }

    @Override
    int cameraMinWidth() {
        return 1920;  // a 1920x1080 crop of the 4:3 frame (1920x1440)
    }

    @Override
    boolean fastCamera() {
        return true;
    }

    private Button button(String text, android.view.View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 13);
        b.setAlpha(0.8f);
        b.setOnClickListener(l);
        return b;
    }

    @Override
    protected void onCreate(Bundle b) {
        String m = getIntent().getStringExtra("model");
        for (int i = 0; i < SR_MODELS.length; i++) if (SR_MODELS[i][1].equals(m)) modelIdx = i;
        String r = getIntent().getStringExtra("ref");
        for (int i = 0; i < REFS.length; i++) if (REFS[i].replace("-", "").equals(r)) refIdx = i;
        split = getIntent().getFloatExtra("split", 0.5f);
        super.onCreate(b);
        LinearLayout bar = new LinearLayout(this);
        Button mb = button(SR_MODELS[modelIdx][0], null), rb = button("left: " + REFS[refIdx], null);
        mb.setOnClickListener(v -> {
            modelIdx = (modelIdx + 1) % SR_MODELS.length;
            mb.setText(SR_MODELS[modelIdx][0]);
        });
        rb.setOnClickListener(v -> {
            refIdx = (refIdx + 1) % REFS.length;
            rb.setText("left: " + REFS[refIdx]);
        });
        bar.addView(mb);
        bar.addView(rb);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        lp.topMargin = (int) (72 * getResources().getDisplayMetrics().density);  // below the model buttons
        ((FrameLayout) overlay.getParent()).addView(bar, lp);
        overlay.setOnTouchListener((v, e) -> {  // drag anywhere to move the divider
            if (e.getAction() == MotionEvent.ACTION_DOWN || e.getAction() == MotionEvent.ACTION_MOVE) {
                float f = overlay.toFrameFrac(e.getX());
                if (f >= 0) split = f;
            }
            return true;
        });
    }

    /** A test image, EXIF-upright, center-cropped to 16:9 (9:16 if portrait) and scaled to the HR size. */
    private static Bitmap hrImage(File f) {
        Bitmap src = decodeFit(f, 4096, 4096);
        boolean port = src.getHeight() > src.getWidth();
        int hw = port ? 1080 : 1920, hh = port ? 1920 : 1080;
        int cw = src.getWidth(), ch = (int) ((long) cw * hh / hw);
        if (ch > src.getHeight()) {
            ch = src.getHeight();
            cw = (int) ((long) ch * hw / hh);
        }
        Bitmap out = Bitmap.createBitmap(hw, hh, Bitmap.Config.ARGB_8888);
        int cx = (src.getWidth() - cw) / 2, cy = (src.getHeight() - ch) / 2;
        new Canvas(out).drawBitmap(src, new Rect(cx, cy, cx + cw, cy + ch), new Rect(0, 0, hw, hh),
                new Paint(Paint.FILTER_BITMAP_FLAG));
        return out;
    }

    /** Three rotating sets of bitmaps (the view may still be drawing the previous frame's). */
    private final Bitmap[][] sets = new Bitmap[3][];

    private Bitmap[] set(long seq, int hw, int hh) {
        int k = (int) (seq % sets.length);
        Bitmap[] s = sets[k];
        if (s == null || s[0].getWidth() != hw || s[0].getHeight() != hh) {
            s = new Bitmap[] {Bitmap.createBitmap(hw, hh, Bitmap.Config.ARGB_8888),   // original
                    Bitmap.createBitmap(hw / 4, hh / 4, Bitmap.Config.ARGB_8888),       // low-res
                    Bitmap.createBitmap(hw, hh, Bitmap.Config.ARGB_8888),               // SR
                    Bitmap.createBitmap(hw, hh, Bitmap.Config.ARGB_8888)};              // bicubic
            sets[k] = s;
        }
        return s;
    }

    @Override
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        String sopts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        if (cameraMode) startCamera();
        List<File> images = new ArrayList<>();
        if (!cameraMode) {
            File[] fs = new File(getFilesDir(), "imgs").listFiles();
            if (fs != null) for (File f : fs) if (f.getName().endsWith(".jpg")) images.add(f);
            Collections.sort(images);
            if (images.isEmpty()) {
                overlay.setStats("no images in " + new File(getFilesDir(), "imgs"));
                return;
            }
        }
        int cur = -1;
        long[] done = new long[32];
        int nDone = 0;
        double[] avg = new double[SrEngine.T_N];
        long seq = 0;
        Bitmap hrImg = null;
        int hrFor = -1;
        while (running) {
            int want = modelIdx;
            if (want != cur) {
                overlay.setStats("loading " + SR_MODELS[want][0] + " (the first launch compiles its HTP graph)...");
                long t0 = System.nanoTime();
                String err = SrEngine.nativeInit(new File(getFilesDir(), "models").getAbsolutePath(),
                        getApplicationInfo().nativeLibraryDir, SR_MODELS[want][1], sopts);
                if (err != null) {
                    Log.e(TAG, "init failed: " + err);
                    overlay.setStats("INIT FAILED (" + SR_MODELS[want][1] + "):\n" + err);
                    return;
                }
                Log.i(TAG, String.format(Locale.US, "init %s ok in %.0f ms", SR_MODELS[want][1],
                        (System.nanoTime() - t0) / 1e6));
                cur = want;
                nDone = 0;
            }
            int ref = refIdx;
            float[] t = new float[SrEngine.T_N];
            Bitmap[] s;
            try {
                if (cameraMode) {
                    Image img = reader != null ? reader.acquireLatestImage() : null;
                    if (img == null) {
                        try { Thread.sleep(2); } catch (InterruptedException e) { return; }
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = SrEngine.hrDims(img.getWidth(), img.getHeight(), rot);
                    s = set(seq, d[0], d[1]);
                    try {
                        SrEngine.runYuv(img, rot, ref == 1 ? s[0] : null, s[1], s[2], ref == 0 ? s[3] : null, t);
                    } finally {
                        img.close();
                    }
                } else {
                    int i = (int) ((seq / 30) % images.size());  // a new test image every 30 frames
                    if (i != hrFor) {
                        hrImg = hrImage(images.get(i));
                        hrFor = i;
                    }
                    s = set(seq, hrImg.getWidth(), hrImg.getHeight());
                    SrEngine.run(hrImg, s[1], s[2], ref == 0 ? s[3] : null, t);
                    s[0] = hrImg;
                }
            } catch (RuntimeException e) {
                Log.e(TAG, "run failed: " + e.getMessage());
                overlay.setStats("RUN FAILED:\n" + e.getMessage());
                return;
            }
            seq++;
            long now = System.nanoTime();
            done[nDone++ % done.length] = now;
            int k = Math.min(nDone, done.length);
            double fps = k > 1 ? (k - 1) / ((now - done[(nDone - k) % done.length]) / 1e9) : 0;
            for (int i = 0; i < avg.length; i++) avg[i] = nDone == 1 ? t[i] : 0.9 * avg[i] + 0.1 * t[i];
            int hw = s[2].getWidth(), hh = s[2].getHeight();
            String st = String.format(Locale.US,
                    "%s  %s  %dx%d -> %dx%d (x4)\nFPS %.1f (end to end)  SR on the HTP %.2f ms\n"
                    + "frame->low-res %.1f  SR->bitmap %.1f%s ms\ndrag to move the divider",
                    SR_MODELS[cur][0], cameraMode ? "camera" : "images", hw / 4, hh / 4, hw, hh, fps,
                    avg[SrEngine.T_HTP], avg[SrEngine.T_PRE], avg[SrEngine.T_POST],
                    ref == 0 ? String.format(Locale.US, "  bicubic (CPU) %.1f", avg[SrEngine.T_BICUBIC]) : "");
            Engine.Result r = new Engine.Result(0, false, 1, 1f);
            r.frame = s[2];
            r.id = seq;
            Bitmap left = ref == 0 ? s[3] : ref == 1 ? s[0] : s[1];
            overlay.updateSplit(r, st, left, ref == 2, split, REFS[ref], SR_MODELS[cur][0]);
            if (nDone % 30 == 0)
                Log.i(TAG, String.format(Locale.US, "%s frame %d fps %.1f htp %.2f pre %.1f post %.1f bicubic %.1f",
                        SR_MODELS[cur][1], seq, fps, avg[SrEngine.T_HTP], avg[SrEngine.T_PRE], avg[SrEngine.T_POST],
                        avg[SrEngine.T_BICUBIC]));
        }
    }
}
