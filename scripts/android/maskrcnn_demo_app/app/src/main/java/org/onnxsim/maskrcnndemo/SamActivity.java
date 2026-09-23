package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.widget.Button;
import android.widget.FrameLayout;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Segment Anything, tap to segment (its own process, see the manifest): EfficientViT-SAM-L0 from
 * ../vision_models/sam, encoder and decoder on the HTP (sam_engine.cpp).
 *   camera: live preview (no inference); a tap freezes that frame and runs the encoder once, then
 *           the decoder for the tap; further taps only run the decoder; "Live" unfreezes.
 *   images: each test image is encoded once; taps run the decoder; "Next" goes to the next image.
 * Extra "tap" ("fx,fy", fractions of the frame): an automatic tap after each encode (scripted runs).
 */
public class SamActivity extends MainActivity {
    private static final String TAG = "SamDemo";
    private final ConcurrentLinkedQueue<float[]> taps = new ConcurrentLinkedQueue<>();
    private volatile boolean frozen, next;

    @Override
    String activityKey() {
        return "sam";
    }

    @Override
    int cameraMinWidth() {
        return SamEngine.IN;
    }

    @Override
    boolean fastCamera() {
        return true;
    }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        Button live = new Button(this);
        live.setText(cameraMode ? "Live" : "Next image");
        live.setAllCaps(false);
        live.setAlpha(0.8f);
        live.setOnClickListener(v -> {
            if (cameraMode) frozen = false;
            else next = true;
        });
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        lp.topMargin = (int) (72 * getResources().getDisplayMetrics().density);  // below the model buttons
        ((FrameLayout) overlay.getParent()).addView(live, lp);
        overlay.setOnTouchListener((v, e) -> {
            if (e.getAction() != MotionEvent.ACTION_UP) return true;
            float[] f = overlay.toFrame(e.getX(), e.getY());
            if (f != null) taps.add(f);
            return true;
        });
    }

    @Override
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        String sopts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        String autoTap = getIntent().getStringExtra("tap");
        if (cameraMode) startCamera();
        overlay.setStats("loading EfficientViT-SAM-L0 (the first launch compiles the HTP graphs, ~20 s)...");
        long t0 = System.nanoTime();
        String err = SamEngine.nativeInit(new File(getFilesDir(), "models").getAbsolutePath(),
                getApplicationInfo().nativeLibraryDir, sopts);
        double initMs = (System.nanoTime() - t0) / 1e6;
        if (err != null) {
            Log.e(TAG, "init failed: " + err);
            overlay.setStats("INIT FAILED:\n" + err);
            return;
        }
        Log.i(TAG, String.format(Locale.US, "init ok in %.0f ms", initMs));
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
        float[] et = new float[2], dt = new float[1], iou = new float[4];
        byte[] mask = new byte[SamEngine.LR * SamEngine.LR];
        Engine.Result r = null;  // the frame on screen (no boxes)
        Bitmap maskBmp = null;
        float mx = -1, my = -1, encMs = 0, decMs = 0, decAvg = 0;
        int slot = 0, nDec = 0, img = 0;
        boolean needEncode = !cameraMode;
        long[] liveT = new long[16];
        int nLive = 0;
        try {
            while (running) {
                boolean changed = false;
                if (!cameraMode && next) {
                    next = false;
                    img = (img + 1) % images.size();
                    needEncode = true;
                }
                float[] tap = taps.poll();
                if (cameraMode && !frozen && tap != null) {  // a tap on the live preview: freeze + encode it
                    frozen = true;
                    needEncode = true;
                }
                if (cameraMode && (!frozen || needEncode)) {
                    Image im = reader != null ? reader.acquireLatestImage() : null;
                    if (im == null) {
                        if (tap != null) taps.add(tap);  // retry on the next frame
                        Thread.sleep(2);
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = SamEngine.fitDims(im.getWidth(), im.getHeight(), rot);
                    Engine.Result nr = new Engine.Result(1, false, 1, 1f);
                    nr.frame = Bitmap.createBitmap(d[0], d[1], Bitmap.Config.ARGB_8888);
                    try {
                        SamEngine.check(SamEngine.nativeYuv(im.getPlanes()[0].getBuffer(), im.getPlanes()[1].getBuffer(),
                                im.getPlanes()[2].getBuffer(), im.getPlanes()[0].getRowStride(),
                                im.getPlanes()[1].getRowStride(), im.getPlanes()[1].getPixelStride(), im.getWidth(),
                                im.getHeight(), rot, nr.frame, needEncode, et));
                    } finally {
                        im.close();
                    }
                    r = nr;
                    if (needEncode) {
                        needEncode = false;
                        changed = true;
                        encMs = et[0] + et[1];
                        maskBmp = null;
                        mx = -1;
                        Log.i(TAG, String.format(Locale.US, "encode: pre %.1f encoder %.1f ms", et[0], et[1]));
                    } else {
                        maskBmp = null;
                        long now = System.nanoTime();
                        liveT[nLive++ % liveT.length] = now;
                        int k = Math.min(nLive, liveT.length);
                        double fps = k > 1 ? (k - 1) / ((now - liveT[(nLive - k) % liveT.length]) / 1e9) : 0;
                        overlay.update(r, String.format(Locale.US,
                                "EfficientViT-SAM-L0  camera: live %.0f FPS (preview only)\ntap an object to freeze the frame and segment it", fps),
                                null, 0, 0, -1, -1);
                        continue;
                    }
                } else if (needEncode) {
                    needEncode = false;
                    Engine.Result nr = new Engine.Result(1, false, 1, 1f);
                    nr.frame = decodeFit(images.get(img), SamEngine.IN, SamEngine.IN);
                    SamEngine.check(SamEngine.nativeEncode(nr.frame, et));
                    r = nr;
                    changed = true;
                    encMs = et[0] + et[1];
                    maskBmp = null;
                    mx = -1;
                    Log.i(TAG, String.format(Locale.US, "encode %s: pre %.1f encoder %.1f ms", images.get(img).getName(), et[0], et[1]));
                }
                if (tap == null && autoTap != null && mx < 0 && r != null) {  // scripted tap after each encode
                    String[] a = autoTap.split(",");
                    tap = new float[] {Float.parseFloat(a[0]) * r.frame.getWidth(), Float.parseFloat(a[1]) * r.frame.getHeight()};
                }
                if (tap != null && r != null) {
                    slot = SamEngine.nativeDecode(tap[0], tap[1], mask, iou, dt);
                    if (slot < 0) throw new RuntimeException(SamEngine.nativeLastError());
                    decMs = dt[0];
                    decAvg = nDec++ == 0 ? decMs : 0.8f * decAvg + 0.2f * decMs;
                    int cw = (r.frame.getWidth() + 1) / 2, ch = (r.frame.getHeight() + 1) / 2;
                    int[] px = new int[cw * ch];
                    for (int y = 0; y < ch; y++)
                        for (int x = 0; x < cw; x++) px[y * cw + x] = mask[y * SamEngine.LR + x] != 0 ? 0x9000A0FF : 0;
                    maskBmp = Bitmap.createBitmap(px, cw, ch, Bitmap.Config.ARGB_8888);
                    mx = tap[0];
                    my = tap[1];
                    changed = true;
                    Log.i(TAG, String.format(Locale.US, "decode at (%.0f, %.0f): %.1f ms, slot %d, iou %.3f", mx, my, decMs,
                            slot, iou[slot]));
                }
                if (changed)
                    overlay.update(r, String.format(Locale.US,
                            "EfficientViT-SAM-L0  %s\nencoder %.1f ms (once per %s)   decoder %.1f ms per tap (avg %.1f)\n%s",
                            cameraMode ? "camera, frozen frame" : "images " + (img + 1) + "/" + images.size(), encMs,
                            cameraMode ? "frozen frame" : "image", decMs, decAvg,
                            mx < 0 ? "tap an object to segment it" : String.format(Locale.US, "mask slot %d, predicted IoU %.3f", slot, iou[slot])),
                            maskBmp, maskBmp != null ? 2f * maskBmp.getWidth() : 0, maskBmp != null ? 2f * maskBmp.getHeight() : 0, mx, my);
                if (taps.isEmpty()) Thread.sleep(5);
            }
        } catch (InterruptedException e) {
            // shutting down
        } catch (RuntimeException e) {
            Log.e(TAG, "run failed: " + e.getMessage());
            overlay.setStats("RUN FAILED:\n" + e.getMessage());
        }
    }
}
