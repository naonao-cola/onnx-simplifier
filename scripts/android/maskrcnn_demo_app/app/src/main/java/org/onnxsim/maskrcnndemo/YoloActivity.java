package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * YOLO mode (its own process, see the manifest): a deploy-pipeline YOLO model (yolo26n / yolo11n,
 * ../deploy) on the HTP, from the camera or the test images, with boxes, labels and an FPS panel.
 * Same camera path and extras as MainActivity, plus
 *   model   yolo26n (default) or yolo11n: <files>/models/<model>.onnx
 *   opts    YoloEngine options, e.g. "conf=0.25" (post=end2end for yolo26*, nms otherwise)
 * The model buttons switch YOLO models in place (the engine re-inits its HTP session).
 */
public class YoloActivity extends MainActivity {
    private static final String TAG = "YoloDemo";
    private volatile String wantModel;

    @Override
    int cameraMinWidth() {
        return YoloEngine.IN;
    }

    @Override
    boolean fastCamera() {
        return true;
    }

    @Override
    String activityKey() {
        return "yolo";
    }

    @Override
    void switchInPlace(String model) {
        wantModel = model;  // picked up by the loop
    }

    @Override
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        String model = getIntent().getStringExtra("model") != null ? getIntent().getStringExtra("model") : "yolo26n";
        String yopts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        wantModel = model;
        File models = new File(getFilesDir(), "models");
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
        String cur = null;
        double initMs = 0;
        long[] done = new long[32];
        int nDone = 0;
        double[] avg = new double[YoloEngine.T_N];
        long seq = 0;
        while (running) {
            String want = wantModel;
            if (!want.equals(cur)) {  // (re)load: first frame or a model button
                overlay.setStats("loading " + want + "...");
                long t0 = System.nanoTime();
                String err = YoloEngine.nativeInit(models.getAbsolutePath(), getApplicationInfo().nativeLibraryDir,
                        want, want.startsWith("yolo26") || yopts.contains("post=") ? yopts : "post=nms;" + yopts);
                initMs = (System.nanoTime() - t0) / 1e6;
                if (err != null) {
                    Log.e(TAG, "init failed: " + err);
                    overlay.setStats("INIT FAILED (" + want + "):\n" + err);
                    return;
                }
                Log.i(TAG, String.format(Locale.US, "init %s ok in %.0f ms", want, initMs));
                cur = want;
                nDone = 0;
            }
            Engine.Result r = new Engine.Result(YoloEngine.MAX_DET, false, YoloEngine.T_N, 0.25f);
            try {
                if (cameraMode) {
                    Image img = reader != null ? reader.acquireLatestImage() : null;
                    if (img == null) {
                        try { Thread.sleep(2); } catch (InterruptedException e) { return; }
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = YoloEngine.fitDims(img.getWidth(), img.getHeight(), rot);
                    r.frame = Bitmap.createBitmap(d[0], d[1], Bitmap.Config.ARGB_8888);
                    try {
                        YoloEngine.runYuv(img, rot, r.frame, r);
                    } finally {
                        img.close();
                    }
                } else {
                    r.frame = decodeFit(images.get((int) (seq % images.size())), YoloEngine.IN, YoloEngine.IN);
                    YoloEngine.run(r.frame, r);
                }
                r.id = seq++;
            } catch (RuntimeException e) {
                Log.e(TAG, "run failed: " + e.getMessage());
                overlay.setStats("RUN FAILED:\n" + e.getMessage());
                return;
            }
            if (nDone == 0 || nDone == 9) {
                long up = SystemClock.uptimeMillis() - android.os.Process.getStartUptimeMillis();
                Log.i(TAG, String.format(Locale.US, "startup: result %d shown %d ms after process start (init %.0f ms)",
                        nDone + 1, up, initMs));
            }
            long now = System.nanoTime();
            done[nDone++ % done.length] = now;
            int k = Math.min(nDone, done.length);
            double fps = k > 1 ? (k - 1) / ((now - done[(nDone - k) % done.length]) / 1e9) : 0;
            for (int i = 0; i < YoloEngine.T_N; i++) avg[i] = nDone == 1 ? r.times[i] : 0.9 * avg[i] + 0.1 * r.times[i];
            int shown = 0;
            for (int i = 0; i < r.n; i++) if (r.scores[i] >= r.thresh) shown++;
            String s = String.format(Locale.US,
                    "%s  %s\nFPS %.1f (end to end)  inference %.1f ms -> %.0f FPS possible\n"
                    + "pre %.1f  htp %.2f  post %.2f ms  detections %d%s",
                    cur, cameraMode ? "camera" : "images", fps, avg[YoloEngine.T_TOTAL],
                    1000.0 / Math.max(avg[YoloEngine.T_TOTAL], 1e-3), avg[YoloEngine.T_PRE], avg[YoloEngine.T_HTP],
                    avg[YoloEngine.T_POST], shown, cameraMode ? "  rot " + frameRotation() : "");
            overlay.update(r, s);
            if (nDone % 30 == 0)
                Log.i(TAG, String.format(Locale.US, "%s frame %d fps %.1f total %.2f pre %.2f htp %.2f post %.2f n %d",
                        cur, r.id, fps, avg[0], avg[1], avg[2], avg[3], shown));
        }
    }
}
