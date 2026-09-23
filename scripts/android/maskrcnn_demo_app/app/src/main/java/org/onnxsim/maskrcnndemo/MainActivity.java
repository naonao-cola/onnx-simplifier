package org.onnxsim.maskrcnndemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.WindowManager;
import android.widget.FrameLayout;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Live Mask R-CNN on the phone. Two modes (intent extra "mode"): "camera" (default; back camera via
 * Camera2 into a small TextureView preview, latest frame wins) and "images" (loops over the JPEGs in
 * <files>/imgs). Intent extra "pipe" picks the pipeline file (default pipe_e_opt.txt;
 * pipe_e_opt_ctx.txt loads HTP sessions from EP-context models). Models live in
 * <app internal files>/models (copied in by deploy.sh via run-as).
 */
public class MainActivity extends Activity {
    private static final String TAG = "MaskRcnnDemo";
    private OverlayView overlay;
    private TextureView preview;
    private HandlerThread camThread;
    private Handler camHandler;
    private CameraDevice camera;
    private volatile boolean running = true;
    private Thread worker;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        FrameLayout root = new FrameLayout(this);
        overlay = new OverlayView(this);
        root.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        preview = new TextureView(this);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(320, 240, Gravity.BOTTOM | Gravity.END);
        lp.setMargins(0, 0, 16, 16);
        root.addView(preview, lp);
        setContentView(root);

        String mode = getIntent().getStringExtra("mode");
        final boolean cameraMode = mode == null || mode.equals("camera");
        final String pipe = getIntent().getStringExtra("pipe") != null ? getIntent().getStringExtra("pipe") : "pipe_e_opt.txt";
        if (!cameraMode) preview.setVisibility(android.view.View.GONE);
        if (cameraMode && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 1);

        worker = new Thread(() -> runLoop(cameraMode, pipe), "maskrcnn");
        worker.start();
    }

    private void runLoop(boolean cameraMode, String pipe) {
        File models = new File(getFilesDir(), "models");
        overlay.setStats("loading models (" + pipe + ")...");
        long t0 = System.nanoTime();
        String err = Engine.nativeInit(models.getAbsolutePath(), getApplicationInfo().nativeLibraryDir,
                new File(models, pipe).getAbsolutePath());
        double initMs = (System.nanoTime() - t0) / 1e6;
        if (err != null) {
            Log.e(TAG, "init failed: " + err);
            overlay.setStats("INIT FAILED:\n" + err);
            return;
        }
        Log.i(TAG, String.format(Locale.US, "init ok in %.0f ms (%s)", initMs, pipe));
        List<File> images = new ArrayList<>();
        if (!cameraMode) {
            File[] fs = new File(getFilesDir(), "imgs").listFiles();
            if (fs != null) for (File f : fs) if (f.getName().endsWith(".jpg")) images.add(f);
            Collections.sort(images);
            if (images.isEmpty()) {
                overlay.setStats("no images in " + new File(getFilesDir(), "imgs"));
                return;
            }
        } else {
            startCamera();
        }
        long[] done = new long[64];
        int nDone = 0;
        double latSum = 0;
        int latN = 0;
        double[] stage = new double[Engine.T_N];
        int frame = 0;
        while (running) {
            Bitmap src = cameraMode ? grabPreview() : BitmapFactory.decodeFile(images.get(frame % images.size()).getPath());
            if (src == null) {
                try { Thread.sleep(20); } catch (InterruptedException e) { return; }
                continue;
            }
            Bitmap in = fit(src);
            Engine.Result r = new Engine.Result();
            r.frame = in;
            if (!Engine.run(in, r)) {
                String e = Engine.nativeLastError();
                Log.e(TAG, "run failed: " + e);
                overlay.setStats("RUN FAILED:\n" + e);
                return;
            }
            long now = System.nanoTime();
            done[nDone++ % done.length] = now;
            int k = Math.min(nDone, done.length);
            double fps = k > 1 ? (k - 1) / ((now - done[(nDone - k) % done.length]) / 1e9) : 0;
            latSum += r.times[Engine.T_TOTAL];
            latN++;
            for (int i = 0; i < Engine.T_N; i++) stage[i] = 0.9 * stage[i] + 0.1 * r.times[i];
            if (latN == 1) for (int i = 0; i < Engine.T_N; i++) stage[i] = r.times[i];
            int shown = 0;
            for (int i = 0; i < r.n; i++) if (r.scores[i] >= OverlayView.SCORE_THRESH) shown++;
            String s = String.format(Locale.US,
                    "%s  FPS %.2f  latency %.1f ms (avg %.1f)\n"
                    + "pre %.1f  backbone %.1f  rpn %.1f  roialign %.1f  heads %.1f  cpu %.1f ms\n"
                    + "detections %d  (%s)",
                    cameraMode ? "camera" : "images", fps, r.times[Engine.T_TOTAL], stage[Engine.T_TOTAL],
                    stage[Engine.T_PRE], stage[Engine.T_BACKBONE], stage[Engine.T_RPN], stage[Engine.T_ROI],
                    stage[Engine.T_HEADS], stage[Engine.T_CPU], shown, pipe);
            overlay.update(r, s);
            if (frame % 10 == 0)
                Log.i(TAG, String.format(Locale.US, "frame %d fps %.2f lat %.1f %s", frame, fps, r.times[Engine.T_TOTAL],
                        Arrays.toString(r.times)));
            frame++;
        }
    }

    /** Scale to fit 1088x800 keeping aspect (eval_common.canvas does the same, top-left aligned). */
    private static Bitmap fit(Bitmap src) {
        float ratio = Math.min((float) Engine.IN_W / src.getWidth(), (float) Engine.IN_H / src.getHeight());
        int w = Math.max(1, (int) (src.getWidth() * ratio)), h = Math.max(1, (int) (src.getHeight() * ratio));
        Bitmap s = Bitmap.createScaledBitmap(src, w, h, true);
        return s.getConfig() == Bitmap.Config.ARGB_8888 ? s : s.copy(Bitmap.Config.ARGB_8888, false);
    }

    private Bitmap grabPreview() {
        if (!preview.isAvailable()) return null;
        AtomicReference<Bitmap> out = new AtomicReference<>();
        CountDownLatch l = new CountDownLatch(1);
        new Handler(Looper.getMainLooper()).post(() -> {
            if (preview.isAvailable()) out.set(preview.getBitmap(1088, 816));
            l.countDown();
        });
        try { l.await(); } catch (InterruptedException e) { return null; }
        return out.get();
    }

    private void startCamera() {
        camThread = new HandlerThread("cam");
        camThread.start();
        camHandler = new Handler(camThread.getLooper());
        new Handler(Looper.getMainLooper()).post(() -> {
            if (preview.isAvailable()) openCamera();
            else preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override public void onSurfaceTextureAvailable(SurfaceTexture s, int w, int h) { openCamera(); }
                @Override public void onSurfaceTextureSizeChanged(SurfaceTexture s, int w, int h) {}
                @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture s) { return true; }
                @Override public void onSurfaceTextureUpdated(SurfaceTexture s) {}
            });
        });
    }

    private void openCamera() {
        try {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                overlay.setStats("camera permission missing (adb shell pm grant org.onnxsim.maskrcnndemo android.permission.CAMERA)");
                return;
            }
            CameraManager cm = (CameraManager) getSystemService(CAMERA_SERVICE);
            String id = null;
            for (String c : cm.getCameraIdList()) {
                Integer f = cm.getCameraCharacteristics(c).get(CameraCharacteristics.LENS_FACING);
                if (f != null && f == CameraCharacteristics.LENS_FACING_BACK) { id = c; break; }
            }
            if (id == null) id = cm.getCameraIdList()[0];
            StreamConfigurationMap map = cm.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size best = new Size(1440, 1080);
            long bestD = Long.MAX_VALUE;
            for (Size s : map.getOutputSizes(SurfaceTexture.class)) {
                if (s.getWidth() * 3 != s.getHeight() * 4) continue;  // 4:3
                long d = Math.abs((long) s.getWidth() * s.getHeight() - 1440L * 1080);
                if (d < bestD) { bestD = d; best = s; }
            }
            final Size size = best;
            cm.openCamera(id, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice d) {
                    camera = d;
                    try {
                        SurfaceTexture st = preview.getSurfaceTexture();
                        st.setDefaultBufferSize(size.getWidth(), size.getHeight());
                        Surface surf = new Surface(st);
                        CaptureRequest.Builder rb = d.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        rb.addTarget(surf);
                        d.createCaptureSession(Collections.singletonList(surf), new CameraCaptureSession.StateCallback() {
                            @Override public void onConfigured(CameraCaptureSession s) {
                                try { s.setRepeatingRequest(rb.build(), null, camHandler); }
                                catch (Exception e) { Log.e(TAG, "repeat", e); }
                            }
                            @Override public void onConfigureFailed(CameraCaptureSession s) { Log.e(TAG, "configure failed"); }
                        }, camHandler);
                        Log.i(TAG, "camera " + size);
                    } catch (Exception e) { Log.e(TAG, "session", e); }
                }
                @Override public void onDisconnected(CameraDevice d) { d.close(); }
                @Override public void onError(CameraDevice d, int e) { Log.e(TAG, "camera error " + e); d.close(); }
            }, camHandler);
        } catch (Exception e) {
            Log.e(TAG, "openCamera", e);
        }
    }

    @Override
    protected void onDestroy() {
        running = false;
        if (camera != null) camera.close();
        if (camThread != null) camThread.quitSafely();
        super.onDestroy();
        // the native engine keeps DSP/HTP sessions; end the process so a relaunch starts clean
        android.os.Process.killProcess(android.os.Process.myPid());
    }
}
