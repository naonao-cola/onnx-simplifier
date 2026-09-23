package org.onnxsim.maskrcnndemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ExifInterface;
import android.media.Image;
import android.media.ImageReader;
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
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

/**
 * Live Mask R-CNN on the phone. Intent extras:
 *   mode    "camera" (default; back camera) or "images" (loops over the JPEGs in <files>/imgs)
 *   pipe    pipeline file in <files>/models (default pipe_e_opt.txt; pipe_e_opt_ctx.txt loads the
 *           HTP sessions from EP-context models)
 *   opts    engine options, e.g. "quant=lut;merge=seg2,seg4;pipeline=seg1" (maskrcnn_engine.cpp);
 *           "" = the plain e2e pipeline
 *   overlap images mode: decode/scale the next JPEG on a capture thread while the current frame
 *           runs (camera frames are always converted natively, from a latest-frame ImageReader)
 *
 * Orientation: the activity follows the device (fullUser: all four rotations, honoring the user's rotation lock). Camera frames arrive in the sensor's
 * orientation; each frame is rotated by (sensorOrientation - displayRotation) so the model always
 * sees it gravity-up, letterboxed into its landscape 1088x800 input, in the same native pass that
 * converts YUV and quantizes. Boxes/masks come back in that upright frame's coordinates, and the
 * displayed image is that same upright frame, so the overlay needs no further mapping.
 */
public class MainActivity extends Activity {
    private static final String TAG = "MaskRcnnDemo";
    private OverlayView overlay;
    private TextureView preview;
    private HandlerThread camThread;
    private Handler camHandler;
    private CameraDevice camera;
    private ImageReader reader;
    private int sensorOrientation = 90;
    private Size camSize;
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
        root.addView(preview, new FrameLayout.LayoutParams(320, 240, Gravity.BOTTOM | Gravity.END));
        setContentView(root);

        String mode = getIntent().getStringExtra("mode");
        final boolean cameraMode = mode == null || mode.equals("camera");
        final String pipe = getIntent().getStringExtra("pipe") != null ? getIntent().getStringExtra("pipe") : "pipe_e_opt.txt";
        final String opts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        final boolean overlap = getIntent().getBooleanExtra("overlap", false);
        if (!cameraMode) preview.setVisibility(android.view.View.GONE);
        if (cameraMode && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 1);
        worker = new Thread(() -> runLoop(cameraMode, pipe, opts, overlap), "maskrcnn");
        worker.start();
    }

    private int displayRotationDegrees() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_90: return 90;
            case Surface.ROTATION_180: return 180;
            case Surface.ROTATION_270: return 270;
            default: return 0;
        }
    }

    /** Clockwise rotation that turns a back-camera sensor frame gravity-up for the current display rotation. */
    private int frameRotation() {
        return (sensorOrientation - displayRotationDegrees() + 360) % 360;
    }

    // ---- images mode ------------------------------------------------------------------------
    /** Decode a JPEG upright (EXIF orientation applied) and scale it to fit 1088x800 in one pass. */
    static Bitmap decodeFit(File f) {
        Bitmap src = BitmapFactory.decodeFile(f.getPath());
        if (src == null) return null;
        int deg = 0;
        try {
            switch (new ExifInterface(f.getPath()).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                case ExifInterface.ORIENTATION_ROTATE_90: deg = 90; break;
                case ExifInterface.ORIENTATION_ROTATE_180: deg = 180; break;
                case ExifInterface.ORIENTATION_ROTATE_270: deg = 270; break;
                default: break;
            }
        } catch (java.io.IOException e) {
            Log.w(TAG, "exif " + f + ": " + e);
        }
        int uw = deg % 180 == 0 ? src.getWidth() : src.getHeight();
        int uh = deg % 180 == 0 ? src.getHeight() : src.getWidth();
        float ratio = Math.min((float) Engine.IN_W / uw, (float) Engine.IN_H / uh);
        Matrix m = new Matrix();
        m.postRotate(deg);
        m.postScale(ratio, ratio);
        Bitmap out = Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
        return out.getConfig() == Bitmap.Config.ARGB_8888 ? out : out.copy(Bitmap.Config.ARGB_8888, false);
    }

    private final Object frameLock = new Object();
    private Bitmap pending;
    private long pendingId = -1;

    private void captureLoop(List<File> images) {
        long id = 0;
        while (running) {
            Bitmap in = decodeFit(images.get((int) (id % images.size())));
            synchronized (frameLock) {
                while (pending != null && running) {  // process every image, in order
                    try { frameLock.wait(); } catch (InterruptedException e) { return; }
                }
                pending = in;
                pendingId = id++;
                frameLock.notifyAll();
            }
        }
    }

    // ---- main loop --------------------------------------------------------------------------
    private void runLoop(boolean cameraMode, String pipe, String opts, boolean overlapCapture) {
        File models = new File(getFilesDir(), "models");
        overlay.setStats("loading models (" + pipe + ")...");
        long t0 = System.nanoTime();
        String err = Engine.nativeInit(models.getAbsolutePath(), getApplicationInfo().nativeLibraryDir,
                new File(models, pipe).getAbsolutePath(), opts);
        double initMs = (System.nanoTime() - t0) / 1e6;
        if (err != null) {
            Log.e(TAG, "init failed: " + err);
            overlay.setStats("INIT FAILED:\n" + err);
            return;
        }
        Log.i(TAG, String.format(Locale.US, "init ok in %.0f ms (%s, opts '%s', overlap %b)", initMs, pipe, opts,
                overlapCapture));
        final List<File> images = new ArrayList<>();
        if (!cameraMode) {
            File[] fs = new File(getFilesDir(), "imgs").listFiles();
            if (fs != null) for (File f : fs) if (f.getName().endsWith(".jpg")) images.add(f);
            Collections.sort(images);
            if (images.isEmpty()) {
                overlay.setStats("no images in " + new File(getFilesDir(), "imgs"));
                return;
            }
            if (overlapCapture) new Thread(() -> captureLoop(images), "capture").start();
        } else {
            startCamera();
        }
        long[] done = new long[32];
        int nDone = 0;
        double[] stage = new double[Engine.T_N];
        HashMap<Long, Bitmap> inFlight = new HashMap<>();
        long seq = 0;
        String label = (cameraMode ? "camera" : "images" + (overlapCapture ? " +overlap" : ""))
                + (opts.isEmpty() ? "" : " [" + opts + "]");
        while (running) {
            Engine.Result r = new Engine.Result();
            long got;
            try {
                if (cameraMode) {
                    Image img = reader != null ? reader.acquireLatestImage() : null;
                    if (img == null) {
                        try { Thread.sleep(3); } catch (InterruptedException e) { return; }
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = Engine.fitDims(img.getWidth(), img.getHeight(), rot);
                    Bitmap disp = Bitmap.createBitmap(d[0], d[1], Bitmap.Config.ARGB_8888);
                    long id = seq++;
                    inFlight.put(id, disp);
                    try {
                        got = Engine.submitYuv(img, rot, disp, id, r);
                    } finally {
                        img.close();
                    }
                } else {
                    Bitmap in;
                    long id;
                    if (overlapCapture) {
                        synchronized (frameLock) {
                            while (pending == null && running) {
                                try { frameLock.wait(); } catch (InterruptedException e) { return; }
                            }
                            if (!running) return;
                            in = pending;
                            id = pendingId;
                            pending = null;
                            frameLock.notifyAll();
                        }
                    } else {
                        in = decodeFit(images.get((int) (seq % images.size())));
                        id = seq++;
                    }
                    inFlight.put(id, in);
                    got = Engine.submit(in, id, r);
                }
            } catch (RuntimeException e) {
                Log.e(TAG, "run failed: " + e.getMessage());
                overlay.setStats("RUN FAILED:\n" + e.getMessage());
                return;
            }
            if (got < 0) continue;  // pipeline filling
            final long g = got;
            r.frame = inFlight.remove(got);
            inFlight.keySet().removeIf(x -> x < g);
            long now = System.nanoTime();
            done[nDone++ % done.length] = now;
            int k = Math.min(nDone, done.length);
            double fps = k > 1 ? (k - 1) / ((now - done[(nDone - k) % done.length]) / 1e9) : 0;
            for (int i = 0; i < Engine.T_N; i++) stage[i] = nDone == 1 ? r.times[i] : 0.9 * stage[i] + 0.1 * r.times[i];
            int shown = 0;
            for (int i = 0; i < r.n; i++) if (r.scores[i] >= OverlayView.SCORE_THRESH) shown++;
            String s = String.format(Locale.US,
                    "%s\nFPS %.2f  latency %.1f ms (avg %.1f)  stage A %.1f  B %.1f  wait %.1f\n"
                    + "pre %.1f  backbone %.1f  rpn %.1f  roialign %.1f  heads %.1f  cpu %.1f ms\n"
                    + "detections %d  %s%s",
                    label, fps, r.times[Engine.T_TOTAL], stage[Engine.T_TOTAL], stage[Engine.T_STAGE_A],
                    stage[Engine.T_STAGE_B], stage[Engine.T_WAIT], stage[Engine.T_PRE], stage[Engine.T_BACKBONE],
                    stage[Engine.T_RPN], stage[Engine.T_ROI], stage[Engine.T_HEADS], stage[Engine.T_CPU], shown, pipe,
                    cameraMode ? "  rot " + frameRotation() : "");
            overlay.update(r, s);
            if (nDone % 10 == 0)
                Log.i(TAG, String.format(Locale.US, "frame %d fps %.2f lat %.1f %s", got, fps, r.times[Engine.T_TOTAL],
                        Arrays.toString(r.times)));
        }
    }

    // ---- camera -----------------------------------------------------------------------------
    private void startCamera() {
        camThread = new HandlerThread("cam");
        camThread.start();
        camHandler = new Handler(camThread.getLooper());
        new Handler(Looper.getMainLooper()).post(() -> {
            if (preview.isAvailable()) openCamera();
            else preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override public void onSurfaceTextureAvailable(SurfaceTexture s, int w, int h) { openCamera(); }
                @Override public void onSurfaceTextureSizeChanged(SurfaceTexture s, int w, int h) { configureTransform(); }
                @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture s) { return true; }
                @Override public void onSurfaceTextureUpdated(SurfaceTexture s) {}
            });
        });
    }

    /**
     * Keeps the small live preview upright: resize the thumbnail to the upright aspect and set the
     * TextureView transform for the display rotation (the classic Camera2 sample recipe, generalized
     * to the sensor orientation). Only the thumbnail uses this; inference frames come from the
     * ImageReader and are rotated natively.
     */
    private void configureTransform() {
        if (camSize == null) return;
        int rot = frameRotation();
        int longSide = 320, shortSide = 240;
        int vw = rot % 180 == 0 ? longSide : shortSide, vh = rot % 180 == 0 ? shortSide : longSide;
        FrameLayout.LayoutParams lp = (FrameLayout.LayoutParams) preview.getLayoutParams();
        if (lp.width != vw || lp.height != vh) {
            lp.width = vw;
            lp.height = vh;
            preview.setLayoutParams(lp);
        }
        // the buffer is drawn stretched to the view; undo the stretch and rotate about the centre
        Matrix m = new Matrix();
        RectF view = new RectF(0, 0, vw, vh);
        float cx = view.centerX(), cy = view.centerY();
        int disp = displayRotationDegrees();
        if (disp == 90 || disp == 270) {
            RectF buf = new RectF(0, 0, camSize.getHeight(), camSize.getWidth());
            buf.offset(cx - buf.centerX(), cy - buf.centerY());
            m.setRectToRect(view, buf, Matrix.ScaleToFit.FILL);
            float s = Math.max((float) vh / camSize.getHeight(), (float) vw / camSize.getWidth());
            m.postScale(s, s, cx, cy);
            // SurfaceTexture already presents the buffer in the device's natural (portrait) orientation;
            // landscape display rotations turn it back by 90 * (rotation index - 2)
            m.postRotate(90 * (disp / 90 - 2), cx, cy);
        } else if (disp == 180) {
            m.postRotate(180, cx, cy);
        }
        preview.setTransform(m);
    }

    @Override
    public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        configureTransform();
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
            CameraCharacteristics ch = cm.getCameraCharacteristics(id);
            Integer so = ch.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation = so != null ? so : 90;
            StreamConfigurationMap map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            // smallest 4:3 YUV size that still covers the model input's long side (1088)
            Size best = null;
            for (Size s : map.getOutputSizes(ImageFormat.YUV_420_888)) {
                if (s.getWidth() * 3 != s.getHeight() * 4 || s.getWidth() < 1088) continue;
                if (best == null || s.getWidth() < best.getWidth()) best = s;
            }
            camSize = best != null ? best : new Size(1440, 1080);
            reader = ImageReader.newInstance(camSize.getWidth(), camSize.getHeight(), ImageFormat.YUV_420_888, 3);
            configureTransform();
            cm.openCamera(id, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice d) {
                    camera = d;
                    try {
                        SurfaceTexture st = preview.getSurfaceTexture();
                        st.setDefaultBufferSize(camSize.getWidth(), camSize.getHeight());
                        Surface surf = new Surface(st);
                        CaptureRequest.Builder rb = d.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        rb.addTarget(surf);
                        rb.addTarget(reader.getSurface());
                        d.createCaptureSession(Arrays.asList(surf, reader.getSurface()), new CameraCaptureSession.StateCallback() {
                            @Override public void onConfigured(CameraCaptureSession s) {
                                try { s.setRepeatingRequest(rb.build(), null, camHandler); }
                                catch (Exception e) { Log.e(TAG, "repeat", e); }
                            }
                            @Override public void onConfigureFailed(CameraCaptureSession s) { Log.e(TAG, "configure failed"); }
                        }, camHandler);
                        Log.i(TAG, "camera " + camSize + " sensorOrientation " + sensorOrientation);
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
        synchronized (frameLock) { frameLock.notifyAll(); }
        if (camera != null) camera.close();
        if (camThread != null) camThread.quitSafely();
        super.onDestroy();
        // the native engine keeps DSP/HTP sessions; end the process so a relaunch starts clean
        android.os.Process.killProcess(android.os.Process.myPid());
    }
}
