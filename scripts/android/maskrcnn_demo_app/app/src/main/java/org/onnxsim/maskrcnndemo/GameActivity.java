package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

import java.io.File;
import java.util.Locale;

/**
 * Game upscaling (its own process, see the manifest): replays a short stretch of Arm's Bistro test
 * sequence (960x540 renders with depth, motion and camera; deploy.sh GAME=...) through NSS super
 * sampling to 1920x1080 and, with NFRU on, one generated frame between each two (game_engine.cpp: the
 * networks on the HTP, everything else OpenCL on the Adreno). The screen shows NSS (or NSS + NFRU)
 * right of a draggable divider and the native 540p render left of it.
 *   buttons  NFRU on/off
 *   extras   nfru true|false, split 0..1, opts (htp_performance_mode=burst)
 */
public class GameActivity extends MainActivity {
    private static final String TAG = "GameDemo";
    private volatile boolean nfruOn;
    private volatile float split = 0.5f;

    @Override
    String activityKey() {
        return "game";
    }

    @Override
    protected void onCreate(Bundle b) {
        nfruOn = getIntent().getBooleanExtra("nfru", false);
        split = getIntent().getFloatExtra("split", 0.5f);
        getIntent().putExtra("mode", "images");  // no camera
        super.onCreate(b);
        LinearLayout bar = new LinearLayout(this);
        Button nb = new Button(this);
        nb.setAllCaps(false);
        nb.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 13);
        nb.setAlpha(0.8f);
        nb.setText(nfruOn ? "NFRU: on" : "NFRU: off");
        nb.setOnClickListener(v -> {
            nfruOn = !nfruOn;
            nb.setText(nfruOn ? "NFRU: on" : "NFRU: off");
        });
        bar.addView(nb);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        lp.topMargin = (int) (72 * getResources().getDisplayMetrics().density);
        ((FrameLayout) overlay.getParent()).addView(bar, lp);
        overlay.setOnTouchListener((v, e) -> {
            if (e.getAction() == MotionEvent.ACTION_DOWN || e.getAction() == MotionEvent.ACTION_MOVE) {
                float f = overlay.toFrameFrac(e.getX());
                if (f >= 0) split = f;
            }
            return true;
        });
    }

    /** Three rotating sets {low-res, NSS, generated}: the view may still be drawing the previous ones. */
    private final Bitmap[][] sets = new Bitmap[3][];

    private Bitmap[] set(long seq) {
        int k = (int) (seq % sets.length);
        if (sets[k] == null)
            sets[k] = new Bitmap[] {
                Bitmap.createBitmap(GameEngine.LR_W, GameEngine.LR_H, Bitmap.Config.ARGB_8888),
                Bitmap.createBitmap(GameEngine.HR_W, GameEngine.HR_H, Bitmap.Config.ARGB_8888),
                Bitmap.createBitmap(GameEngine.HR_W, GameEngine.HR_H, Bitmap.Config.ARGB_8888)};
        return sets[k];
    }

    private void show(Bitmap right, Bitmap left, String st, String rightLabel, long id) {
        Engine.Result r = new Engine.Result(0, false, 1, 1f);
        r.frame = right;
        r.id = id;
        overlay.updateSplit(r, st, left, true, split, "native 960x540", rightLabel);
    }

    @Override
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        String gopts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        overlay.setStats("loading NSS + NFRU (the first launch compiles their HTP graphs)...");
        long t0 = System.nanoTime();
        String err = GameEngine.nativeInit(new File(getFilesDir(), "models").getAbsolutePath(),
                getApplicationInfo().nativeLibraryDir, gopts);
        if (err != null) {
            Log.e(TAG, "init failed: " + err);
            overlay.setStats("INIT FAILED:\n" + err);
            return;
        }
        Log.i(TAG, String.format(Locale.US, "init ok in %.0f ms", (System.nanoTime() - t0) / 1e6));
        int frames = GameEngine.nativeFrames();
        float[] avg = new float[GameEngine.T_N];
        long seq = 0, shown = 0;
        long[] disp = new long[32];
        int nDisp = 0;
        while (running) {
            int t = (int) (seq % frames);
            boolean nfru = nfruOn;
            float[] tm = new float[GameEngine.T_N];
            Bitmap[] s = set(seq);
            boolean gen;
            long c0 = System.nanoTime();
            try {
                gen = GameEngine.step(t, nfru, s[0], s[1], s[2], tm);
            } catch (RuntimeException e) {
                Log.e(TAG, "run failed: " + e.getMessage());
                overlay.setStats("RUN FAILED:\n" + e.getMessage());
                return;
            }
            double stepMs = (System.nanoTime() - c0) / 1e6;
            seq++;
            for (int i = 0; i < avg.length; i++) avg[i] = seq == 1 ? tm[i] : 0.9f * avg[i] + 0.1f * tm[i];
            int k = Math.min(nDisp, disp.length);
            long now = System.nanoTime();
            double fps = k > 1 ? (k - 1) / ((now - disp[(nDisp - k) % disp.length]) / 1e9) : 0;
            String st = String.format(Locale.US,
                    "Bistro frame %d/%d  960x540 -> 1920x1080  NFRU %s\ndisplayed FPS %.1f\n"
                    + "NSS %.1f ms (CNN on the HTP %.1f)%s\ndrag to move the divider",
                    t, frames, nfru ? "on" : "off", fps, avg[GameEngine.T_NSS], avg[GameEngine.T_NSS_HTP],
                    nfru ? String.format(Locale.US, "\nNFRU %.1f ms per generated frame (net on the HTP %.1f)",
                            avg[GameEngine.T_NFRU], avg[GameEngine.T_NFRU_HTP]) : "");
            if (gen) {
                // the generated frame between t - 1 and t now, frame t half a step later (posted, so the
                // next step's work overlaps it: even pacing without idling the GPU / HTP)
                show(s[2], s[0], st, "NSS + NFRU (generated)", shown++);
                final Bitmap nb = s[1], lb = s[0];
                final String fst = st;
                final long id = shown++;
                overlay.postDelayed(() -> show(nb, lb, fst, "NSS + NFRU", id), (long) (stepMs / 2));
                disp[nDisp++ % disp.length] = System.nanoTime();
                disp[nDisp++ % disp.length] = System.nanoTime();
            } else {
                show(s[1], s[0], st, "NSS", shown++);
                disp[nDisp++ % disp.length] = System.nanoTime();
            }
            if (seq % 30 == 0)
                Log.i(TAG, String.format(Locale.US,
                        "frame %d nfru %b fps %.1f nss %.1f (htp %.1f) nfru %.1f (htp %.1f) upload %.1f copy %.1f", t, nfru,
                        fps, avg[GameEngine.T_NSS], avg[GameEngine.T_NSS_HTP], avg[GameEngine.T_NFRU],
                        avg[GameEngine.T_NFRU_HTP], avg[GameEngine.T_UPLOAD], avg[GameEngine.T_COPY]));
        }
    }
}
