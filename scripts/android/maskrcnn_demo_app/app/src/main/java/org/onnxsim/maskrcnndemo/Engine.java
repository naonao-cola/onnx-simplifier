package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/maskrcnn_engine.cpp (the PR #1841 e2e pipeline as a library). */
final class Engine {
    static final int MAX_DET = 100;
    static final int T_TOTAL = 0, T_PRE = 1, T_BACKBONE = 2, T_RPN = 3, T_ROI = 4, T_HEADS = 5, T_CPU = 6, T_N = 7;
    static final int IN_W = 1088, IN_H = 800;

    static {
        System.loadLibrary("maskrcnn_demo");
    }

    /** Returns null on success, else an error message. */
    static native String nativeInit(String modelDir, String nativeLibDir, String pipe);
    static native int nativeRun(Bitmap rgba, float[] boxes, int[] labels, float[] scores, float[] masks, float[] times);
    static native String nativeLastError();

    /** One frame's output, in model-input pixel coordinates (image top-left aligned in 1088x800). */
    static final class Result {
        final float[] boxes = new float[4 * MAX_DET];
        final int[] labels = new int[MAX_DET];
        final float[] scores = new float[MAX_DET];
        final float[] masks = new float[784 * MAX_DET];
        final float[] times = new float[T_N];
        int n;
        Bitmap frame;   // the (resized) frame the result belongs to
    }

    static boolean run(Bitmap rgba, Result r) {
        r.n = nativeRun(rgba, r.boxes, r.labels, r.scores, r.masks, r.times);
        return r.n >= 0;
    }
}
