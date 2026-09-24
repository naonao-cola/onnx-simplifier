package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/sr_engine.cpp (an x4 super-resolution model on the HTP). */
final class SrEngine {
    static final int T_TOTAL = 0, T_PRE = 1, T_HTP = 2, T_POST = 3, T_BICUBIC = 4, T_N = 5;
    static final int SCALE = 4;

    static {
        System.loadLibrary("sr_demo");
    }

    /**
     * model: stem prefix in modelDir (sr_xlsr_int8 -> sr_xlsr_int8_270x480.onnx and _480x270.onnx;
     * their EP-context models are compiled on first use). opts: "htp_performance_mode=burst".
     * Returns null on success, else the error. May be called again to switch models.
     */
    static native String nativeInit(String modelDir, String nativeLibDir, String model, String opts);

    /** Any bitmap may be null (not filled); sizes: hr/sr/bic = nativeHrDims, lr = that / 4. */
    static native int nativeRunYuv(java.nio.ByteBuffer y, java.nio.ByteBuffer u, java.nio.ByteBuffer v, int yRowStride,
                                   int uvRowStride, int uvPixelStride, int w, int h, int rot, Bitmap hr, Bitmap lr,
                                   Bitmap sr, Bitmap bic, float[] times);
    static native int nativeRun(Bitmap hr, Bitmap lr, Bitmap sr, Bitmap bic, float[] times);
    static native void nativeHrDims(int w, int h, int rot, int[] out);
    static native String nativeLastError();

    static int[] hrDims(int w, int h, int rot) {
        int[] d = new int[2];
        nativeHrDims(w, h, rot, d);
        return d;
    }

    static void runYuv(android.media.Image img, int rot, Bitmap hr, Bitmap lr, Bitmap sr, Bitmap bic, float[] t) {
        android.media.Image.Plane[] p = img.getPlanes();
        if (nativeRunYuv(p[0].getBuffer(), p[1].getBuffer(), p[2].getBuffer(), p[0].getRowStride(),
                p[1].getRowStride(), p[1].getPixelStride(), img.getWidth(), img.getHeight(), rot, hr, lr, sr, bic, t) < 0)
            throw new RuntimeException(nativeLastError());
    }

    static void run(Bitmap hr, Bitmap lr, Bitmap sr, Bitmap bic, float[] t) {
        if (nativeRun(hr, lr, sr, bic, t) < 0) throw new RuntimeException(nativeLastError());
    }
}
