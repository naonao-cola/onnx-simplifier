package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/sam_engine.cpp (EfficientViT-SAM-L0 encoder + decoder on the HTP). */
final class SamEngine {
    static final int IN = 512, LR = 256;

    static {
        System.loadLibrary("sam_demo");
    }

    /** Loads sam_l0_enc / sam_l0_dec (EP-context models compiled on the first launch). null = ok. */
    static native String nativeInit(String modelDir, String nativeLibDir, String opts);
    static native void nativeFitDims(int w, int h, int rot, int[] out);
    /** Camera frame -> disp (upright, fitDims size); with encode also runs the encoder. times: pre, encoder ms. */
    static native boolean nativeYuv(java.nio.ByteBuffer y, java.nio.ByteBuffer u, java.nio.ByteBuffer v, int yRowStride,
                                    int uvRowStride, int uvPixelStride, int w, int h, int rot, Bitmap disp,
                                    boolean encode, float[] times);
    /** Upright RGBA bitmap fit to 512 (longest side) -> encoder. times: pre, encoder ms. */
    static native boolean nativeEncode(Bitmap rgba, float[] times);
    /** Tap at (x, y) display pixels -> mask (256x256, 1 inside), iou[4]; returns the slot, -1 on error. */
    static native int nativeDecode(float x, float y, byte[] mask, float[] iou, float[] times);
    static native String nativeLastError();

    static int[] fitDims(int w, int h, int rot) {
        int[] d = new int[2];
        nativeFitDims(w, h, rot, d);
        return d;
    }

    static void check(boolean ok) {
        if (!ok) throw new RuntimeException(nativeLastError());
    }
}
