package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/rtdetr_engine.cpp (RT-DETR-r18: 4 HTP pieces + 3 HVX MSDA calls). */
final class RtDetrEngine {
    static final int MAX_DET = 300, IN = 640;
    static final int T_TOTAL = 0, T_PRE = 1, T_HTP = 2, T_MSDA = 3, T_MSDA_DSP = 4, T_POST = 5, T_N = 6;

    static {
        System.loadLibrary("rtdetr_demo");
    }

    /** opts: "flags=260;thresh=0.4;htp_performance_mode=burst". null = ok. */
    static native String nativeInit(String modelDir, String nativeLibDir, String opts);
    static native int nativeRun(Bitmap rgba, float[] boxes, int[] labels, float[] scores, float[] times);
    static native int nativeRunYuv(java.nio.ByteBuffer y, java.nio.ByteBuffer u, java.nio.ByteBuffer v, int yRowStride,
                                   int uvRowStride, int uvPixelStride, int w, int h, int rot, Bitmap disp,
                                   float[] boxes, int[] labels, float[] scores, float[] times);
    static native void nativeFitDims(int w, int h, int rot, int[] out);
    static native String nativeLastError();
}
