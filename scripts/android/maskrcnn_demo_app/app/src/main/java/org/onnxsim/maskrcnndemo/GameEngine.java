package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/game_engine.cpp (NSS super sampling + NFRU frame generation replay). */
final class GameEngine {
    /** times[]: NSS frame wall, NSS CNN on the HTP, NFRU wall (per generated frame), upload, NFRU net, copy-out. */
    static final int T_NSS = 0, T_NSS_HTP = 1, T_NFRU = 2, T_UPLOAD = 3, T_NFRU_HTP = 4, T_COPY = 5, T_N = 6;
    static final int LR_W = 960, LR_H = 540, HR_W = 1920, HR_H = 1080;

    static {
        System.loadLibrary("game_demo");
    }

    /** modelDir holds game_seq.bin, game_nss_cnn.onnx, game_nfru_net.onnx; null on success, else the error. */
    static native String nativeInit(String modelDir, String nativeLibDir, String opts);
    static native int nativeFrames();
    static native int nativeStep(int t, boolean nfru, Bitmap lr, Bitmap nss, Bitmap gen, float[] times);
    static native String nativeLastError();

    /** Returns true if gen holds a generated frame (NFRU on, t > 0). */
    static boolean step(int t, boolean nfru, Bitmap lr, Bitmap nss, Bitmap gen, float[] times) {
        int r = nativeStep(t, nfru, lr, nss, gen, times);
        if (r < 0) throw new RuntimeException(nativeLastError());
        return r == 1;
    }
}
