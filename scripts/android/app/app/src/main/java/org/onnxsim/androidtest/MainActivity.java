package org.onnxsim.androidtest;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public final class MainActivity extends Activity {
    private static final String TAG = "OnnxSimDeviceTest";

    static {
        System.loadLibrary("onnxruntime");
        System.loadLibrary("onnxruntime_providers_qnn");
        System.loadLibrary("onnxsim_device_test");
    }

    private static native String runModel(String originalPath, String simplifiedPath,
                                          String inputPath, String outputPath,
                                          String target, String qnnLibraryPath);

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        String target = getIntent().getStringExtra("target");
        if (target == null) target = "cpu";
        File resultFile = new File(getFilesDir(), "result_" + target + ".txt");
        resultFile.delete();
        String result;
        try {
            File original = copyAsset("original.onnx");
            File simplified = copyAsset("simplified.onnx");
            File input = copyAsset("input.f32");
            File output = new File(getFilesDir(), "output_" + target + ".f32");
            String qnnPath = "libonnxruntime_providers_qnn.so";
            result = runModel(original.getAbsolutePath(), simplified.getAbsolutePath(),
                    input.getAbsolutePath(), output.getAbsolutePath(), target, qnnPath);
        } catch (Throwable error) {
            result = "FAIL " + target + ": " + error;
        }
        try (FileOutputStream stream = new FileOutputStream(
                resultFile)) {
            stream.write(result.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (Exception error) {
            Log.e(TAG, "Could not write result", error);
        }
        Log.i(TAG, result);
        finish();
    }

    private File copyAsset(String name) throws Exception {
        File destination = new File(getCacheDir(), name);
        try (InputStream input = getAssets().open(name);
             FileOutputStream output = new FileOutputStream(destination)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        }
        return destination;
    }
}
