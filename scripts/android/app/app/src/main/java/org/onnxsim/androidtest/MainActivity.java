package org.onnxsim.androidtest;

import android.app.Activity;
import android.os.Bundle;
import android.system.Os;
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
        String profilePrefix = "output_" + target + ".f32.profile_";
        File[] oldProfiles = getFilesDir().listFiles(
                (directory, name) -> name.startsWith(profilePrefix));
        if (oldProfiles != null) {
            for (File profile : oldProfiles) profile.delete();
        }
        String qnnProfilePrefix = "output_" + target + ".f32.";
        File[] oldQnnProfiles = getFilesDir().listFiles(
                (directory, name) -> name.startsWith(qnnProfilePrefix) &&
                        (name.endsWith(".qnn.csv") || name.endsWith(".optrace.csv") ||
                         name.endsWith(".qnn.log") || name.endsWith(".optrace_qnn.log") ||
                         name.endsWith(".ctx.onnx")));
        if (oldQnnProfiles != null) {
            for (File profile : oldQnnProfiles) profile.delete();
        }
        File[] oldContexts = getCacheDir().listFiles(
                (directory, name) -> name.endsWith(".ctx.onnx") ||
                        name.endsWith(".ctx_qnn.bin") || name.endsWith("_schematic.bin"));
        if (oldContexts != null) {
            for (File context : oldContexts) context.delete();
        }
        String result;
        try {
            if (target.startsWith("qnn-")) configureQnnRuntime();
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

    private void configureQnnRuntime() throws Exception {
        String[] paths = {
            getApplicationInfo().nativeLibraryDir,
            "/odm/lib/rfsa/adsp",
            "/vendor/lib/rfsa/adsp",
            "/vendor/lib/rfsa/adsp/",
            "/system/lib/rfsa/adsp",
            "/system/vendor/lib/rfsa/adsp",
            "/dsp",
        };
        String existing = System.getenv("ADSP_LIBRARY_PATH");
        StringBuilder value = new StringBuilder(String.join(";", paths));
        if (existing != null && !existing.isEmpty()) value.append(';').append(existing);
        Os.setenv("ADSP_LIBRARY_PATH", value.toString(), true);
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
