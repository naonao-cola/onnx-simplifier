// Generic TFLite Micro runtime for M5Stack Cardputer.
//
// Loads whatever .tflite model currently sits in the "model" flash
// partition (see partitions.csv, fixed at offset 0x310000) and runs
// inference over it -- proving the load-from-flash path works for ANY
// int8 TFLite Micro model, without this firmware knowing anything
// model-specific ahead of time. Real per-model input feature extraction is
// inherently model-specific, though: if the loaded model's input tensor
// looks like the 40-channel/49-frame int8 audio-spectrogram shape shared by
// the Google Speech Commands keyword-spotting reference models (micro_speech
// TinyConv, MLCommons DS-CNN, etc. -- see onnx-cardputer-flash/README.md's
// candidate table), this runtime feeds it the Cardputer's own microphone
// through the same audio frontend (TF's own microfrontend library, already
// vendored inside Chirale_TensorFLowLite) those models were trained against,
// instead of a zeroed placeholder. Any other input shape/type falls back to
// the original zeroed sanity check -- this is a deliberate narrow special
// case, not a general "runs any model's real preprocessing" claim.
//
// The model is memory-mapped straight out of flash (esp_partition_mmap),
// not copied into RAM: the ESP32-S3 here has ~320KB of SRAM total, and
// some candidate models (see onnx-cardputer-flash/README.md's table) are
// themselves several hundred KB -- copying would not fit.

#include <M5Cardputer.h>
#include <esp_partition.h>
#include <esp_spi_flash.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

namespace {

constexpr const char* kModelPartitionLabel = "model";
constexpr int kTensorArenaSize = 100 * 1024; // generous default; a specific model may need less
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Google Speech Commands reference models' shared audio frontend config
// (micro_speech's own micro_model_settings.h / feature_provider.cc, not
// this library's generic defaults -- see kFeatureSliceSize etc. there):
// 16kHz audio, 30ms windows every 20ms, 40 mel-scale channels per window.
// One second of audio yields exactly 49 windows, matching this model
// family's (1, 40, 1, 49) input shape (channel-major, then time -- the
// onnx2tf conversion's own Transpose, not the frontend's natural
// [frame][channel] order).
constexpr int kAudioSampleRateHz = 16000;
constexpr int kFeatureSliceSize = 40;   // mel-filterbank channels
constexpr int kFeatureSliceCount = 49;  // time frames
constexpr int kAudioBufferSamples = kAudioSampleRateHz; // 1s, comfortably covers 49 frames
int16_t audio_buffer[kAudioBufferSamples];

bool mic_input_ready = false;
FrontendState frontend_state;

void printLine(const String& s) {
  Serial.println(s);
  M5Cardputer.Display.println(s);
}

// True if the model's single input is exactly this audio frontend's output
// shape/type -- the only case this runtime knows how to feed real sensor
// data into (see the file header comment).
bool looksLikeAudioFrontendInput(tflite::MicroInterpreter* interp) {
  if (interp->inputs_size() != 1) return false;
  TfLiteTensor* t = interp->input(0);
  return t->type == kTfLiteInt8 &&
         t->bytes == static_cast<size_t>(kFeatureSliceSize) * kFeatureSliceCount;
}

// Records one second of real audio from the Cardputer's built-in mic, runs
// it through the same microfrontend (FFT -> mel filterbank -> noise
// reduction -> PCAN gain control -> log) the reference models were trained
// against, and writes the quantized result into `input` (assumed to already
// be validated by looksLikeAudioFrontendInput). The uint16 frontend output
// -> int8 model input scaling below is TF's own micro_speech
// feature_provider.cc conversion (value_scale=256, value_div=666 -- "scaling
// values derived from those used in input_data.py in the training
// pipeline"), reproduced here rather than re-derived, since it must match
// whatever a given model was actually trained/calibrated with.
void CaptureMicFeatures(TfLiteTensor* input) {
  memset(input->data.int8, 0, input->bytes);

  M5Cardputer.Mic.record(audio_buffer, kAudioBufferSamples, kAudioSampleRateHz);
  while (M5Cardputer.Mic.isRecording()) delay(1);

  FrontendReset(&frontend_state);

  const int16_t* samples = audio_buffer;
  size_t samples_left = kAudioBufferSamples;
  int frames_written = 0;
  while (samples_left > 0 && frames_written < kFeatureSliceCount) {
    size_t samples_read = 0;
    FrontendOutput output = FrontendProcessSamples(&frontend_state, samples, samples_left, &samples_read);
    samples += samples_read;
    samples_left -= samples_read;
    if (output.size == 0) continue; // not enough samples yet for a full window

    for (size_t c = 0; c < output.size && c < static_cast<size_t>(kFeatureSliceSize); c++) {
      constexpr int32_t kValueScale = 256;
      constexpr int32_t kValueDiv = 666;
      int32_t value = ((static_cast<int32_t>(output.values[c]) * kValueScale) + (kValueDiv / 2)) / kValueDiv;
      value -= 128;
      if (value < -128) value = -128;
      if (value > 127) value = 127;
      // Model input is (1, 40, 1, 49): channel-major, then time.
      input->data.int8[c * kFeatureSliceCount + frames_written] = static_cast<int8_t>(value);
    }
    frames_written++;
  }
}

// Prints one output tensor's actual values (not just shape/type) -- raw
// quantized values plus their dequantized (real-unit) form for int8/uint8,
// since a bare "OK" or a shape dump says nothing about what the model
// actually predicted. Caps how many elements get printed: even a small
// classification head's tensor is fine in full, but nothing here assumes
// the output is small.
void printOutputValues(int index, TfLiteTensor* t) {
  constexpr int kMaxPrinted = 16;
  int count = 1;
  for (int d = 0; d < t->dims->size; d++) count *= t->dims->data[d];
  String raw = "";
  String dequant = "";
  bool has_dequant = t->params.scale != 0.0f && (t->type == kTfLiteInt8 || t->type == kTfLiteUInt8);
  for (int i = 0; i < count && i < kMaxPrinted; i++) {
    if (i > 0) { raw += ","; dequant += ","; }
    float real_value = 0;
    int32_t raw_value = 0;
    switch (t->type) {
      case kTfLiteInt8:    raw_value = t->data.int8[i]; break;
      case kTfLiteUInt8:   raw_value = t->data.uint8[i]; break;
      case kTfLiteFloat32: real_value = t->data.f[i]; break;
      default: break;
    }
    if (t->type == kTfLiteFloat32) {
      dequant += String(real_value, 4);
    } else {
      raw += String(raw_value);
      if (has_dequant) dequant += String((raw_value - t->params.zero_point) * t->params.scale, 4);
    }
  }
  if (t->type == kTfLiteFloat32) {
    printLine("  out[" + String(index) + "] values: " + dequant + (count > kMaxPrinted ? ",..." : ""));
  } else {
    printLine("  out[" + String(index) + "] raw: " + raw + (count > kMaxPrinted ? ",..." : ""));
    if (has_dequant) printLine("  out[" + String(index) + "] dequant: " + dequant + (count > kMaxPrinted ? ",..." : ""));
  }
}

// Runs one inference, printing its result, wall-clock latency, and (on
// success) the actual output values -- not just "OK" and a shape dump.
// Uses real microphone input when the loaded model matches the audio
// frontend's shape/type (see looksLikeAudioFrontendInput); otherwise falls
// back to a zeroed input, which only proves Invoke() runs without saying
// anything about the model's real accuracy.
TfLiteStatus RunAndReportInference() {
  TfLiteTensor* audio_input = mic_input_ready ? interpreter->input(0) : nullptr;
  if (audio_input != nullptr) {
    CaptureMicFeatures(audio_input);
  } else {
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
      TfLiteTensor* t = interpreter->input(i);
      memset(t->data.raw, 0, t->bytes);
    }
  }

  unsigned long start_ms = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long elapsed_ms = millis() - start_ms;

  printLine((invoke_status == kTfLiteOk ? "test inference: OK (" : "test inference: FAILED (") +
            String(elapsed_ms) + " ms)");
  if (invoke_status == kTfLiteOk) {
    for (size_t i = 0; i < interpreter->outputs_size(); i++) {
      printOutputValues(i, interpreter->output(i));
    }
  }
  return invoke_status;
}

// True once the mapped region looks like a real TFLite flatbuffer rather
// than erased flash (0xFF bytes) or garbage -- checked before ever calling
// tflite::GetModel() on it, since an empty "model" partition is the normal
// state right after flashing this runtime for the first time.
bool looksLikeTfliteModel(const uint8_t* data, size_t size) {
  if (size < 8) return false;
  // TFLite flatbuffers carry the ASCII identifier "TFL3" at byte offset 4
  // (flatbuffers::BufferHasIdentifier's own layout).
  return data[4] == 'T' && data[5] == 'F' && data[6] == 'L' && data[7] == '3';
}

} // namespace

void setup() {
  auto cfg = M5.config();
  M5Cardputer.begin(cfg);
  M5Cardputer.Display.setTextSize(1);
  // Without this, text past the bottom of the (small, 240x135) screen is
  // simply not drawn -- loop()'s repeated status/output lines would run off
  // the visible area within a few cycles. With it, the display behaves like
  // a scrolling terminal: once full, it scrolls up a line per new line.
  M5Cardputer.Display.setTextScroll(true);
  Serial.begin(115200);

  printLine("onnx-cardputer-flash runtime");
  printLine("mapping model partition...");

  const esp_partition_t* partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, static_cast<esp_partition_subtype_t>(0x40), kModelPartitionLabel);
  if (!partition) {
    printLine("ERROR: no 'model' partition found");
    printLine("(partitions.csv mismatch -- reflash this runtime)");
    return;
  }

  const void* mapped = nullptr;
  spi_flash_mmap_handle_t mmap_handle;
  esp_err_t err = esp_partition_mmap(partition, 0, partition->size,
                                      SPI_FLASH_MMAP_DATA, &mapped, &mmap_handle);
  if (err != ESP_OK) {
    printLine("ERROR: esp_partition_mmap failed: " + String(esp_err_to_name(err)));
    return;
  }

  const uint8_t* model_bytes = static_cast<const uint8_t*>(mapped);
  if (!looksLikeTfliteModel(model_bytes, partition->size)) {
    printLine("no model flashed yet.");
    printLine("Flash a .tflite at 0x310000");
    printLine("over Web Serial, then reset.");
    return;
  }

  model = tflite::GetModel(model_bytes);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printLine("ERROR: model schema version mismatch");
    printLine("(model: " + String(model->version()) + ", runtime: " + String(TFLITE_SCHEMA_VERSION) + ")");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printLine("ERROR: AllocateTensors() failed");
    printLine("(try a bigger kTensorArenaSize)");
    interpreter = nullptr;
    return;
  }

  printLine("model loaded ok.");
  printLine("inputs: " + String(interpreter->inputs_size()) +
            "  outputs: " + String(interpreter->outputs_size()));
  for (size_t i = 0; i < interpreter->inputs_size(); i++) {
    TfLiteTensor* t = interpreter->input(i);
    String dims = "";
    for (int d = 0; d < t->dims->size; d++) dims += String(t->dims->data[d]) + (d + 1 < t->dims->size ? "x" : "");
    printLine("  in[" + String(i) + "]: " + dims + " type=" + String(t->type));
  }

  if (looksLikeAudioFrontendInput(interpreter)) {
    auto mic_cfg = M5Cardputer.Mic.config();
    mic_cfg.sample_rate = kAudioSampleRateHz;
    M5Cardputer.Mic.config(mic_cfg);
    if (M5Cardputer.Mic.begin()) {
      FrontendConfig frontend_config;
      FrontendFillConfigWithDefaults(&frontend_config);
      frontend_config.filterbank.num_channels = kFeatureSliceSize;
      frontend_config.window.size_ms = 30;
      frontend_config.window.step_size_ms = 20;
      if (FrontendPopulateState(&frontend_config, &frontend_state, kAudioSampleRateHz)) {
        mic_input_ready = true;
        printLine("mic ready -- feeding real audio into the model below");
      } else {
        printLine("ERROR: FrontendPopulateState failed");
      }
    } else {
      printLine("ERROR: M5Cardputer.Mic.begin() failed");
    }
  }

  RunAndReportInference(); // prints its own status/latency/output values
}

void loop() {
  delay(2000);
  // Repeated so a status/error line printed once during setup() isn't the
  // only chance to observe it -- this device's native USB-Serial/JTOG port
  // disconnects and re-enumerates on every reset, so a host reconnecting
  // afterward can otherwise miss a one-shot boot-time print entirely. Also
  // means a mic-driven model gets a fresh real inference (a new one-second
  // recording) every cycle, not just once at boot.
  if (interpreter != nullptr) {
    RunAndReportInference();
  }
}
