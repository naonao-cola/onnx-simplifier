// Generic TFLite Micro runtime for M5Stack Cardputer.
//
// Loads whatever .tflite model currently sits in the "model" flash
// partition (see partitions.csv, fixed at offset 0x310000) and runs one
// sanity inference over it -- proving the load-from-flash path works for
// ANY int8 TFLite Micro model, without this firmware knowing anything
// model-specific ahead of time. That's deliberate: real per-model input
// feature extraction (MFCC framing for a keyword-spotting model, etc.) is
// model-specific and belongs in a follow-up sketch built against a chosen
// model, not baked into a "generic" runtime that has to work for whichever
// model this device gets flashed with next.
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

namespace {

constexpr const char* kModelPartitionLabel = "model";
constexpr int kTensorArenaSize = 100 * 1024; // generous default; a specific model may need less
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

void printLine(const String& s) {
  Serial.println(s);
  M5Cardputer.Display.println(s);
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

  // Sanity inference over a zeroed input -- proves Invoke() actually runs
  // this model on this device; it says nothing about the model's real
  // accuracy (that needs real sensor data, wired up per-model).
  for (size_t i = 0; i < interpreter->inputs_size(); i++) {
    TfLiteTensor* t = interpreter->input(i);
    memset(t->data.raw, 0, t->bytes);
  }
  TfLiteStatus invoke_status = interpreter->Invoke();
  printLine(invoke_status == kTfLiteOk ? "test inference: OK" : "test inference: FAILED");
  if (invoke_status == kTfLiteOk) {
    for (size_t i = 0; i < interpreter->outputs_size(); i++) {
      TfLiteTensor* t = interpreter->output(i);
      String dims = "";
      for (int d = 0; d < t->dims->size; d++) dims += String(t->dims->data[d]) + (d + 1 < t->dims->size ? "x" : "");
      printLine("  out[" + String(i) + "]: " + dims + " type=" + String(t->type));
    }
  }
}

void loop() {
  delay(1000);
}
