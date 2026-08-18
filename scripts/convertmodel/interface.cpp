#include "onnxsim.h"
#include "model_info.h"
#include "onnxoptimizer/optimize.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "tensor_pool.h"
#include "tensor_pool_bridge.h"
#include "tensor_pool_gguf_bridge.h"

// Version strings baked in by CMake (read from VERSION and
// third_party/onnx-optimizer/VERSION_NUMBER). Fall back to "unknown" so the
// file still compiles outside the CMake build (e.g. an IDE/static check).
#ifndef ONNXSIM_VERSION_STRING
#define ONNXSIM_VERSION_STRING "unknown"
#endif
#ifndef ONNX_OPTIMIZER_VERSION_STRING
#define ONNX_OPTIMIZER_VERSION_STRING "unknown"
#endif

// In the ORT-web build (ONNXSIM_WASM_ORT_WEB) onnxsim is compiled with
// NO_BUILTIN_ORT -- no ONNX Runtime is linked in -- and constant folding is
// delegated to the page's onnxruntime-web via this executor instead.
#ifdef ONNXSIM_WASM_ORT_WEB
#include "js_model_executor.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace em = emscripten;

namespace {

// Where the C++ profiler writes its Chrome trace inside Emscripten's in-memory
// filesystem when the page asks for a profile. Read back into a string and
// removed by ReadAndClearProfileTrace() so nothing accumulates across runs.
constexpr const char* kProfileTracePath = "onnxsim_profile.json";

// Turn onnxsim's simplification profiler on for the next Simplify() call. The
// profiler is driven by environment variables (so every binding gets it for
// free); setenv() feeds the same libc environ that onnxsim.cpp reads via
// std::getenv(). ONNXSIM_MERGE_ORT_PROFILE additionally folds ONNX Runtime's
// per-session constant-folding profiles into the same trace.
void EnableProfiling() {
    setenv("ONNXSIM_PROFILE", kProfileTracePath, 1);
    setenv("ONNXSIM_MERGE_ORT_PROFILE", "1", 1);
}

void DisableProfiling() {
    unsetenv("ONNXSIM_PROFILE");
    unsetenv("ONNXSIM_MERGE_ORT_PROFILE");
}

// Read the trace the profiler wrote (empty string if it is missing) and delete
// the file so the MEMFS does not grow run over run.
std::string ReadAndClearProfileTrace() {
    std::string trace;
    std::ifstream ifs(kProfileTracePath, std::ios::binary);
    if (ifs) {
        std::ostringstream ss;
        ss << ifs.rdbuf();
        trace = ss.str();
    }
    std::remove(kProfileTracePath);
    return trace;
}

// Where the standalone safetensors/gguf export/import functions below stage
// the archive inside Emscripten's in-memory filesystem, mirroring
// kProfileTracePath. On export, the TensorPool bridge bakes this path into
// the archive as the embedded model's external_data "location" (see
// SaveModelAsSafetensorsStandalone / SaveModelAsGGUFStandalone), so it is a
// fixed internal name rather than the user-facing download filename --
// onnxsim's own loaders resolve the embedded model by pool name, not by this
// path, so the mismatch with whatever name the browser saves the download
// under (or the name of a file the user uploads for import) is harmless.
constexpr const char* kSafetensorsArchivePath = "onnxsim_export.safetensors";
constexpr const char* kGGUFArchivePath = "onnxsim_export.gguf";

// Read `path` back into a string and delete it, same pattern as
// ReadAndClearProfileTrace above. Returns an empty string if the file could
// not be opened.
std::string ReadAndDeleteFile(const std::string& path) {
    std::string data;
    // Seek to measure the size and read directly into a pre-sized string,
    // rather than streaming through an ostringstream (which grows its own
    // internal buffer geometrically and copies out of it into the returned
    // string on .str()) -- one read into the final buffer instead of several
    // reallocate-and-copy passes. Matters here because a safetensors/gguf
    // archive embeds the whole model and can be tens of MB.
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (ifs) {
        const std::streamoff size = ifs.tellg();
        if (size > 0) {
            data.resize(static_cast<size_t>(size));
            ifs.seekg(0);
            ifs.read(&data[0], size);
        }
    }
    std::remove(path.c_str());
    return data;
}

// Stage `data` at `path` inside the MEMFS, for the import functions below to
// hand to a TensorPool loader that only knows how to read from a path.
// Returns false (and leaves nothing behind) on a write failure.
bool WriteFile(const std::string& path, const std::string& data) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(ofs);
}

// Serialize `model` into the shared static buffer and hand JS a typed-memory
// view over it (null on failure). The buffer is `static` so the view stays
// valid after this function returns; the worker copies it out immediately
// (toBase64 / a fresh Uint8Array), and calls are sequential, so a single shared
// buffer is safe -- matching how the other bindings here return model bytes.
em::val SerializeModel(const onnx::ModelProto& model) {
    static std::string result;
    if (!model.SerializeToString(&result)) {
        std::cerr << "Serialize failed" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(),
                                         reinterpret_cast<uint8_t*>(result.data())));
}

// Materialize the raw little-endian bytes of `tensor` into `out`, reading either
// its `raw_data` or the typed repeated fields (float_data/int64_data/...). The
// wasm target is little-endian, so writing native bytes yields the
// little-endian layout onnxruntime-web's typed arrays expect. Returns false for
// a dtype we do not pack (STRING / COMPLEX / the narrow float variants stored
// out of band); the caller reports it. The bridged dtypes match
// CppModelExecutor / the ONNX backend node tests.
bool TensorProtoToRawBytes(const onnx::TensorProto& tensor, std::string& out) {
    using TP = onnx::TensorProto;
    out.clear();
    if (tensor.has_raw_data()) {
        out = tensor.raw_data();
        return true;
    }
    auto append = [&out](const void* p, size_t n) {
        out.append(reinterpret_cast<const char*>(p), n);
    };
    switch (tensor.data_type()) {
        case TP::FLOAT:
            for (float v : tensor.float_data()) append(&v, sizeof(v));
            return true;
        case TP::DOUBLE:
            for (double v : tensor.double_data()) append(&v, sizeof(v));
            return true;
        case TP::INT64:
            for (int64_t v : tensor.int64_data()) append(&v, sizeof(v));
            return true;
        case TP::UINT64:
            for (uint64_t v : tensor.uint64_data()) append(&v, sizeof(v));
            return true;
        case TP::INT32:
            for (int32_t v : tensor.int32_data()) append(&v, sizeof(v));
            return true;
        case TP::UINT32:
            // uint32 values live in uint64_data per the TensorProto encoding.
            for (uint64_t v : tensor.uint64_data()) {
                uint32_t w = static_cast<uint32_t>(v);
                append(&w, sizeof(w));
            }
            return true;
        case TP::INT16:
        case TP::UINT16:
        case TP::FLOAT16:
        case TP::BFLOAT16:
            // Stored widened in int32_data; pack the low 16 bits.
            for (int32_t v : tensor.int32_data()) {
                uint16_t w = static_cast<uint16_t>(v);
                append(&w, sizeof(w));
            }
            return true;
        case TP::INT8:
        case TP::UINT8:
        case TP::BOOL:
            for (int32_t v : tensor.int32_data()) {
                uint8_t b = static_cast<uint8_t>(v);
                append(&b, sizeof(b));
            }
            return true;
        default:
            return false;
    }
}

// Highest opset version this build supports for the default ONNX domain
// (ai.onnx), read from the linked onnx at runtime. Returns 1 if the registry
// somehow lacks the default domain. Used to give parsed graphs a sensible
// default opset import.
int HighestDefaultOpset() {
    const auto& ranges =
        onnx::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
    auto it = ranges.find("");  // "" is the default ONNX domain (ai.onnx)
    int max_opset = (it != ranges.end()) ? it->second.second : 0;
    return max_opset > 0 ? max_opset : 1;
}

// Make sure `model` carries an opset import for the default ONNX domain and for
// every domain its local functions live in, so a model assembled from parsed
// text (which may omit the prolog, or reference local functions) still validates
// and can be simplified / visualized. Existing opset imports are left untouched;
// only missing domains are added. A local function's own domain is imported at
// the version the function declares for that domain, defaulting to 1 for a
// custom domain that declares none.
void EnsureOpsetImports(onnx::ModelProto& model) {
    std::set<std::string> present;
    for (const auto& op : model.opset_import()) {
        present.insert(op.domain());
    }
    if (!present.count("")) {
        auto* op = model.add_opset_import();
        op->set_domain("");  // default ONNX domain (ai.onnx)
        op->set_version(HighestDefaultOpset());
        present.insert("");
    }
    for (const auto& fn : model.functions()) {
        const std::string& dom = fn.domain();
        if (present.count(dom)) continue;
        int64_t ver = 1;
        for (const auto& op : fn.opset_import()) {
            if (op.domain() == dom) {
                ver = op.version();
                break;
            }
        }
        auto* op = model.add_opset_import();
        op->set_domain(dom);
        op->set_version(ver);
        present.insert(dom);
    }
}

}  // namespace

// Returns an object { model: Uint8Array, trace: string }. `trace` is the
// profiler's Chrome trace JSON when `profile` is true, otherwise "". Returning
// an object (rather than the bare model view) lets the worker hand the trace to
// the in-page flame-graph viewer without a second round-trip.
em::val onnxsimplify_export(const std::string& data, em::val skip_optimizers, bool constant_folding, bool shape_inference, size_t tensor_size_threshold, int target_opset_version, bool profile, bool annotate, bool graph_diff) {
    InitEnv();

    std::cerr << "LOG_THRESHOLD: " << std::getenv("LOG_THRESHOLD") << std::endl;
    onnx::ModelProto xmodel;
    std::cerr << "parsing message" << std::endl;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }

    if (profile) {
        EnableProfiling();
    }

    std::cerr << "simplify begin" << std::endl;
    onnx::ModelProto optimized;
    try {
        optimized = Simplify(
#ifdef ONNXSIM_WASM_ORT_WEB
            // Runs each fold group through the page's onnxruntime-web. Its Run
            // blocks on a JS Promise, so this whole function is Asyncified and
            // returns a Promise to JS (the worker awaits it).
            *GetJsModelExecutor(),
#else
            *GetBuiltinModelExecutor(),
#endif
            xmodel,
            em::vecFromJSArray<std::string>(skip_optimizers),
            constant_folding,
            shape_inference,
            tensor_size_threshold,
            // A target opset version of <= 0 means "leave the opset unchanged".
            target_opset_version > 0
                ? std::make_optional(target_opset_version)
                : std::nullopt
        );
    } catch (const std::exception& e) {
        std::cerr << "simplify error: " << e.what() << std::endl;
        if (profile) {
            DisableProfiling();
            ReadAndClearProfileTrace();
        }
        return em::val::null();
    }
    std::cerr << "simplify end" << std::endl;

    // Print the before/after diff (op counts + model size) to stdout so the page
    // surfaces it in the log, mirroring the Python CLI's "here is the difference"
    // output. std::cout is routed to the worker's `print` handler.
    std::cout << "Finish! Here is the difference:\n"
              << FormatSimplifyingInfo(xmodel, optimized) << std::flush;

    // Optional, more detailed node/value-level diff -- which nodes/values were
    // removed, added, or changed (matched by output tensor name). Off by
    // default (it can be long for a big model), controlled by the page's
    // "graph diff" toggle.
    if (graph_diff) {
        std::cout << FormatGraphDiff(xmodel, optimized) << std::flush;
    }

    // Collect the trace right after Simplify() (the profiler has flushed it by
    // now) and turn profiling back off, so a later check/serialize failure
    // cannot leave the environment armed for the next request.
    std::string trace;
    if (profile) {
        DisableProfiling();
        trace = ReadAndClearProfileTrace();
    }

    // Bake onnxsim's MAC/FLOP model-info metrics into the model's
    // metadata_props (mirrors Python model_info.annotate_metadata) so the page's
    // "Run inference" panel can read and display them. On by default.
    if (annotate) {
        try {
            AnnotateModelInfo(optimized);
        } catch (const std::exception& e) {
            std::cerr << "annotate model info failed: " << e.what() << std::endl;
        }
    }

    try {
        std::cerr << "checking model" << std::endl;
        onnx::checker::check_model(optimized);
    } catch (const onnx::checker::ValidationError& e) {
        std::cerr << "model check failed: " << e.what() << std::endl;
        return em::val::null();
    }

    std::cerr << "serializing model" << std::endl;
    static std::string result;
    if (!optimized.SerializeToString(&result)) {
        std::cerr << "Serialize failed" << std::endl;
        return em::val::null();
    }
    std::cerr << "model simplify ended" << std::endl;
    em::val out = em::val::object();
    out.set("model", em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data()))));
    out.set("trace", em::val(trace));
    return out;
}

em::val onnxoptimizer_optimize(const std::string& data, em::val passes_ary, bool annotate) {
    std::vector<std::string> passes = em::vecFromJSArray<std::string>(passes_ary);
    onnx::ModelProto xmodel;
    std::cerr << "parsing message" << std::endl;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    onnx::ModelProto optimized;
    try {
        optimized = onnx::optimization::Optimize(xmodel, passes);
    } catch (const std::exception& e) {
        std::cerr << "optimize error: " << e.what() << std::endl;
        return em::val::null();
    }
    if (annotate) {
        try {
            AnnotateModelInfo(optimized);
        } catch (const std::exception& e) {
            std::cerr << "annotate model info failed: " << e.what() << std::endl;
        }
    }
    std::cerr << "serializing model" << std::endl;
    static std::string result;
    if (!optimized.SerializeToString(&result)) {
        std::cerr << "Serialize failed" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

em::val onnxoptimizer_optimize_fixed(const std::string& data, em::val passes_ary, bool annotate) {
    std::vector<std::string> passes = em::vecFromJSArray<std::string>(passes_ary);
    onnx::ModelProto xmodel;
    std::cerr << "parsing message" << std::endl;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    onnx::ModelProto optimized;
    try {
        optimized = onnx::optimization::OptimizeFixed(xmodel, passes);
    } catch (const std::exception& e) {
        std::cerr << "optimize error: " << e.what() << std::endl;
        return em::val::null();
    }
    if (annotate) {
        try {
            AnnotateModelInfo(optimized);
        } catch (const std::exception& e) {
            std::cerr << "annotate model info failed: " << e.what() << std::endl;
        }
    }
    std::cerr << "serializing model" << std::endl;
    static std::string result;
    if (!optimized.SerializeToString(&result)) {
        std::cerr << "Serialize failed" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

// Annotate a model's MAC/FLOP model-info metrics into its metadata_props
// without simplifying/optimizing it, and return the annotated bytes. The page
// uses this to give the *original* uploaded model the same onnxsim.* metrics
// the converted model gets, so the "Run inference" panel can report throughput
// (GFLOP/s) for both and compare their inference speed. Execution is unaffected
// — only metadata_props are added — so the annotated bytes run identically.
// Returns null on a parse/serialize failure (annotation itself is best-effort:
// a model whose shapes cannot be inferred simply gains no metrics).
em::val onnxsim_annotate_model_info(const std::string& data) {
    onnx::ModelProto xmodel;
    std::cerr << "parsing message" << std::endl;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    try {
        AnnotateModelInfo(xmodel);
    } catch (const std::exception& e) {
        std::cerr << "annotate model info failed: " << e.what() << std::endl;
    }
    std::cerr << "serializing model" << std::endl;
    static std::string result;
    if (!xmodel.SerializeToString(&result)) {
        std::cerr << "Serialize failed" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

// Export `data` (a serialized onnx::ModelProto) as a single standalone
// safetensors archive: every initializer's bytes move into the archive with
// real, byte-accurate offsets -- openable by the `safetensors` Python package
// / HF tooling with no onnxsim involved -- and the graph itself is embedded
// alongside them (tensor_pool_bridge.h's SaveModelAsSafetensorsStandalone), so
// the one file this returns is both the model's weights and its graph. Used
// by the page's download-format selector for the ".onnx.safetensors" option.
// Returns null on a parse/export failure.
em::val onnxsim_export_safetensors(const std::string& data) {
    onnx::ModelProto xmodel;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    onnxsim::tensor_pool::TensorPool pool;
    try {
        onnxsim::tensor_pool::SaveModelAsSafetensorsStandalone(
            xmodel, kSafetensorsArchivePath, pool);
    } catch (const std::exception& e) {
        std::cerr << "safetensors export failed: " << e.what() << std::endl;
        std::remove(kSafetensorsArchivePath);
        return em::val::null();
    }
    static std::string result;
    result = ReadAndDeleteFile(kSafetensorsArchivePath);
    if (result.empty()) {
        std::cerr << "failed to read back the exported safetensors file" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

// GGUF counterpart of onnxsim_export_safetensors above; see
// tensor_pool_gguf_bridge.h's SaveModelAsGGUFStandalone. Used by the page's
// download-format selector for the ".onnx.gguf" option.
em::val onnxsim_export_gguf(const std::string& data) {
    onnx::ModelProto xmodel;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    onnxsim::tensor_pool::TensorPool pool;
    try {
        onnxsim::tensor_pool::SaveModelAsGGUFStandalone(xmodel, kGGUFArchivePath,
                                                         pool);
    } catch (const std::exception& e) {
        std::cerr << "gguf export failed: " << e.what() << std::endl;
        std::remove(kGGUFArchivePath);
        return em::val::null();
    }
    static std::string result;
    result = ReadAndDeleteFile(kGGUFArchivePath);
    if (result.empty()) {
        std::cerr << "failed to read back the exported gguf file" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

// Import counterpart of onnxsim_export_safetensors: `data` is the whole raw
// bytes of a standalone safetensors archive (as produced by that function,
// or any other tool following the same self-describing-archive convention --
// tensor_pool_bridge.h's EmbedModel/ExtractModel -- an embedded "model.onnx"
// entry alongside the tensors). Stages it into the MEMFS, loads it back into
// an ordinary, fully in-memory onnx::ModelProto via LoadModelFromSafetensors
// (hydrate_all=true, so every initializer is hydrated -- no lingering
// EXTERNAL references into the now-deleted staged file), and returns its
// serialized bytes. Used by the page's file picker to accept a
// ".onnx.safetensors" upload. Returns null if the archive has no embedded
// model (e.g. a plain weights-only safetensors file with no onnxsim-authored
// graph) or on any other stage/load/parse failure.
em::val onnxsim_import_safetensors(const std::string& data) {
    if (!WriteFile(kSafetensorsArchivePath, data)) {
        std::cerr << "failed to stage the uploaded safetensors file" << std::endl;
        return em::val::null();
    }
    onnx::ModelProto xmodel;
    onnxsim::tensor_pool::TensorPool pool;
    bool ok = false;
    try {
        ok = onnxsim::tensor_pool::LoadModelFromSafetensors(
            kSafetensorsArchivePath, &xmodel, pool);
    } catch (const std::exception& e) {
        std::cerr << "safetensors import failed: " << e.what() << std::endl;
        std::remove(kSafetensorsArchivePath);
        return em::val::null();
    }
    std::remove(kSafetensorsArchivePath);
    if (!ok) {
        std::cerr << "safetensors file has no embedded onnxsim model (a plain "
                     "weights-only archive is not importable as a graph)"
                  << std::endl;
        return em::val::null();
    }
    return SerializeModel(xmodel);
}

// GGUF counterpart of onnxsim_import_safetensors above; see
// tensor_pool_gguf_bridge.h's LoadModelFromGGUF. Used by the page's file
// picker to accept a ".onnx.gguf" upload. `skipped` (any other pooled
// tensors LoadModelFromGGUF could not hydrate, e.g. a quantized weight) is
// logged rather than surfaced structurally -- best-effort, matching how the
// rest of this file reports non-fatal issues via std::cerr.
em::val onnxsim_import_gguf(const std::string& data) {
    if (!WriteFile(kGGUFArchivePath, data)) {
        std::cerr << "failed to stage the uploaded gguf file" << std::endl;
        return em::val::null();
    }
    onnx::ModelProto xmodel;
    onnxsim::tensor_pool::TensorPool pool;
    bool ok = false;
    std::vector<std::string> skipped;
    try {
        ok = onnxsim::tensor_pool::LoadModelFromGGUF(kGGUFArchivePath, &xmodel,
                                                      pool, true, &skipped);
    } catch (const std::exception& e) {
        std::cerr << "gguf import failed: " << e.what() << std::endl;
        std::remove(kGGUFArchivePath);
        return em::val::null();
    }
    std::remove(kGGUFArchivePath);
    if (!ok) {
        std::cerr << "gguf file has no embedded onnxsim model (a plain "
                     "weights-only archive is not importable as a graph)"
                  << std::endl;
        return em::val::null();
    }
    if (!skipped.empty()) {
        std::cerr << "gguf import: " << skipped.size()
                  << " tensor(s) left un-hydrated (quantized/unsupported dtype)"
                  << std::endl;
    }
    return SerializeModel(xmodel);
}

// Report the versions of the libraries baked into this module, for detailed
// bug reports. onnxsim and onnx-optimizer come from CMake (their VERSION files);
// onnx is reported as its max IR version + the highest opset it supports for the
// default (ai.onnx) domain, both read from the linked onnx at runtime; protobuf
// is decoded from GOOGLE_PROTOBUF_VERSION. Returns a plain JS object of strings.
em::val onnxsim_versions() {
    em::val out = em::val::object();
    out.set("onnxsim", std::string(ONNXSIM_VERSION_STRING));
    out.set("onnx_optimizer", std::string(ONNX_OPTIMIZER_VERSION_STRING));

    // onnx: IR version + highest supported opset for the default ONNX domain.
    {
        int max_opset = 0;
        const auto& ranges =
            onnx::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
        auto it = ranges.find("");  // "" is the default ONNX domain (ai.onnx)
        if (it != ranges.end()) {
            max_opset = it->second.second;
        }
        std::ostringstream os;
        os << "IR v" << static_cast<int>(onnx::IR_VERSION) << ", opset "
           << max_opset;
        out.set("onnx", os.str());
    }

    // protobuf: GOOGLE_PROTOBUF_VERSION is major*1000000 + minor*1000 + patch.
#ifdef GOOGLE_PROTOBUF_VERSION
    {
        int pv = GOOGLE_PROTOBUF_VERSION;
        std::ostringstream os;
        os << (pv / 1000000) << "." << (pv / 1000 % 1000) << "." << (pv % 1000);
        out.set("protobuf", os.str());
    }
#else
    out.set("protobuf", std::string("unknown"));
#endif
    return out;
}

// Parse an ONNX *text* representation into a model and return its bytes. Accepts
// either a whole-model text (with an `<ir_version: ..., opset_import: [...]>`
// prolog) or a bare graph body -- the textual form onnx.parser.parse_graph /
// parse_model accept. Local function definitions (one or more `<domain: ...>
// name (...) => (...) {...}` blocks after the graph) are parsed too: the model
// path captures them natively, and either path is normalized so the model
// carries an opset import for the default ONNX domain and for every domain its
// functions use, plus a valid ir_version -- so a graph that calls local
// functions still validates and can be simplified / visualized / run like an
// uploaded model. Returns { model: Uint8Array } on success or { error: string }
// describing the parse failure(s).
em::val onnxsim_parse_graph(const std::string& text) {
    em::val out = em::val::object();
    // Try whole-model text first: this is the only path that captures trailing
    // FunctionProto definitions (GraphProto has no functions field), and it also
    // accepts a bare graph because the `<...>` prolog is optional. Each
    // OnnxParser consumes its input, so a fresh parser is constructed per attempt
    // (the member Parse API is stable across onnx versions).
    onnx::ModelProto model;
    onnx::OnnxParser model_parser(text.c_str());
    onnx::Common::Status model_st = model_parser.Parse(model);
    if (!model_st.IsOK() || !model.has_graph()) {
        // Fall back to graph-only text and wrap it into a model. This path has no
        // functions (the graph syntax cannot carry them), so a model that needs
        // local functions must go through the model path above.
        onnx::GraphProto graph;
        onnx::OnnxParser graph_parser(text.c_str());
        onnx::Common::Status graph_st = graph_parser.Parse(graph);
        if (!graph_st.IsOK()) {
            out.set("error", std::string("failed to parse as a model (") +
                                 model_st.ErrorMessage() +
                                 ") and as a graph (" + graph_st.ErrorMessage() +
                                 ")");
            return out;
        }
        model.Clear();
        *model.mutable_graph() = graph;
    }
    // A bare graph (or a graph with local functions but no explicit prolog) parses
    // without an ir_version or opset imports; fill in sensible defaults so the
    // result validates. Local functions require IR v8+, and EnsureOpsetImports
    // adds an opset import for each function's domain.
    if (model.ir_version() == 0) {
        model.set_ir_version(onnx::IR_VERSION);
    }
    EnsureOpsetImports(model);
    em::val bytes = SerializeModel(model);
    if (bytes.isNull()) {
        out.set("error", std::string("failed to serialize the parsed model"));
        return out;
    }
    out.set("model", bytes);
    return out;
}

// Inline the model's local (model-defined) functions into its main graph: every
// call-site of a FunctionProto listed in `model.functions` is replaced by the
// function body, and the functions are removed from the model. This flattens a
// model that uses local functions into a plain op graph, which onnx-optimizer /
// Simplify and constant folding can then see through. Schema-defined (built-in)
// functions are left alone. Returns the inlined model bytes (null on failure);
// `annotate` optionally bakes MAC/FLOP model info into metadata_props, matching
// the other convert entry points.
em::val onnxsim_inline_functions(const std::string& data, bool annotate) {
    onnx::ModelProto xmodel;
    std::cerr << "parsing message" << std::endl;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    try {
        onnx::inliner::InlineLocalFunctions(xmodel);
    } catch (const std::exception& e) {
        std::cerr << "inline functions error: " << e.what() << std::endl;
        return em::val::null();
    }
    if (annotate) {
        try {
            AnnotateModelInfo(xmodel);
        } catch (const std::exception& e) {
            std::cerr << "annotate model info failed: " << e.what() << std::endl;
        }
    }
    return SerializeModel(xmodel);
}

// Parse a serialized onnx.TensorProto (e.g. an input_N.pb / output_N.pb from an
// ONNX backend test case) into a form onnxruntime-web can consume in JS.
// Returns { dtype: int (TensorProto.DataType), name: string, dims: [int...],
// data: Uint8Array (raw little-endian) } or { error: string }.
em::val onnxsim_parse_tensor(const std::string& data) {
    em::val out = em::val::object();
    onnx::TensorProto tensor;
    if (!tensor.ParseFromArray(data.data(), data.size())) {
        out.set("error", std::string("failed to parse TensorProto"));
        return out;
    }
    static std::string raw;
    if (!TensorProtoToRawBytes(tensor, raw)) {
        std::ostringstream os;
        os << "unsupported tensor data type " << tensor.data_type()
           << " (STRING / COMPLEX and similar are not bridged)";
        out.set("error", os.str());
        return out;
    }
    out.set("dtype", static_cast<int>(tensor.data_type()));
    out.set("name", tensor.name());
    em::val dims = em::val::array();
    for (int i = 0; i < tensor.dims_size(); ++i) {
        dims.set(i, static_cast<double>(tensor.dims(i)));
    }
    out.set("dims", dims);
    out.set("data", em::val(em::typed_memory_view(
                        raw.size(), reinterpret_cast<uint8_t*>(raw.data()))));
    return out;
}

// Run a single simplification building block once (not in the fixed point) for
// debugging, returning the transformed model bytes (null on failure). Shape
// inference and data propagation need no model executor. Constant folding runs
// through the same executor Simplify uses -- in the ORT-web build that awaits
// onnxruntime-web, so (like onnxsimplify_export) onnxsim_fold_constant is
// Asyncified and returns a Promise the worker awaits.
em::val onnxsim_infer_shapes(const std::string& data) {
    InitEnv();
    onnx::ModelProto xmodel;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    try {
        return SerializeModel(InferShapesOnce(xmodel));
    } catch (const std::exception& e) {
        std::cerr << "shape inference error: " << e.what() << std::endl;
        return em::val::null();
    }
}

em::val onnxsim_data_propagation(const std::string& data) {
    InitEnv();
    onnx::ModelProto xmodel;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    try {
        return SerializeModel(PropagateDataOnce(xmodel));
    } catch (const std::exception& e) {
        std::cerr << "data propagation error: " << e.what() << std::endl;
        return em::val::null();
    }
}

em::val onnxsim_fold_constant(const std::string& data, size_t tensor_size_threshold) {
    InitEnv();
    onnx::ModelProto xmodel;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }
    try {
        onnx::ModelProto folded = FoldConstantOnce(
#ifdef ONNXSIM_WASM_ORT_WEB
            *GetJsModelExecutor(),
#else
            *GetBuiltinModelExecutor(),
#endif
            xmodel, tensor_size_threshold);
        return SerializeModel(folded);
    } catch (const std::exception& e) {
        std::cerr << "constant folding error: " << e.what() << std::endl;
        return em::val::null();
    }
}

// True when this module was built to delegate constant folding to
// onnxruntime-web (ONNXSIM_WASM_ORT_WEB). The worker uses this to decide whether
// it must load onnxruntime-web and register a runner on the Module before
// simplifying, and whether onnxsimplify_export returns a Promise (Asyncify) it
// needs to await. In the default built-in-ORT build this is false and the
// worker's behaviour is unchanged.
bool onnxsim_needs_ort_web() {
#ifdef ONNXSIM_WASM_ORT_WEB
    return true;
#else
    return false;
#endif
}

std::vector<std::string> onnxoptimizer_passes() {
    return onnx::optimization::GetAvailablePasses();
}

std::vector<std::string> onnxoptimizer_fuse_elimination_passes() {
    return onnx::optimization::GetFuseAndEliminationPass();
}

EMSCRIPTEN_BINDINGS(module) {
    function("onnxsimplify_export", &onnxsimplify_export);
    function("onnxsim_annotate_model_info", &onnxsim_annotate_model_info);
    function("onnxsim_export_safetensors", &onnxsim_export_safetensors);
    function("onnxsim_export_gguf", &onnxsim_export_gguf);
    function("onnxsim_import_safetensors", &onnxsim_import_safetensors);
    function("onnxsim_import_gguf", &onnxsim_import_gguf);
    function("onnxoptimizer_optimize", &onnxoptimizer_optimize);
    function("onnxoptimizer_optimize_fixed", &onnxoptimizer_optimize_fixed);
    em::function("onnxoptimizer_passes", &onnxoptimizer_passes);
    em::function("onnxoptimizer_fuse_elimination_passes", &onnxoptimizer_fuse_elimination_passes);
    em::function("onnxsim_needs_ort_web", &onnxsim_needs_ort_web);
    // Version reporting, text-graph parsing, TensorProto parsing, and the
    // single-pass debugging entry points (see the definitions above).
    em::function("onnxsim_versions", &onnxsim_versions);
    em::function("onnxsim_parse_graph", &onnxsim_parse_graph);
    em::function("onnxsim_inline_functions", &onnxsim_inline_functions);
    em::function("onnxsim_parse_tensor", &onnxsim_parse_tensor);
    em::function("onnxsim_infer_shapes", &onnxsim_infer_shapes);
    em::function("onnxsim_data_propagation", &onnxsim_data_propagation);
    em::function("onnxsim_fold_constant", &onnxsim_fold_constant);

    em::register_vector<std::string>("string_list");
}
