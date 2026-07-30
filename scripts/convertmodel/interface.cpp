#include "onnxsim.h"
#include "model_info.h"
#include "onnxoptimizer/optimize.h"
#include "onnx/checker.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

}  // namespace

// Returns an object { model: Uint8Array, trace: string }. `trace` is the
// profiler's Chrome trace JSON when `profile` is true, otherwise "". Returning
// an object (rather than the bare model view) lets the worker hand the trace to
// the in-page flame-graph viewer without a second round-trip.
em::val onnxsimplify_export(const std::string& data, em::val skip_optimizers, bool constant_folding, bool shape_inference, size_t tensor_size_threshold, int target_opset_version, bool profile, bool annotate) {
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
            *GetBuiltinModelExecutor(),
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

std::vector<std::string> onnxoptimizer_passes() {
    return onnx::optimization::GetAvailablePasses();
}

std::vector<std::string> onnxoptimizer_fuse_elimination_passes() {
    return onnx::optimization::GetFuseAndEliminationPass();
}

EMSCRIPTEN_BINDINGS(module) {
    function("onnxsimplify_export", &onnxsimplify_export);
    function("onnxoptimizer_optimize", &onnxoptimizer_optimize);
    function("onnxoptimizer_optimize_fixed", &onnxoptimizer_optimize_fixed);
    em::function("onnxoptimizer_passes", &onnxoptimizer_passes);
    em::function("onnxoptimizer_fuse_elimination_passes", &onnxoptimizer_fuse_elimination_passes);

    em::register_vector<std::string>("string_list");
}
