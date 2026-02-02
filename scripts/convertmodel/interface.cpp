#include "onnxsim.h"
#include "onnxoptimizer/optimize.h"
#include "onnx/checker.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace em = emscripten;

em::val onnxsimplify_export(const std::string& data, em::val skip_optimizers, bool constant_folding, bool shape_inference, size_t tensor_size_threshold) {
    InitEnv();

    std::cerr << "LOG_THRESHOLD: " << std::getenv("LOG_THRESHOLD") << std::endl;
    onnx::ModelProto xmodel;
    std::cerr << "parsing message" << std::endl;
    if (!xmodel.ParseFromArray(data.data(), data.size())) {
        std::cerr << "Parse failed" << std::endl;
        return em::val::null();
    }

    std::cerr << "simplify begin" << std::endl;
    onnx::ModelProto optimized;
    try {
        optimized = Simplify(
            xmodel,
            em::vecFromJSArray<std::string>(skip_optimizers),
            constant_folding,
            shape_inference,
            tensor_size_threshold
        );
    } catch (const std::exception& e) {
        std::cerr << "simplify error: " << e.what() << std::endl;
        return em::val::null();
    }
    std::cerr << "simplify end" << std::endl;

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
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

em::val onnxoptimizer_optimize(const std::string& data, em::val passes_ary) {
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
    std::cerr << "serializing model" << std::endl;
    static std::string result;
    if (!optimized.SerializeToString(&result)) {
        std::cerr << "Serialize failed" << std::endl;
        return em::val::null();
    }
    return em::val(em::typed_memory_view(result.size(), reinterpret_cast<uint8_t*>(result.data())));
}

em::val onnxoptimizer_optimize_fixed(const std::string& data, em::val passes_ary) {
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
