#include "onnxsim.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>

#ifndef NO_BUILTIN_ORT
#include "onnxruntime/core/common/endian.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#endif
#include "contrib_schemas.h"
#include "onnx/common/file_utils.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/passes/logging.h"

struct Config {
  std::vector<std::string> optimizer_passes;
  // default value is max
  size_t tensor_size_threshold = -1;
};

Config config;

bool IsOfficialOp(const std::string& domain, const std::string& op) {
  if (domain != "ai.onnx" && domain != "ai.onnx.ml" && !domain.empty()) {
    return false;
  }
  // these experimental ops were in onnx default domain but are no
  // longer supported by onnx now.
  static std::set<std::string> experimental_ops = {"ATen",
                                                   "Affine",
                                                   "ConstantFill",
                                                   "Crop",
                                                   "DynamicSlice",
                                                   "GRUUnit",
                                                   "GivenTensorFill",
                                                   "ImageScaler",
                                                   "ParametricSoftplus",
                                                   "Scale",
                                                   "ScaledTanh"};
  return experimental_ops.find(op) == experimental_ops.end();
}

bool IsDeterministic(const std::string& domain, const std::string& op,
                     int opset_version) {
  // Query the determinism attribute of the operator schema instead of
  // maintaining a hardcoded list of non-deterministic ops. See
  // https://github.com/onnx/onnx/pull/7176.
  //
  // The ONNX operator schema registry stores the default ONNX domain as an
  // empty string.
  const std::string& lookup_domain = domain == "ai.onnx" ? "" : domain;
  const auto* schema =
      onnx::OpSchemaRegistry::Schema(op, opset_version, lookup_domain);
  if (schema == nullptr) {
    // Unknown op. Assume it is not deterministic.
    return false;
  }
  // Only fold ops that are known to be deterministic. Ops whose determinism
  // cannot be statically determined (e.g. context-dependent functions) are
  // treated as non-deterministic to be safe.
  return schema->GetNodeDeterminism() ==
         onnx::OpSchema::NodeDeterminism::Deterministic;
}

bool IsQDQ(const std::string& domain, const std::string& op) {
  if (domain == "ai.onnx" || domain.empty()) {
    return op == "QuantizeLinear" || op == "DequantizeLinear";
  }
  return false;
}

auto FindInitializerByName(const onnx::ModelProto& model,
                           const std::string& name) {
  for (const auto& initializer : model.graph().initializer()) {
    if (initializer.name() == name) {
      return initializer;
    }
  }
  throw std::invalid_argument("no initializer " + name);
}

auto FindValueInfoProtoByName(const onnx::ModelProto& model,
                              const std::string& name) {
  for (const auto& vi : model.graph().value_info()) {
    if (vi.name() == name) {
      return vi;
    }
  }
  for (const auto& initializer : model.graph().initializer()) {
    if (initializer.name() == name) {
      onnx::ValueInfoProto vi;
      for (const auto& dim : initializer.dims()) {
        vi.mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim()
            ->set_dim_value(dim);
      }
      vi.mutable_type()->mutable_tensor_type()->set_elem_type(
          initializer.data_type());
      vi.set_name(name);
      return vi;
    }
  }
  throw std::invalid_argument("no value info " + name);
}

#ifndef NO_BUILTIN_ORT
onnx::TensorProto TensorToTensorProto(const Ort::Value& tensor) {
  onnx::TensorProto tensor_proto;
  for (const auto& dim : tensor.GetTensorTypeAndShapeInfo().GetShape()) {
    tensor_proto.add_dims(dim);
  }
  onnx::TensorProto::DataType onnx_dtype =
      (onnx::TensorProto::DataType)tensor.GetTensorTypeAndShapeInfo()
          .GetElementType();
  tensor_proto.set_data_type(onnx_dtype);

  switch (onnx_dtype) {
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)                   \
  case onnx::TensorProto::onnx_dtype: {                                   \
    const auto* dptr = tensor.GetTensorData<cpp_type>();                  \
    for (size_t i = 0;                                                    \
         i < tensor.GetTensorTypeAndShapeInfo().GetElementCount(); i++) { \
      tensor_proto.add_##storage_dtype##_data(dptr[i]);                   \
    }                                                                     \
    break;                                                                \
  }

    CASE_DTYPE(FLOAT, float, float)
    CASE_DTYPE(DOUBLE, double, double)
    CASE_DTYPE(INT64, int64, int64_t)
    CASE_DTYPE(UINT64, uint64, uint64_t)
    CASE_DTYPE(INT32, int32, int32_t)
    CASE_DTYPE(UINT8, int32, uint8_t)
    CASE_DTYPE(INT8, int32, int8_t)
    CASE_DTYPE(UINT16, int32, uint16_t)
    CASE_DTYPE(INT16, int32, int16_t)
    CASE_DTYPE(BOOL, int32, int8_t)
#undef CASE_DTYPE
    default:
      throw std::invalid_argument("Unknown dtype " +
                                  std::to_string(tensor_proto.data_type()));
  }
  return tensor_proto;
}

Ort::Value TensorProtoToTensor(const onnx::TensorProto& tensor_proto) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto tensor = Ort::Value::CreateTensor(
      allocator, tensor_proto.dims().data(), tensor_proto.dims_size(),
      (ONNXTensorElementDataType)tensor_proto.data_type());
  if (tensor_proto.has_raw_data()) {
    if (onnxruntime::endian::native == onnxruntime::endian::big) {
      throw std::invalid_argument("only little endian is supported");
    }
    memcpy(tensor.GetTensorMutableData<void>(), tensor_proto.raw_data().data(),
           tensor_proto.raw_data().size());
  } else {
    switch (tensor_proto.data_type()) {
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)         \
  case onnx::TensorProto::onnx_dtype: {                         \
    std::vector<cpp_type> vec;                                  \
    for (const auto& x : tensor_proto.storage_dtype##_data()) { \
      vec.push_back(x);                                         \
    }                                                           \
    memcpy(tensor.GetTensorMutableData<void>(), vec.data(),     \
           vec.size() * sizeof(cpp_type));                      \
    break;                                                      \
  }
      CASE_DTYPE(FLOAT, float, float)
      CASE_DTYPE(DOUBLE, double, double)
      CASE_DTYPE(INT64, int64, int64_t)
      CASE_DTYPE(UINT64, uint64, uint64_t)
      CASE_DTYPE(INT32, int32, int32_t)
      CASE_DTYPE(UINT8, int32, uint8_t)
      CASE_DTYPE(INT8, int32, int8_t)
      CASE_DTYPE(UINT16, int32, uint16_t)
      CASE_DTYPE(INT16, int32, int16_t)
      CASE_DTYPE(BOOL, int32, int8_t)
#undef CASE_DTYPE
      default:
        throw std::invalid_argument("Unknown dtype " +
                                    std::to_string(tensor_proto.data_type()));
    }
  }
  return tensor;
}

std::shared_ptr<Ort::Env> GetEnv() {
  static std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>();
  return env;
}

struct CppModelExecutor : public ModelExecutor {
  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    std::transform(
        model.graph().input().begin(), model.graph().input().end(),
        std::back_inserter(input_name_ptrs),
        [](const onnx::ValueInfoProto& x) { return x.name().c_str(); });
    std::transform(
        model.graph().output().begin(), model.graph().output().end(),
        std::back_inserter(output_name_ptrs),
        [](const onnx::ValueInfoProto& x) { return x.name().c_str(); });
    Ort::SessionOptions sess_opts;
    sess_opts.SetLogSeverityLevel(3);
    sess_opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    std::string model_str = model.SerializeAsString();
    Ort::Session session(*GetEnv(), model_str.data(), model_str.size(),
                         sess_opts);
    Ort::RunOptions run_opts;
    run_opts.SetRunLogSeverityLevel(3);
    std::vector<Ort::Value> input_tensors;
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(input_tensors), TensorProtoToTensor);
    auto output_tensors = session.Run(
        run_opts, input_name_ptrs.data(), input_tensors.data(),
        input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());

    std::vector<onnx::TensorProto> output_tps;
    std::transform(output_tensors.begin(), output_tensors.end(),
                   std::back_inserter(output_tps), TensorToTensorProto);
    return output_tps;
  }
};

std::shared_ptr<const ModelExecutor> GetBuiltinModelExecutor() {
  static std::shared_ptr<const ModelExecutor> executor =
      std::make_shared<CppModelExecutor>();
  return executor;
}

void InitEnv() { GetEnv(); }
#else
void InitEnv() {
  // do nothing
}
#endif

std::vector<onnx::TensorProto> RunOp(const ModelExecutor& executor,
                                     onnx::ModelProto& model,
                                     const onnx::NodeProto& op) {
  std::vector<std::string> input_names;
  std::vector<onnx::TensorProto> input_tps;
  std::set<std::string> initializer_names;

  onnx::ModelProto op_model;
  op_model.set_ir_version(model.ir_version());
  for (const auto& x : model.opset_import()) {
    *op_model.add_opset_import() = x;
  }
  *op_model.mutable_graph()->add_node() = op;

  for (const auto& input : op.input()) {
    if (std::find(input_names.begin(), input_names.end(), input) !=
        input_names.end()) {
      continue;
    }
    // skip "" which represents the unset optional input
    if (input.empty()) {
      continue;
    }
    if (initializer_names.find(input) != initializer_names.end()) {
      continue;
    }
    auto in_tp = FindInitializerByName(model, input);
    if (in_tp.dims().size() == 1 && in_tp.dims()[0] == 0) {
      initializer_names.insert(input);
      *op_model.mutable_graph()->add_initializer() = in_tp;
      continue;
    }
    input_names.push_back(input);
    input_tps.push_back(in_tp);
  }

  for (const auto& x : input_names) {
    // skip "" which represents the unset optional input
    if (x.empty()) {
      continue;
    }
    *op_model.mutable_graph()->add_input() = FindValueInfoProtoByName(model, x);
  }
  for (const auto& x : op.output()) {
    onnx::ValueInfoProto vi;
    // In principle output ValueInfoProto must have type. But it is not checked.
    vi.set_name(x);
    *op_model.mutable_graph()->add_output() = vi;
  }

  using namespace ONNX_NAMESPACE::optimization;
  VLOG(1) << "Running node: " << op;
  auto output_tps = executor._Run(op_model, input_tps);
  for (size_t i = 0; i < op.output_size(); i++) {
    output_tps[i].set_name(op.output(i));
  }
  return output_tps;
}

void RunOpAndAddInitializer(const ModelExecutor& executor,
                            onnx::ModelProto& model,
                            const onnx::NodeProto& op) {
  const auto output_tps = RunOp(executor, model, op);
  for (const auto& output_tp : output_tps) {
    *model.mutable_graph()->add_initializer() = output_tp;
  }
}

bool HasSubgraph(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.type() == onnx::AttributeProto::GRAPH ||
        attr.type() == onnx::AttributeProto::GRAPHS) {
      return true;
    }
  }
  return false;
}

size_t size_of_dtype(onnx::TensorProto::DataType dtype) {
  switch (dtype) {
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:
      return 1;
    case onnx::TensorProto::DataType::TensorProto_DataType_BFLOAT16:
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:
      return 2;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:
      return 4;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:
      return 8;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128:
      return 16;
    // Don't know the size of string.. Just return 16.
    case onnx::TensorProto::DataType::TensorProto_DataType_STRING:
      return 16;
    default:
    case onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED:
      throw std::invalid_argument("Undefined or unknown datatype");
  }
  throw std::invalid_argument("Unknown datatype " + std::to_string(dtype));
}

bool ProduceLargeTensor(const onnx::ModelProto& model,
                        const onnx::NodeProto& node, size_t threshold) {
  std::set<std::string> large_tensor_ops{"Tile", "ConstantOfShape", "Expand"};
  if (large_tensor_ops.find(node.op_type()) == large_tensor_ops.end()) {
    return false;
  }
  for (const auto& value_info : model.graph().value_info()) {
    if (value_info.name() == node.output(0)) {
      size_t size = size_of_dtype(static_cast<onnx::TensorProto::DataType>(
          value_info.type().tensor_type().elem_type()));
      for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
        size *= dim.dim_value();
      }
      if (size <= threshold) {
        return false;
      }
    }
  }
  // If the output is not in value_info, we assume it is large.
  // There is a possibility that value_info is presented by the shape inference
  // later and `ProduceLargeTensor` is called again and returns false at that
  // time.
  return true;
}

std::pair<std::vector<onnx::NodeProto>, std::vector<onnx::NodeProto>>
GetConstantNodes(const onnx::ModelProto& model) {
  // tensor with empty name("") represents the empty value of an optional input
  // so "" should be treated as a name of a constant tensor.
  std::vector<std::string> const_names{""};
  std::vector<onnx::NodeProto> const_nodes;
  std::vector<onnx::NodeProto> non_const_nodes;
  std::transform(
      model.graph().initializer().begin(), model.graph().initializer().end(),
      std::back_inserter(const_names), [](const auto& x) { return x.name(); });
  // Map each domain to its imported opset version so the correct operator
  // schema can be looked up. The default ONNX domain is normalized to an empty
  // string, which is how the schema registry stores it.
  std::unordered_map<std::string, int> domain_to_version;
  for (const auto& opset : model.opset_import()) {
    const std::string& domain =
        opset.domain() == "ai.onnx" ? "" : opset.domain();
    domain_to_version[domain] = opset.version();
  }
  auto opset_version_of = [&domain_to_version](const std::string& domain) {
    const std::string& key = domain == "ai.onnx" ? "" : domain;
    auto iter = domain_to_version.find(key);
    return iter == domain_to_version.end() ? 0 : iter->second;
  };
  // node is already topo sorted
  for (const auto& node : model.graph().node()) {
    // clang-format off
    if (IsOfficialOp(node.domain(), node.op_type()) &&
        IsDeterministic(node.domain(), node.op_type(),
                        opset_version_of(node.domain())) &&
        !IsQDQ(node.domain(), node.op_type()) &&
        !HasSubgraph(node) &&
        !ProduceLargeTensor(model, node, config.tensor_size_threshold) &&
        // clang-format on
        std::all_of(node.input().begin(), node.input().end(),
                    [&const_names](const auto& x) {
                      return std::find(const_names.begin(), const_names.end(),
                                       x) != const_names.end();
                    })) {
      const_names.insert(const_names.end(), node.output().begin(),
                         node.output().end());
      const_nodes.push_back(node);
    } else {
      non_const_nodes.push_back(node);
    }
  }
  return {const_nodes, non_const_nodes};
}

// Recursively collect the names of every tensor consumed as a node input,
// descending into subgraphs (e.g. the branches of "If" or the body of "Loop").
// Because ONNX subgraphs can reference tensors from the enclosing scope, an
// initializer in the main graph may be used only by a node inside a subgraph.
// Collecting names recursively ensures such initializers are not mistaken for
// unused ones (issue #174).
void CollectUsedTensorNames(const onnx::GraphProto& graph,
                            std::set<std::string>& used) {
  for (const auto& node : graph.node()) {
    for (const auto& input : node.input()) {
      if (!input.empty()) {
        used.insert(input);
      }
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectUsedTensorNames(attr.g(), used);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectUsedTensorNames(subgraph, used);
      }
    }
  }
  // Graph outputs must be kept even if no node consumes them.
  for (const auto& output : graph.output()) {
    used.insert(output.name());
  }
}

// Remove initializers of the main graph that are no longer referenced by any
// node (including nodes in subgraphs). Constant folding replaces a subgraph of
// const ops (e.g. a Transpose on a weight) with a freshly computed initializer,
// but leaves the original operand initializers in place. Without cleanup those
// dangling weights are duplicated in the graph, which can push the model past
// the 2GB protobuf limit before the onnx optimizer gets a chance to remove
// them (issue #174).
onnx::ModelProto EliminateUnusedInitializer(const onnx::ModelProto& model) {
  onnx::ModelProto result;
  result.CopyFrom(model);

  std::set<std::string> used;
  CollectUsedTensorNames(result.graph(), used);
  // Keep initializers that double as graph inputs (their default value);
  // dropping them would silently turn them into required inputs.
  for (const auto& input : result.graph().input()) {
    used.insert(input.name());
  }

  google::protobuf::RepeatedPtrField<onnx::TensorProto> kept;
  for (auto& initializer : *result.mutable_graph()->mutable_initializer()) {
    if (used.count(initializer.name()) > 0) {
      *kept.Add() = std::move(initializer);
    }
  }
  result.mutable_graph()->mutable_initializer()->Swap(&kept);

  return result;
}

// Mutates the model in place; ``onnx::shape_inference::InferShapes`` already
// works in place, so no extra ModelProto copy is made (the previous ``const&``
// signature forced a defensive ``CopyFrom`` because the input could not be
// mutated).
void _InferShapes(onnx::ModelProto& model) {
  onnx::shape_inference::InferShapes(model);
}

// Build a lookup from tensor name to its type, gathering shapes from every
// place a shape can be declared: value_info (populated by shape inference),
// graph inputs and graph outputs. Pointers reference `model`, so the map must
// not outlive it and `model` must not be mutated while the map is in use.
std::unordered_map<std::string, const onnx::TypeProto*> BuildTypeMap(
    const onnx::ModelProto& model) {
  std::unordered_map<std::string, const onnx::TypeProto*> type_map;
  auto add = [&type_map](const onnx::ValueInfoProto& vi) {
    if (vi.has_type()) {
      type_map[vi.name()] = &vi.type();
    }
  };
  for (const auto& vi : model.graph().value_info()) add(vi);
  for (const auto& vi : model.graph().input()) add(vi);
  for (const auto& vi : model.graph().output()) add(vi);
  return type_map;
}

// Fetch the element type and a fully static shape of `name` from `type_map`.
// Returns false unless the tensor has a known integer (INT64/INT32) element
// type and a shape whose every dimension is a fixed value. A rank-0 (scalar)
// tensor yields an empty `dims` (element count 1).
bool GetStaticIntTensorInfo(
    const std::unordered_map<std::string, const onnx::TypeProto*>& type_map,
    const std::string& name, onnx::TensorProto::DataType& elem_type,
    std::vector<int64_t>& dims) {
  auto iter = type_map.find(name);
  if (iter == type_map.end() || !iter->second->has_tensor_type()) {
    return false;
  }
  const auto& tensor_type = iter->second->tensor_type();
  elem_type = static_cast<onnx::TensorProto::DataType>(tensor_type.elem_type());
  if (elem_type != onnx::TensorProto::INT64 &&
      elem_type != onnx::TensorProto::INT32) {
    return false;
  }
  if (!tensor_type.has_shape()) {
    // Rank is unknown.
    return false;
  }
  dims.clear();
  for (const auto& dim : tensor_type.shape().dim()) {
    if (!dim.has_dim_value()) {
      return false;
    }
    dims.push_back(dim.dim_value());
  }
  return true;
}

// Partial shape evaluation (issue #139) via ONNX data propagation.
//
// The plain constant folder only folds a node when *all* of its inputs are
// constant, so shape-computing ops like `Shape` are never folded: their input
// is an activation. Yet those ops depend solely on shapes, which shape
// inference knows -- fully or partially -- even when some dimensions stay
// dynamic.
//
// ONNX shape inference can *propagate* those partially known values: with data
// propagation enabled it fills a DataValueMap mapping each tensor to a
// TensorShapeProto whose entries are either a concrete dim_value or a symbolic
// dim_param. Ops across the shape family (Shape, Gather, Slice, Concat,
// Squeeze/Unsqueeze, Cast, Add/Sub/Mul, ...) participate, so a chain like
//   Shape([batch, C, H, W]) -> Gather([1, 2, 3])  ==>  [C, H, W]
// is propagated end to end and comes out fully concrete even though the batch
// dimension stays dynamic (the mask-rcnn pattern from issue #139).
//
// This pass rewrites every node whose lone output has a fully concrete
// propagated value into a `Constant` node. Downstream ops then fold through the
// ordinary constant folder, and now-dead nodes are removed by the optimizer.
void _EvalPartialShape(onnx::ModelProto& model) {
  // This pass runs shape inference with *data propagation* (lenient options)
  // purely to discover foldable shape values; it must not otherwise change the
  // model. InferShapes mutates value_info and output types in place, so
  // snapshot those annotations and restore them on the paths that fold nothing,
  // leaving the model byte-for-byte unchanged (the old code returned the
  // untouched input there). The snapshot is metadata only -- no tensor weights
  // -- so it is cheap, unlike the full-model ``CopyFrom`` it replaces. Restoring
  // also keeps this pass's data-propagation value_info out of the model, which
  // matters: it differs from the regular shape-inference pass's value_info, and
  // leaving it behind could make the outer fixed point oscillate.
  auto saved_value_info = model.graph().value_info();
  auto saved_output = model.graph().output();
  auto restore = [&]() {
    *model.mutable_graph()->mutable_value_info() = saved_value_info;
    *model.mutable_graph()->mutable_output() = saved_output;
  };

  onnx::shape_inference::DataValueMap data_map;
  try {
    const onnx::ShapeInferenceOptions options(/*check_type=*/false,
                                              /*error_mode=*/0,
                                              /*enable_data_propagation=*/true);
    onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(),
                                       options, &data_map);
  } catch (const std::exception&) {
    // If shape inference fails we simply have no propagated values to exploit.
    restore();
    return;
  }

  if (data_map.empty()) {
    restore();
    return;
  }

  const auto type_map = BuildTypeMap(model);

  // Maps the output of a foldable node to the constant tensor it produces. Each
  // such node is rewritten into a `Constant` node holding this value.
  std::unordered_map<std::string, onnx::TensorProto> folded_values;

  for (const auto& node : model.graph().node()) {
    // Shape-family ops are single-output; only replace a node when its lone
    // output is fully known, so dropping it can never orphan a second output.
    if (node.output_size() != 1) {
      continue;
    }
    const std::string& output = node.output(0);
    auto data_iter = data_map.find(output);
    if (data_iter == data_map.end()) {
      continue;
    }

    // Every element must be statically known. Data propagation represents an
    // unknown element as a dimension with neither dim_value nor dim_param, so
    // requiring dim_value on every entry both proves the value is concrete and
    // filters out activations whose rank alone is known.
    const onnx::TensorShapeProto& value = data_iter->second;
    bool fully_known = true;
    std::vector<int64_t> values;
    for (const auto& dim : value.dim()) {
      if (!dim.has_dim_value()) {
        fully_known = false;
        break;
      }
      values.push_back(dim.dim_value());
    }
    if (!fully_known) {
      continue;
    }

    // Build the constant tensor with the output's real dtype and shape. The
    // propagated data is a flat sequence, so require a fully static shape whose
    // element count matches what was propagated.
    onnx::TensorProto::DataType elem_type;
    std::vector<int64_t> dims;
    if (!GetStaticIntTensorInfo(type_map, output, elem_type, dims)) {
      continue;
    }
    int64_t element_count = 1;
    for (int64_t d : dims) {
      element_count *= d;
    }
    if (element_count != static_cast<int64_t>(values.size())) {
      continue;
    }

    onnx::TensorProto tp;
    tp.set_data_type(elem_type);
    for (int64_t d : dims) {
      tp.add_dims(d);
    }
    if (elem_type == onnx::TensorProto::INT64) {
      for (int64_t v : values) {
        tp.add_int64_data(v);
      }
    } else {
      for (int64_t v : values) {
        tp.add_int32_data(static_cast<int32_t>(v));
      }
    }
    folded_values.emplace(output, std::move(tp));
  }

  if (folded_values.empty()) {
    restore();
    return;
  }

  // Rewrite each foldable node into a `Constant` node in the same position,
  // keeping the graph topologically sorted. Emitting a `Constant` node (rather
  // than injecting an initializer) leaves the value in producer form, so the
  // ordinary constant folder and optimizer decide how to materialize it.
  google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
  original_nodes.Swap(model.mutable_graph()->mutable_node());
  for (auto& node : original_nodes) {
    auto iter = node.output_size() == 1 ? folded_values.find(node.output(0))
                                        : folded_values.end();
    if (iter == folded_values.end()) {
      *model.mutable_graph()->add_node() = std::move(node);
      continue;
    }
    onnx::NodeProto* constant = model.mutable_graph()->add_node();
    constant->set_name(node.name());
    constant->set_op_type("Constant");
    constant->add_output(iter->first);
    onnx::AttributeProto* attr = constant->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    *attr->mutable_t() = std::move(iter->second);
  }
}

onnx::ModelProto _FoldConstant(const ModelExecutor& executor,
                               const onnx::ModelProto& model) {
  const auto& tmp = model;
  {
    onnx::ModelProto model;
    model.CopyFrom(tmp);
    auto [const_nodes, non_const_nodes] = GetConstantNodes(model);
    (void)non_const_nodes;
    // Outputs of const nodes that were successfully folded into initializers.
    std::set<std::string> folded_outputs;
    for (const auto& x : const_nodes) {
      try {
        RunOpAndAddInitializer(executor, model, x);
        for (const auto& output : x.output()) {
          folded_outputs.insert(output);
        }
      } catch (const std::exception& e) {
        std::cerr << "WARNING: failed to run \"" << x.op_type() <<
          "\" op (name is \"" << x.name() << "\"), skip... " << e.what() << std::endl;
      }
    }
    // Rebuild the node list in its original topological order, dropping only
    // the const nodes that were successfully folded into initializers. A const
    // node that failed to fold must keep its original position: appending it to
    // the end can place it after a non-const consumer (e.g. a Loop reading a
    // SequenceEmpty output), which breaks topological sorting and makes the
    // resulting model fail onnx's checker (issues #238, #335, #352).
    google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
    original_nodes.Swap(model.mutable_graph()->mutable_node());
    for (auto& node : original_nodes) {
      const bool folded = node.output_size() > 0 &&
                          folded_outputs.count(node.output(0)) > 0;
      if (!folded) {
        *model.mutable_graph()->add_node() = std::move(node);
      }
    }
    // Drop initializers left dangling by folding so the intermediate model does
    // not balloon in size (issue #174).
    return EliminateUnusedInitializer(model);
  }
}

onnx::ModelProto Optimize(const onnx::ModelProto& model) {
  return onnx::optimization::OptimizeFixed(model, config.optimizer_passes);
}

// A 128-bit fingerprint of a model, used by FixedPointFn to detect when an
// iteration stopped changing the model without keeping a second full ModelProto
// around just for the comparison. Two models with the same fingerprint are
// treated as equal; the odds of a false match are ~2^-128 per comparison, and a
// false match would only stop simplification one round early (the model stays
// valid), never produce an incorrect model.
struct ModelFingerprint {
  uint64_t h1;
  uint64_t h2;
  bool operator==(const ModelFingerprint& other) const {
    return h1 == other.h1 && h2 == other.h2;
  }
};

ModelFingerprint Fingerprint(const onnx::ModelProto& model) {
  // ModelProto contains no protobuf ``map<>`` fields, so serialization order is
  // stable and equal models serialize to identical bytes.
  const std::string bytes = model.SerializeAsString();
  // Two independent rolling hashes (FNV-1a and a splitmix-style mix) combined
  // into a 128-bit value.
  uint64_t h1 = 1469598103934665603ULL;  // FNV-1a offset basis
  uint64_t h2 = 0;
  for (unsigned char c : bytes) {
    h1 = (h1 ^ c) * 1099511628211ULL;  // FNV-1a prime
    h2 = (h2 + c) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
  h2 ^= bytes.size();
  return {h1, h2};
}

// Alternately apply ``f1`` and ``f2`` until the model stops changing (a joint
// fixed point) or ``max_iters`` alternations elapse. Each application produces a
// fresh model, so ``model`` is move-assigned in place and only a single
// ModelProto is held live across the loop; convergence is detected by comparing
// the fingerprints of consecutive states rather than keeping the previous
// ModelProto for a ``MessageDifferencer::Equals`` call. This mirrors the
// original consecutive-pair comparison exactly -- it stops as soon as the last
// applied function left the model unchanged -- while roughly halving the number
// of full model copies held at once (which matters because these fixed points
// nest).
// The transforms mutate the model in place (``std::function<void(T&)>``), so a
// transform that already works in place (e.g. ``_InferShapes``) makes no copy
// at all, and one that must build a fresh model (e.g. ``Optimize``, whose
// underlying ``OptimizeFixed`` returns a new proto) move-assigns it back. The
// returned function likewise mutates in place, so it composes when these fixed
// points nest and a single ModelProto is threaded through the whole thing.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    size_t _max_iters = max_iters;
    f1(model);
    ModelFingerprint fp_prev = Fingerprint(model);
    f2(model);
    ModelFingerprint fp_cur = Fingerprint(model);
    while (_max_iters-- > 0) {
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f1(model);
      fp_prev = fp_cur;
      fp_cur = Fingerprint(model);
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f2(model);
      fp_prev = fp_cur;
      fp_cur = Fingerprint(model);
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// A no-op in-place transform (mutates nothing), used when shape inference or
// constant folding is disabled.
void Identity(onnx::ModelProto&) {}

// Recursively collect the op types of operators that live in ONNX's *default*
// domain but have no registered schema. These are custom operators -- most
// commonly TensorRT plugins such as ``BatchedNMS_TRT`` or ``EfficientNMS_TRT``
// -- that were exported into the default domain instead of a vendor-specific
// one.
//
// Custom ops that already live in a non-default domain (e.g. ``com.microsoft``
// or ``TRT``) are intentionally ignored: onnx::checker::check_model already
// tolerates unknown ops in non-standard domains, which is exactly the manual
// workaround reported in GitHub issue #220.
void CollectCustomDefaultDomainOps(const onnx::GraphProto& graph,
                                   int default_opset_version,
                                   std::set<std::string>& custom_ops) {
  for (const auto& node : graph.node()) {
    const std::string& domain = node.domain();
    const bool is_default_domain = domain.empty() || domain == "ai.onnx";
    if (is_default_domain &&
        onnx::OpSchemaRegistry::Schema(node.op_type(), default_opset_version,
                                       /*domain=*/"") == nullptr) {
      custom_ops.insert(node.op_type());
    }
    // Recurse into subgraphs held in node attributes (If/Loop/Scan bodies).
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectCustomDefaultDomainOps(attr.g(), default_opset_version, custom_ops);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectCustomDefaultDomainOps(subgraph, default_opset_version, custom_ops);
      }
    }
  }
}

// Register a permissive placeholder schema for every default-domain custom op
// found in ``model``. Without a schema, onnx::checker::check_model rejects the
// model with "No Op registered for <op> with domain_version of <n>" and
// simplification never even starts (GitHub issues #107 and #220). The
// placeholder accepts any number of inputs/outputs of any tensor type and any
// attributes, so the checker passes and the op is preserved untouched through
// simplification. It carries no shape/type inference function, so shape
// inference simply flows past the op as before.
void RegisterCustomDefaultDomainOpSchemas(const onnx::ModelProto& model) {
  int default_opset_version = 1;
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      default_opset_version =
          std::max(default_opset_version, static_cast<int>(opset.version()));
    }
  }

  std::set<std::string> custom_ops;
  CollectCustomDefaultDomainOps(model.graph(), default_opset_version, custom_ops);

  for (const auto& op_type : custom_ops) {
    onnx::OpSchema schema;
    schema.SetName(op_type)
        .SetDomain("")
        .SinceVersion(1)
        .SetDoc(
            "Placeholder schema registered by onnxsim for a custom operator "
            "(e.g. a TensorRT plugin) exported into the default ONNX domain, so "
            "that the model passes validation and is simplified with the "
            "operator preserved unchanged.")
        .Input(0, "inputs", "Variadic inputs of the custom operator.", "T",
               onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
               /*min_arity=*/0)
        .Output(0, "outputs", "Variadic outputs of the custom operator.", "T",
                onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
                /*min_arity=*/0)
        .TypeConstraint("T", onnx::OpSchema::all_tensor_types(),
                        "Allow inputs and outputs of any tensor type.")
        // Custom ops carry arbitrary, plugin-specific attributes; accept them
        // all rather than trying to enumerate them.
        .AllowUncheckedAttributes();
    // Never fail or throw: a duplicate registration (e.g. simplifying two models
    // that use the same custom op in one process) is a harmless no-op.
    onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                         /*fail_duplicate_schema=*/false,
                         /*fail_with_exception=*/false);
  }
}

// Collect the names of every node in `graph`, descending into subgraphs held in
// node attributes (If/Loop/Scan bodies). Used to de-duplicate the names later
// assigned to nameless nodes.
void CollectNodeNames(const onnx::GraphProto& graph,
                      std::set<std::string>& names) {
  for (const auto& node : graph.node()) {
    if (!node.name().empty()) {
      names.insert(node.name());
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectNodeNames(attr.g(), names);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectNodeNames(subgraph, names);
      }
    }
  }
}

// Give a unique, deterministic name to every node that has none. Nodes without
// a name survive simplification unnamed -- either because they were nameless in
// the input model or because an onnx-optimizer pass created a replacement node
// without setting a name -- which trips up downstream tooling that keys on node
// names (issue #269). Each generated name is derived from the op type plus a
// running counter and de-duplicated against every name already present in the
// graph (including names generated earlier in this pass). Subgraphs are handled
// recursively so nodes inside If/Loop/Scan bodies are named too.
void AssignMissingNodeNames(onnx::GraphProto& graph,
                            std::set<std::string>& used_names,
                            size_t& counter) {
  for (auto& node : *graph.mutable_node()) {
    if (node.name().empty()) {
      std::string name;
      do {
        name = node.op_type() + "_" + std::to_string(counter++);
      } while (used_names.count(name) > 0);
      used_names.insert(name);
      node.set_name(name);
    }
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        AssignMissingNodeNames(*attr.mutable_g(), used_names, counter);
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        AssignMissingNodeNames(subgraph, used_names, counter);
      }
    }
  }
}

// Assign names to any nodes left nameless after simplification (issue #269).
void AssignMissingNodeNames(onnx::ModelProto& model) {
  std::set<std::string> used_names;
  CollectNodeNames(model.graph(), used_names);
  size_t counter = 0;
  AssignMissingNodeNames(*model.mutable_graph(), used_names, counter);
}

void Check(const onnx::ModelProto& model) { onnx::checker::check_model(model); }

onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold) {
  // Make shape inference aware of ONNX Runtime's quantized contrib operators
  // (QLinearAdd and friends) so shape deduction does not stop at them.
  onnxsim::RegisterContribOpSchemas();
  // Register permissive placeholder schemas for custom ops exported into the
  // default ONNX domain (e.g. TensorRT plugins such as BatchedNMS_TRT) so the
  // checker below does not reject the model (GitHub issues #107, #220).
  RegisterCustomDefaultDomainOpSchemas(model);

  Check(model);

  config.tensor_size_threshold = tensor_size_threshold;
  config.optimizer_passes.clear();
  // skip_optimizers == nullopt means skiping all optimizers, so
  // config.optimizer_passes is empty
  if (skip_optimizers) {
    std::vector<std::string> passes;
    const auto all_passes = onnx::optimization::GetFuseAndEliminationPass();
    for (const auto& pass : all_passes) {
      if (std::find(skip_optimizers->begin(), skip_optimizers->end(), pass) ==
          skip_optimizers->end()) {
        passes.push_back(pass);
      }
    }
    config.optimizer_passes = passes;
  }

  // Every transform mutates the model in place.
  using ModelFn = std::function<void(onnx::ModelProto&)>;
  ModelFn FoldConstant;
  if (constant_folding) {
    FoldConstant = [&executor](onnx::ModelProto& model) {
      // Partial shape evaluation (issue #139) turns Shape/Gather-on-shape into
      // constants that the ordinary constant folder can then propagate.
      _EvalPartialShape(model);
      model = _FoldConstant(executor, model);
    };
  } else {
    FoldConstant = Identity;
  }
  ModelFn InferShapes = shape_inference ? _InferShapes : Identity;
  // ``Optimize`` builds a fresh model (``OptimizeFixed`` returns a new proto),
  // so wrap it as an in-place transform that move-assigns the result back.
  ModelFn OptimizeInPlace = [](onnx::ModelProto& model) {
    model = Optimize(model);
  };

  int fixed_point_iters =
      std::getenv("ONNXSIM_FIXED_POINT_ITERS")
          ? std::atoi(std::getenv("ONNXSIM_FIXED_POINT_ITERS"))
          : 50;

  ModelFn OptAndShape =
      FixedPointFn(InferShapes, OptimizeInPlace, fixed_point_iters);
  bool converged = false;
  ModelFn OptAndShapeAndFold =
      FixedPointFn(OptAndShape, FoldConstant, fixed_point_iters, &converged);
  // The fixed points mutate in place, so make one working copy of the (const)
  // input model and simplify it in place.
  onnx::ModelProto sim_model = model;
  OptAndShapeAndFold(sim_model);
  // Simplification (and some onnx-optimizer passes) can leave nodes without a
  // name; assign unique names to them so downstream tools that key on node
  // names keep working (issue #269).
  AssignMissingNodeNames(sim_model);
  Check(sim_model);
  if (!converged) {
    std::cout << "WARNING: the simplification stopped because of timeout. "
                 "Please set environment variable `ONNXSIM_FIXED_POINT_ITERS` "
                 "to a number higher than "
              << fixed_point_iters << "if you want further simplification."
              << std::endl;
  }
  return sim_model;
}

void SimplifyPath(const ModelExecutor& executor, const std::string& in_path,
                  const std::string& out_path,
                  std::optional<std::vector<std::string>> skip_optimizers,
                  bool constant_folding, bool shape_inference,
                  size_t tensor_size_threshold) {
  onnx::ModelProto model;
  onnx::optimization::loadModel(&model, in_path, true);

  model = Simplify(executor, model, skip_optimizers, constant_folding,
                   shape_inference, tensor_size_threshold);

  onnx::optimization::saveModel(&model, out_path, true, "");
}
