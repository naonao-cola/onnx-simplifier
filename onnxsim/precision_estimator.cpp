#include "precision_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "dlpack_dtype.h"
#include "onnx/shape_inference/implementation.h"

namespace onnxsim {
namespace {

// INT32_MAX // (127 * 255) -- see precision_estimator.py's docstring, point
// 1. A literal formula, not a hardcoded constant, so it stays obviously in
// sync with passes/quantize_matmul_common.h's MaxSafeInt32ReductionDepth(),
// which onnxsim.quantize_dynamic actually enforces -- exactly the same
// reasoning the Python module gives for the same choice.
constexpr int64_t MaxSafeInt32ReductionDepth() {
  return (std::numeric_limits<int32_t>::max()) / (127 * 255);
}

// An 8-bit symmetric quantizer has floor(127) positive levels (scale =
// max|w| / 127); a channel's max(|w|) / median(|w|) ratio past this leaves
// its median-magnitude weight within one quantization step of zero.
constexpr double kOutlierRatioThreshold = 127.0;

// See precision_estimator.py's MAX_EXACT_FLOAT32_REDUCTION_DEPTH docstring.
constexpr int64_t MaxExactFloat32ReductionDepth() {
  return (int64_t{1} << 24) / (127 * 255);
}

// sqrt(12): see precision_estimator.py's _UNIFORM_QUANTIZER_NOISE_DIVISOR.
const double kUniformQuantizerNoiseDivisor = 127.0 * std::sqrt(12.0);

// Ops whose output range is fixed by the op itself, for any input -- not a
// property of the data, so it needs no calibration run to know. See
// precision_estimator.py's FIXED_ACTIVATION_RANGES.
const std::unordered_map<std::string, std::pair<double, double>>&
FixedActivationRanges() {
  static const std::unordered_map<std::string, std::pair<double, double>> kMap = {
      {"Sigmoid", {0.0, 1.0}},
      {"HardSigmoid", {0.0, 1.0}},
      {"Tanh", {-1.0, 1.0}},
      {"Softmax", {0.0, 1.0}},
  };
  return kMap;
}

// Reads a FLOAT TensorProto's values into a flat, host-order vector,
// regardless of whether it is stored as raw bytes (little-endian on the wire
// regardless of host) or the typed float_data field (already host-order,
// decoded by protobuf itself). Returns an empty vector for any other dtype.
std::vector<float> ReadFloatTensorFlat(const onnx::TensorProto& t) {
  if (t.data_type() != onnx::TensorProto_DataType_FLOAT) {
    return {};
  }
  if (t.has_raw_data()) {
    const std::string& raw = t.raw_data();
    const size_t n = raw.size() / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), raw.data(), n * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(
          reinterpret_cast<uint8_t*>(out.data()), out.size() * sizeof(float),
          sizeof(float));
    }
    return out;
  }
  return {t.float_data().begin(), t.float_data().end()};
}

int64_t GetAttrInt(const onnx::NodeProto& node, const std::string& name,
                   int64_t dflt) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) {
      return attr.i();
    }
  }
  return dflt;
}

std::optional<double> GetAttrFloat(const onnx::NodeProto& node,
                                   const std::string& name) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) {
      return static_cast<double>(attr.f());
    }
  }
  return std::nullopt;
}

// max(|w|) / median(|w|) over one channel's weights, ignoring zeros -- see
// precision_estimator.py's _channel_outlier_ratio for why zeros are excluded
// (a pruned channel's sparsity shouldn't read as an outlier). NaN when fewer
// than two nonzero weights remain (the ratio isn't meaningful).
double ChannelOutlierRatio(std::vector<float> abs_weights_for_channel) {
  std::vector<float> nonzero;
  nonzero.reserve(abs_weights_for_channel.size());
  for (float v : abs_weights_for_channel) {
    if (v > 0.0f) nonzero.push_back(v);
  }
  if (nonzero.size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const float peak = *std::max_element(nonzero.begin(), nonzero.end());
  // np.median: sort and take the middle element(s), averaging the two
  // central elements for an even count.
  std::sort(nonzero.begin(), nonzero.end());
  const size_t n = nonzero.size();
  const double med = (n % 2 == 1)
                         ? static_cast<double>(nonzero[n / 2])
                         : (static_cast<double>(nonzero[n / 2 - 1]) +
                            static_cast<double>(nonzero[n / 2])) /
                               2.0;
  if (med == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(peak) / med;
}

// Mirrors precision_estimator.py's _recommendation.
std::string BuildRecommendation(bool safe, bool float32_exact,
                                bool outlier_risk,
                                const std::string& activation_producer_op,
                                bool has_activation_range, double lo,
                                double hi) {
  if (!safe) {
    return "reduction depth exceeds the int32-safe bound: "
           "onnxsim.quantize_dynamic already skips this node; INT8 with a "
           "wider (int64) accumulator, or splitting the reduction (blockwise "
           "quantization), would be needed";
  }
  std::vector<std::string> notes;
  if (!float32_exact) {
    notes.push_back(
        "the accumulator's Cast<float> is not bit-exact past this reduction "
        "depth (float32's 24-bit mantissa), though the rounding involved "
        "(~2**-24 relative) is far below INT8's own ~1/127 quantization "
        "error and does not change the recommendation");
  }
  if (outlier_risk) {
    notes.push_back(
        "a channel's outliers dominate its scale: per-group quantization or "
        "INT16 would preserve more resolution for this node's "
        "typical-magnitude weights");
  }
  if (has_activation_range) {
    std::ostringstream os;
    os << "the activation input comes from " << activation_producer_op
       << ", whose output is always in [" << lo << ", " << hi
       << "] regardless of input data: a fixed static scale would quantize "
          "it exactly, without calibration data or quantize_dynamic's "
          "runtime DynamicQuantizeLinear";
    notes.push_back(os.str());
  }
  if (notes.empty()) {
    return "INT8 (onnxsim.quantize_dynamic's scheme) looks safe and "
           "well-resolved";
  }
  std::ostringstream os;
  os << "int32-safe, but ";
  for (size_t i = 0; i < notes.size(); ++i) {
    if (i > 0) os << "; ";
    os << notes[i];
  }
  return os.str();
}

// The analytically-known output range of `producer`, if any. `initializers`
// is used to resolve a Clip node's (opset >= 11) min/max inputs.
bool ActivationRange(
    const onnx::NodeProto* producer,
    const std::unordered_map<std::string, const onnx::TensorProto*>&
        initializers,
    std::string* op_type, double* lo, double* hi) {
  if (producer == nullptr) return false;
  const auto& fixed = FixedActivationRanges();
  auto it = fixed.find(producer->op_type());
  if (it != fixed.end()) {
    *op_type = producer->op_type();
    *lo = it->second.first;
    *hi = it->second.second;
    return true;
  }
  if (producer->op_type() != "Clip") return false;
  double clip_lo = -std::numeric_limits<double>::infinity();
  double clip_hi = std::numeric_limits<double>::infinity();
  for (const auto& attr : producer->attribute()) {  // opset < 11
    if (attr.name() == "min") clip_lo = attr.f();
    if (attr.name() == "max") clip_hi = attr.f();
  }
  auto read_scalar = [&](const std::string& name,
                         double* out) -> bool {
    auto init_it = initializers.find(name);
    if (init_it == initializers.end()) return false;
    std::vector<float> vals = ReadFloatTensorFlat(*init_it->second);
    if (vals.empty()) return false;
    *out = static_cast<double>(vals[0]);
    return true;
  };
  if (producer->input_size() > 1 && !producer->input(1).empty()) {
    read_scalar(producer->input(1), &clip_lo);  // opset >= 11
  }
  if (producer->input_size() > 2 && !producer->input(2).empty()) {
    read_scalar(producer->input(2), &clip_hi);
  }
  if (std::isinf(clip_lo) || std::isinf(clip_hi)) return false;
  *op_type = "Clip";
  *lo = clip_lo;
  *hi = clip_hi;
  return true;
}

std::optional<WeightPrecisionEstimate> EstimateMatMulGemm(
    const onnx::NodeProto& node,
    const std::unordered_map<std::string, const onnx::TensorProto*>&
        initializers,
    const std::unordered_map<std::string, const onnx::NodeProto*>& producer) {
  if (node.op_type() != "MatMul" && node.op_type() != "Gemm") return std::nullopt;
  if (node.input_size() < 2) return std::nullopt;
  auto init_it = initializers.find(node.input(1));
  if (init_it == initializers.end()) return std::nullopt;
  const onnx::TensorProto& w_t = *init_it->second;
  if (w_t.data_type() != onnx::TensorProto_DataType_FLOAT ||
      w_t.dims_size() != 2) {
    return std::nullopt;
  }

  const bool transposed =
      node.op_type() == "Gemm" && GetAttrInt(node, "transB", 0) != 0;
  if (node.op_type() == "Gemm") {
    const int64_t trans_a = GetAttrInt(node, "transA", 0);
    const std::optional<double> alpha = GetAttrFloat(node, "alpha");
    if (trans_a != 0 || (alpha.has_value() && *alpha != 1.0)) {
      return std::nullopt;
    }
  }

  const int64_t dim0 = w_t.dims(0);
  const int64_t dim1 = w_t.dims(1);
  const int64_t k = transposed ? dim1 : dim0;
  const int64_t n = transposed ? dim0 : dim1;
  const int channel_axis = transposed ? 0 : 1;

  const std::vector<float> data = ReadFloatTensorFlat(w_t);
  std::vector<double> finite_ratios;
  for (int64_t c = 0; c < n; ++c) {
    std::vector<float> channel;
    if (channel_axis == 1) {
      channel.reserve(static_cast<size_t>(dim0));
      for (int64_t i = 0; i < dim0; ++i) {
        channel.push_back(std::fabs(data[static_cast<size_t>(i * dim1 + c)]));
      }
    } else {
      channel.reserve(static_cast<size_t>(dim1));
      for (int64_t j = 0; j < dim1; ++j) {
        channel.push_back(std::fabs(data[static_cast<size_t>(c * dim1 + j)]));
      }
    }
    const double ratio = ChannelOutlierRatio(std::move(channel));
    if (!std::isnan(ratio)) finite_ratios.push_back(ratio);
  }
  const double max_ratio = finite_ratios.empty()
                                ? std::numeric_limits<double>::quiet_NaN()
                                : *std::max_element(finite_ratios.begin(),
                                                    finite_ratios.end());
  const bool safe = k <= MaxSafeInt32ReductionDepth();
  const bool float32_exact = k <= MaxExactFloat32ReductionDepth();
  const bool outlier_risk =
      !finite_ratios.empty() && max_ratio > kOutlierRatioThreshold;

  const onnx::NodeProto* act_producer = nullptr;
  if (node.input_size() > 0) {
    auto p_it = producer.find(node.input(0));
    if (p_it != producer.end()) act_producer = p_it->second;
  }
  std::string act_op;
  double act_lo = 0.0, act_hi = 0.0;
  const bool has_act_range =
      ActivationRange(act_producer, initializers, &act_op, &act_lo, &act_hi);

  WeightPrecisionEstimate est;
  est.node_name = !node.name().empty()
                      ? node.name()
                      : (node.op_type() + "(" + node.input(1) + ")");
  est.op_type = node.op_type();
  est.reduction_depth = k;
  est.num_channels = n;
  est.int32_accumulator_safe = safe;
  est.float32_cast_exact = float32_exact;
  est.max_outlier_ratio = max_ratio;
  est.outlier_risk = outlier_risk;
  est.activation_producer_op = has_act_range ? act_op : "";
  est.has_activation_range = has_act_range;
  est.activation_range_lo = act_lo;
  est.activation_range_hi = act_hi;
  est.recommendation = BuildRecommendation(safe, float32_exact, outlier_risk,
                                           act_op, has_act_range, act_lo,
                                           act_hi);
  return est;
}

std::optional<WeightPrecisionEstimate> EstimateConv(
    const onnx::NodeProto& node,
    const std::unordered_map<std::string, const onnx::TensorProto*>&
        initializers,
    const std::unordered_map<std::string, const onnx::NodeProto*>& producer) {
  if (node.op_type() != "Conv") return std::nullopt;
  if (node.input_size() < 2) return std::nullopt;
  auto init_it = initializers.find(node.input(1));
  if (init_it == initializers.end()) return std::nullopt;
  const onnx::TensorProto& w_t = *init_it->second;
  if (w_t.data_type() != onnx::TensorProto_DataType_FLOAT ||
      w_t.dims_size() < 3) {
    return std::nullopt;
  }

  const int64_t cout = w_t.dims(0);
  int64_t inner = 1;
  for (int i = 1; i < w_t.dims_size(); ++i) inner *= w_t.dims(i);
  const std::vector<float> data = ReadFloatTensorFlat(w_t);

  std::vector<double> finite_ratios;
  for (int64_t c = 0; c < cout; ++c) {
    std::vector<float> channel(static_cast<size_t>(inner));
    for (int64_t j = 0; j < inner; ++j) {
      channel[static_cast<size_t>(j)] =
          std::fabs(data[static_cast<size_t>(c * inner + j)]);
    }
    const double ratio = ChannelOutlierRatio(std::move(channel));
    if (!std::isnan(ratio)) finite_ratios.push_back(ratio);
  }
  const double max_ratio = finite_ratios.empty()
                                ? std::numeric_limits<double>::quiet_NaN()
                                : *std::max_element(finite_ratios.begin(),
                                                    finite_ratios.end());
  const bool safe = inner <= MaxSafeInt32ReductionDepth();
  const bool float32_exact = inner <= MaxExactFloat32ReductionDepth();
  const bool outlier_risk =
      !finite_ratios.empty() && max_ratio > kOutlierRatioThreshold;

  const onnx::NodeProto* act_producer = nullptr;
  if (node.input_size() > 0) {
    auto p_it = producer.find(node.input(0));
    if (p_it != producer.end()) act_producer = p_it->second;
  }
  std::string act_op;
  double act_lo = 0.0, act_hi = 0.0;
  const bool has_act_range =
      ActivationRange(act_producer, initializers, &act_op, &act_lo, &act_hi);

  WeightPrecisionEstimate est;
  est.node_name = !node.name().empty() ? node.name()
                                       : ("Conv(" + node.input(1) + ")");
  est.op_type = "Conv";
  est.reduction_depth = inner;
  est.num_channels = cout;
  est.int32_accumulator_safe = safe;
  est.float32_cast_exact = float32_exact;
  est.max_outlier_ratio = max_ratio;
  est.outlier_risk = outlier_risk;
  est.activation_producer_op = has_act_range ? act_op : "";
  est.has_activation_range = has_act_range;
  est.activation_range_lo = act_lo;
  est.activation_range_hi = act_hi;
  est.recommendation = BuildRecommendation(safe, float32_exact, outlier_risk,
                                           act_op, has_act_range, act_lo,
                                           act_hi);
  return est;
}

// A tensor's statically-known dims, one optional<int64_t> per dimension (a
// dim_param or otherwise-unresolved dim reads as nullopt); nullopt for the
// whole vector means the tensor's shape wasn't found at all.
using StaticShape = std::vector<std::optional<int64_t>>;

std::unordered_map<std::string, StaticShape> CollectStaticShapes(
    const onnx::ModelProto& model) {
  onnx::ModelProto inferred = model;
  try {
    onnx::shape_inference::InferShapes(inferred);
  } catch (...) {
    inferred = model;
  }
  std::unordered_map<std::string, StaticShape> shapes;
  const onnx::GraphProto& graph = inferred.graph();
  auto collect = [&](const auto& value_infos) {
    for (const auto& vi : value_infos) {
      if (!vi.type().has_tensor_type()) continue;
      const auto& tt = vi.type().tensor_type();
      if (!tt.has_shape()) continue;
      StaticShape dims;
      dims.reserve(static_cast<size_t>(tt.shape().dim_size()));
      for (const auto& d : tt.shape().dim()) {
        dims.push_back(d.has_dim_value() ? std::optional<int64_t>(d.dim_value())
                                         : std::nullopt);
      }
      shapes[vi.name()] = std::move(dims);
    }
  };
  collect(graph.input());
  collect(graph.output());
  collect(graph.value_info());
  for (const auto& init : graph.initializer()) {
    StaticShape dims;
    dims.reserve(static_cast<size_t>(init.dims_size()));
    for (int64_t d : init.dims()) dims.push_back(d);
    shapes[init.name()] = std::move(dims);
  }
  return shapes;
}

std::optional<AttentionPrecisionEstimate> EstimateAttention(
    const onnx::NodeProto& node,
    const std::unordered_map<std::string, StaticShape>& shapes) {
  if (node.op_type() != "Attention" || node.input_size() < 3) {
    return std::nullopt;
  }
  const StaticShape* q_shape = nullptr;
  const StaticShape* k_shape = nullptr;
  {
    auto it = shapes.find(node.input(0));
    if (it != shapes.end()) q_shape = &it->second;
  }
  {
    auto it = shapes.find(node.input(1));
    if (it != shapes.end()) k_shape = &it->second;
  }

  std::optional<int64_t> q_heads;
  {
    const int64_t v = GetAttrInt(node, "q_num_heads", 0);
    if (v != 0) q_heads = v;
  }
  std::optional<int64_t> kv_heads;
  {
    const int64_t v = GetAttrInt(node, "kv_num_heads", 0);
    if (v != 0) kv_heads = v;
  }
  std::optional<int64_t> head_dim;

  if (q_shape != nullptr && q_shape->size() == 4 && (*q_shape)[3].has_value()) {
    if ((*q_shape)[1].has_value()) q_heads = (*q_shape)[1];
    head_dim = (*q_shape)[3];
    if (k_shape != nullptr && k_shape->size() == 4 &&
        (*k_shape)[1].has_value()) {
      kv_heads = (*k_shape)[1];
    }
  } else if (q_shape != nullptr && k_shape != nullptr && q_shape->size() == 3 &&
            k_shape->size() == 3 && kv_heads.has_value() &&
            (*k_shape)[2].has_value()) {
    head_dim = *(*k_shape)[2] / *kv_heads;
  }

  const std::optional<double> default_scale =
      head_dim.has_value() ? std::optional<double>(1.0 / std::sqrt(
                                 static_cast<double>(*head_dim)))
                           : std::nullopt;
  const std::optional<double> actual_scale = GetAttrFloat(node, "scale");
  int matches = -1;
  if (default_scale.has_value() && actual_scale.has_value()) {
    // math.isclose(..., rel_tol=1e-3)
    const double a = *default_scale, b = *actual_scale;
    matches = (std::fabs(a - b) <=
              1e-3 * std::max(std::fabs(a), std::fabs(b)))
                 ? 1
                 : 0;
  }

  std::string recommendation;
  if (!head_dim.has_value()) {
    recommendation =
        "head_dim could not be determined statically (dynamic/unknown shape "
        "and no q_num_heads/kv_num_heads attributes) -- no estimate "
        "available";
  } else if (matches == 0) {
    std::ostringstream os;
    os << "scale=" << *actual_scale
       << " attribute does not match the canonical 1/sqrt(head_dim)="
       << *default_scale
       << ": pre-softmax QK^T logits will grow unnormalized with head_dim, "
          "risking saturation/overflow in a low-precision (e.g. fp16) "
          "softmax";
    recommendation = os.str();
  } else {
    recommendation =
        "scale normalizes QK^T's head_dim-dependent growth as expected; no "
        "int8 accumulator applies here since Q/K/V are runtime activations, "
        "not a static weight";
  }

  AttentionPrecisionEstimate est;
  est.node_name = !node.name().empty() ? node.name() : "Attention";
  est.has_num_query_heads = q_heads.has_value();
  est.num_query_heads = q_heads.value_or(0);
  est.has_num_kv_heads = kv_heads.has_value();
  est.num_kv_heads = kv_heads.value_or(0);
  est.has_head_dim = head_dim.has_value();
  est.head_dim = head_dim.value_or(0);
  est.default_scale =
      default_scale.value_or(std::numeric_limits<double>::quiet_NaN());
  est.actual_scale =
      actual_scale.value_or(std::numeric_limits<double>::quiet_NaN());
  est.scale_matches_default = matches;
  est.recommendation = recommendation;
  return est;
}

}  // namespace

void EstimateQuantizationPrecision(
    const onnx::ModelProto& model,
    std::vector<WeightPrecisionEstimate>* weight_estimates,
    std::vector<AttentionPrecisionEstimate>* attention_estimates) {
  weight_estimates->clear();
  attention_estimates->clear();

  std::unordered_map<std::string, const onnx::TensorProto*> initializers;
  for (const auto& init : model.graph().initializer()) {
    initializers[init.name()] = &init;
  }
  std::unordered_map<std::string, const onnx::NodeProto*> producer;
  for (const auto& node : model.graph().node()) {
    for (const auto& out : node.output()) {
      if (!out.empty()) producer[out] = &node;
    }
  }

  // Attention needs shape inference (for head_dim); only run it if the model
  // actually has an Attention node, since it's the one non-trivial cost here.
  bool has_attention = false;
  for (const auto& node : model.graph().node()) {
    if (node.op_type() == "Attention") {
      has_attention = true;
      break;
    }
  }
  std::unordered_map<std::string, StaticShape> shapes;
  if (has_attention) {
    shapes = CollectStaticShapes(model);
  }

  for (const auto& node : model.graph().node()) {
    if (node.op_type() == "MatMul" || node.op_type() == "Gemm") {
      auto est = EstimateMatMulGemm(node, initializers, producer);
      if (est.has_value()) weight_estimates->push_back(std::move(*est));
    } else if (node.op_type() == "Conv") {
      auto est = EstimateConv(node, initializers, producer);
      if (est.has_value()) weight_estimates->push_back(std::move(*est));
    } else if (node.op_type() == "Attention") {
      auto est = EstimateAttention(node, shapes);
      if (est.has_value()) attention_estimates->push_back(std::move(*est));
    }
  }
}

ModelQuantizationEstimate EstimateModelQuantizationDrop(
    const onnx::ModelProto& model) {
  ModelQuantizationEstimate out;
  EstimateQuantizationPrecision(model, &out.weight_estimates,
                                &out.attention_estimates);

  std::vector<double> outlier_ratios;
  double sum_squared_errors = 0.0;
  for (const auto& est : out.weight_estimates) {
    if (!est.int32_accumulator_safe) {
      out.unsafe_nodes.push_back(est.node_name);
      continue;
    }
    if (est.outlier_risk) {
      out.outlier_risk_nodes.push_back(est.node_name);
    }
    const double ratio =
        std::isnan(est.max_outlier_ratio) ? 1.0 : est.max_outlier_ratio;
    if (!std::isnan(est.max_outlier_ratio)) {
      outlier_ratios.push_back(est.max_outlier_ratio);
    }
    const double per_node_relative_error = ratio / kUniformQuantizerNoiseDivisor;
    sum_squared_errors += per_node_relative_error * per_node_relative_error;
  }

  out.total_nodes_analyzed = static_cast<int64_t>(out.weight_estimates.size() +
                                                   out.attention_estimates.size());
  out.worst_outlier_ratio = outlier_ratios.empty()
                                ? std::numeric_limits<double>::quiet_NaN()
                                : *std::max_element(outlier_ratios.begin(),
                                                    outlier_ratios.end());
  if (!out.unsafe_nodes.empty()) {
    out.risk_level = "unsafe";
    out.estimated_relative_error = std::numeric_limits<double>::quiet_NaN();
  } else {
    out.risk_level = out.outlier_risk_nodes.empty() ? "safe" : "degraded";
    out.estimated_relative_error = std::sqrt(sum_squared_errors);
  }
  return out;
}

}  // namespace onnxsim
