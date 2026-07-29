#include "model_info.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

// Recursively tally op types, descending into every subgraph carried by a node
// attribute (the ``g`` of e.g. an ``If`` branch, or the ``graphs`` of ``Loop``).
// Every initializer of each graph is counted as a ``Constant``, matching the
// Python ``ModelInfo.get_info``.
void CountGraphOps(const onnx::GraphProto& graph,
                   std::map<std::string, int64_t>& op_nums) {
  for (const auto& node : graph.node()) {
    op_nums[node.op_type()] += 1;
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CountGraphOps(attr.g(), op_nums);
      }
      for (const auto& subgraph : attr.graphs()) {
        CountGraphOps(subgraph, op_nums);
      }
    }
  }
  op_nums["Constant"] += graph.initializer_size();
}

// Bytes of a single tensor's data when it lives in an external file, read from
// the ``length`` entry of its ``external_data`` record. Zero for tensors held
// inline (whose bytes are already part of the graph's serialized size).
int64_t TensorExternalSize(const onnx::TensorProto& tensor) {
  if (tensor.data_location() != onnx::TensorProto::EXTERNAL) {
    return 0;
  }
  for (const auto& entry : tensor.external_data()) {
    if (entry.key() == "length") {
      try {
        return std::stoll(entry.value());
      } catch (...) {
        return 0;
      }
    }
  }
  return 0;
}

// Total external-data bytes across every tensor a graph holds -- initializers as
// well as tensors carried in node attributes -- recursing into subgraphs. This
// mirrors Python's ``_external_data_size`` and never loads the data itself.
int64_t ExternalDataSize(const onnx::GraphProto& graph) {
  int64_t total = 0;
  for (const auto& initializer : graph.initializer()) {
    total += TensorExternalSize(initializer);
  }
  for (const auto& node : graph.node()) {
    for (const auto& attr : node.attribute()) {
      if (attr.has_t()) {
        total += TensorExternalSize(attr.t());
      }
      for (const auto& tensor : attr.tensors()) {
        total += TensorExternalSize(tensor);
      }
      if (attr.has_g()) {
        total += ExternalDataSize(attr.g());
      }
      for (const auto& subgraph : attr.graphs()) {
        total += ExternalDataSize(subgraph);
      }
    }
  }
  return total;
}

// Format a byte count with 1024-based binary units, matching the Python
// ``human_readable_size`` (e.g. 1536 -> "1.5KiB").
std::string HumanReadableSize(int64_t num) {
  double value = static_cast<double>(num);
  static const char* const kUnits[] = {"",   "Ki", "Mi", "Gi",
                                        "Ti", "Pi", "Ei", "Zi"};
  char buf[64];
  for (const char* unit : kUnits) {
    if (std::fabs(value) < 1024.0) {
      std::snprintf(buf, sizeof(buf), "%.1f%sB", value, unit);
      return buf;
    }
    value /= 1024.0;
  }
  std::snprintf(buf, sizeof(buf), "%.1fYiB", value);
  return buf;
}

int64_t OpCount(const std::map<std::string, int64_t>& op_nums,
                const std::string& key) {
  auto it = op_nums.find(key);
  return it != op_nums.end() ? it->second : 0;
}

}  // namespace

ModelInfo GetModelInfo(const onnx::ModelProto& model) {
  ModelInfo info;
  CountGraphOps(model.graph(), info.op_nums);
  // ByteSizeLong() (not the 32-bit ByteSize()) so models above 2GB do not
  // overflow; external tensor data is then added from metadata.
  info.model_size = static_cast<int64_t>(model.graph().ByteSizeLong()) +
                    ExternalDataSize(model.graph());
  return info;
}

std::string FormatSimplifyingInfo(const onnx::ModelProto& model_ori,
                                  const onnx::ModelProto& model_opt) {
  const ModelInfo ori = GetModelInfo(model_ori);
  const ModelInfo opt = GetModelInfo(model_opt);

  // Each row is {name, original, simplified}; the simplified cell gets a
  // trailing " *" when the metric improved (a smaller count / size).
  std::vector<std::array<std::string, 3>> rows;
  rows.push_back({"", "Original Model", "Simplified Model"});

  std::set<std::string> keys;
  for (const auto& entry : ori.op_nums) keys.insert(entry.first);
  for (const auto& entry : opt.op_nums) keys.insert(entry.first);
  for (const auto& key : keys) {
    const int64_t o = OpCount(ori.op_nums, key);
    const int64_t s = OpCount(opt.op_nums, key);
    std::string simplified = std::to_string(s);
    if (s < o) simplified += " *";
    rows.push_back({key, std::to_string(o), simplified});
  }

  std::string size_cell = HumanReadableSize(opt.model_size);
  if (opt.model_size < ori.model_size) size_cell += " *";
  rows.push_back(
      {"Model Size", HumanReadableSize(ori.model_size), size_cell});

  std::array<size_t, 3> width = {0, 0, 0};
  for (const auto& row : rows) {
    for (size_t c = 0; c < 3; ++c) {
      width[c] = std::max(width[c], row[c].size());
    }
  }

  auto border = [&]() {
    std::string line = "+";
    for (size_t c = 0; c < 3; ++c) {
      line.append(width[c] + 2, '-');
      line.push_back('+');
    }
    line.push_back('\n');
    return line;
  };
  auto render = [&](const std::array<std::string, 3>& row) {
    std::string line = "|";
    for (size_t c = 0; c < 3; ++c) {
      line.push_back(' ');
      line.append(row[c]);
      line.append(width[c] - row[c].size(), ' ');
      line.append(" |");
    }
    line.push_back('\n');
    return line;
  };

  std::string out;
  out += border();
  out += render(rows.front());
  out += border();
  for (size_t i = 1; i < rows.size(); ++i) {
    out += render(rows[i]);
  }
  out += border();
  return out;
}
