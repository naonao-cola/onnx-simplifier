// Native onnxsim RPC server for Android (and any Linux): the same wire protocol
// as `python -m onnxsim.rpc server` (see docs/rpc.md), running models with
// onnxruntime's C++ API.
//
//   ./onnxsim_rpc_server --host 127.0.0.1 --port 9090 --key phone --workspace
//   /data/local/tmp/ws adb forward tcp:9090 tcp:9090        # on the host, then
//   onnxsim.rpc.connect("127.0.0.1", 9090)
//
// Frame: <u32 header_len><u32 blob_count><JSON header> then blob_count x (<u64
// len><bytes>), little-endian. Tensors are {"name","dtype","shape"} records
// plus one raw blob each. Security: executes whatever model a client sends;
// binds loopback by default; the key is an identifier, not authentication.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json_mini.h"
#include "onnxruntime_cxx_api.h"
#if defined(__ANDROID__)
#include "nnapi_provider_factory.h"
#endif

namespace {

constexpr uint64_t kMaxBlobBytes = 2ULL * 1024 * 1024 * 1024;
constexpr uint32_t kMaxHeaderBytes = 64u * 1024 * 1024;

struct DType {
  const char* name;
  ONNXTensorElementDataType type;
  size_t size;
};
const DType kDTypes[] = {
    {"float16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
    {"float32", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
    {"float64", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, 8},
    {"int8", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, 1},
    {"int16", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, 2},
    {"int32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
    {"int64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
    {"uint8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1},
    {"uint16", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, 2},
    {"uint32", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, 4},
    {"uint64", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, 8},
    {"bool", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 1},
};

const DType* dtype_by_name(const std::string& name) {
  for (const auto& d : kDTypes)
    if (name == d.name) return &d;
  return nullptr;
}
const DType* dtype_by_type(ONNXTensorElementDataType t) {
  for (const auto& d : kDTypes)
    if (d.type == t) return &d;
  return nullptr;
}

struct Options {
  std::string host = "127.0.0.1";
  int port = 9090;
  std::string key;
  std::string workspace = "onnxsim_rpc_workspace";
};

// ---- socket helpers
// ---------------------------------------------------------------------------

bool recv_exact(int fd, void* buffer, size_t count) {
  auto* p = static_cast<uint8_t*>(buffer);
  while (count) {
    ssize_t n = ::recv(fd, p, count, 0);
    if (n <= 0) return false;
    p += n;
    count -= static_cast<size_t>(n);
  }
  return true;
}
bool send_all(int fd, const void* buffer, size_t count) {
  auto* p = static_cast<const uint8_t*>(buffer);
  while (count) {
    ssize_t n = ::send(fd, p, count, MSG_NOSIGNAL);
    if (n <= 0) return false;
    p += n;
    count -= static_cast<size_t>(n);
  }
  return true;
}

struct Message {
  Json header;
  std::vector<std::vector<uint8_t>> blobs;
};

bool read_message(int fd, Message& msg) {
  uint32_t prefix[2];
  if (!recv_exact(fd, prefix, sizeof prefix)) return false;
  if (prefix[0] > kMaxHeaderBytes || prefix[1] > 1000000) return false;
  std::string header(prefix[0], '\0');
  if (!recv_exact(fd, header.data(), header.size())) return false;
  msg.header = Json::parse(header);
  msg.blobs.clear();
  for (uint32_t k = 0; k < prefix[1]; ++k) {
    uint64_t length;
    if (!recv_exact(fd, &length, sizeof length) || length > kMaxBlobBytes)
      return false;
    msg.blobs.emplace_back(length);
    if (length && !recv_exact(fd, msg.blobs.back().data(), length))
      return false;
  }
  return true;
}

bool write_message(int fd, const Json& header,
                   const std::vector<std::vector<uint8_t>>& blobs = {}) {
  std::string text = header.dump();
  uint32_t prefix[2] = {static_cast<uint32_t>(text.size()),
                        static_cast<uint32_t>(blobs.size())};
  if (!send_all(fd, prefix, sizeof prefix) ||
      !send_all(fd, text.data(), text.size()))
    return false;
  for (const auto& blob : blobs) {
    uint64_t length = blob.size();
    if (!send_all(fd, &length, sizeof length) ||
        (length && !send_all(fd, blob.data(), length)))
      return false;
  }
  return true;
}

// ---- models
// -----------------------------------------------------------------------------------

struct Model {
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names, output_names;
};

struct Tensors {
  std::vector<Json> specs;
  std::vector<std::vector<uint8_t>> blobs;
};

class Server {
 public:
  explicit Server(Options options)
      : options_(std::move(options)),
        env_(ORT_LOGGING_LEVEL_WARNING, "onnxsim_rpc") {}

  Json info() const {
    Json j = Json::Object();
    struct utsname u;
    uname(&u);
    j.set("protocol", Json::Int(1))
        .set("key", Json::String(options_.key))
        .set("server", Json::String("native"));
    j.set("platform", Json::String(std::string(u.sysname) + "-" + u.release +
                                   "-" + u.machine));
    j.set("machine", Json::String(u.machine))
        .set("onnxruntime", Json::String(Ort::GetVersionString()));
    Json providers = Json::Array();
    for (const auto& p : Ort::GetAvailableProviders())
      providers.push(Json::String(p));
    j.set("providers", providers)
        .set("tinygrad", Json())
        .set("onnxsim", Json());
    return j;
  }

  void serve() {
    int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(options_.port));
    if (::inet_pton(AF_INET, options_.host.c_str(), &address.sin_addr) != 1) {
      std::fprintf(stderr, "bad --host %s\n", options_.host.c_str());
      std::exit(2);
    }
    if (::bind(listener, reinterpret_cast<sockaddr*>(&address),
               sizeof address) != 0 ||
        ::listen(listener, 8) != 0) {
      std::perror("bind/listen");
      std::exit(2);
    }
    std::printf("onnxsim native RPC server on %s:%d key='%s' onnxruntime %s\n",
                options_.host.c_str(), options_.port, options_.key.c_str(),
                Ort::GetVersionString().c_str());
    std::fflush(stdout);
    while (true) {
      int fd = ::accept(listener, nullptr, nullptr);
      if (fd < 0) continue;
      std::thread([this, fd] {
        handle(fd);
        ::close(fd);
      }).detach();
    }
  }

 private:
  static std::string sanitize(const std::string& name) {
    size_t slash = name.find_last_of("/\\");
    std::string base =
        slash == std::string::npos ? name : name.substr(slash + 1);
    for (char& c : base)
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
            c == '_' || c == '-'))
        c = '_';
    while (!base.empty() && base[0] == '.') base.erase(0, 1);
    if (base.empty()) throw std::runtime_error("invalid file name");
    return base;
  }

  static Json ok() {
    Json j = Json::Object();
    j.set("ok", Json::Bool(true));
    return j;
  }
  static Json fail(const std::string& why) {
    Json j = Json::Object();
    j.set("ok", Json::Bool(false)).set("error", Json::String(why));
    return j;
  }

  std::shared_ptr<Model> load(const std::vector<uint8_t>& bytes,
                              const Json& header) {
    Ort::SessionOptions so;
    so.SetLogSeverityLevel(3);
    if (header.flag("single_threaded")) {
      so.SetIntraOpNumThreads(1);
      so.SetInterOpNumThreads(1);
    }
    if (const Json* providers = header.find("providers")) {
      for (const auto& p : providers->a) {
        if (p.s == "CPUExecutionProvider") continue;
#if defined(__ANDROID__)
        if (p.s == "NnapiExecutionProvider") {
          Ort::ThrowOnError(
              OrtSessionOptionsAppendExecutionProvider_Nnapi(so, 0));
          continue;
        }
#endif
        throw std::runtime_error("execution provider '" + p.s +
                                 "' is not available in this server");
      }
    }
    auto model = std::make_shared<Model>();
    model->session =
        std::make_unique<Ort::Session>(env_, bytes.data(), bytes.size(), so);
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t k = 0; k < model->session->GetInputCount(); ++k)
      model->input_names.push_back(
          model->session->GetInputNameAllocated(k, allocator).get());
    for (size_t k = 0; k < model->session->GetOutputCount(); ++k)
      model->output_names.push_back(
          model->session->GetOutputNameAllocated(k, allocator).get());
    return model;
  }

  // Runs the model on the request's tensors; returns the output tensors (specs
  // + blobs).
  static Tensors execute(Model& model, const Json& header,
                         const std::vector<std::vector<uint8_t>>& blobs,
                         size_t first_blob) {
    const Json* specs = header.find("tensors");
    if (!specs) throw std::runtime_error("missing tensors");
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char*> in_names;
    std::vector<Ort::Value> in_values;
    std::vector<std::vector<int64_t>> shapes;
    for (size_t k = 0; k < specs->a.size(); ++k) {
      const Json& spec = specs->a[k];
      const DType* dt = dtype_by_name(spec.str("dtype"));
      if (!dt)
        throw std::runtime_error("unsupported tensor dtype '" +
                                 spec.str("dtype") + "'");
      shapes.emplace_back();
      size_t count = 1;
      for (const auto& d : spec.find("shape")->a) {
        shapes.back().push_back(d.i);
        count *= static_cast<size_t>(d.i);
      }
      if (first_blob + k >= blobs.size())
        throw std::runtime_error("fewer tensor blobs than descriptions");
      auto& blob = const_cast<std::vector<uint8_t>&>(blobs[first_blob + k]);
      if (blob.size() != count * dt->size)
        throw std::runtime_error("tensor '" + spec.str("name") +
                                 "' has the wrong byte length");
      in_values.push_back(Ort::Value::CreateTensor(
          memory, blob.data(), blob.size(), shapes.back().data(),
          shapes.back().size(), dt->type));
    }
    std::vector<std::string> ordered_names;
    for (const auto& spec : specs->a) ordered_names.push_back(spec.str("name"));
    for (const auto& n : ordered_names) in_names.push_back(n.c_str());
    std::vector<const char*> out_names;
    for (const auto& n : model.output_names) out_names.push_back(n.c_str());
    auto outputs = model.session->Run(Ort::RunOptions{nullptr}, in_names.data(),
                                      in_values.data(), in_values.size(),
                                      out_names.data(), out_names.size());
    Tensors result;
    for (size_t k = 0; k < outputs.size(); ++k) {
      auto info = outputs[k].GetTensorTypeAndShapeInfo();
      const DType* dt = dtype_by_type(info.GetElementType());
      if (!dt)
        throw std::runtime_error("output '" + model.output_names[k] +
                                 "' has an unsupported dtype");
      Json spec = Json::Object();
      spec.set("name", Json::String(model.output_names[k]))
          .set("dtype", Json::String(dt->name));
      Json shape = Json::Array();
      for (int64_t d : info.GetShape()) shape.push(Json::Int(d));
      spec.set("shape", shape);
      result.specs.push_back(spec);
      const auto* raw =
          static_cast<const uint8_t*>(outputs[k].GetTensorRawData());
      result.blobs.emplace_back(raw, raw + info.GetElementCount() * dt->size);
    }
    return result;
  }

  void handle(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    std::map<int64_t, std::shared_ptr<Model>> models;
    int64_t counter = 0;
    Message msg;
    try {
      if (!read_message(fd, msg)) return;
      if (msg.header.str("op") != "hello" ||
          (!options_.key.empty() && msg.header.str("key") != options_.key)) {
        write_message(
            fd, fail("handshake rejected (server key '" + options_.key + "')"));
        return;
      }
      Json hello = ok();
      hello.set("info", info());
      if (!write_message(fd, hello)) return;
      while (read_message(fd, msg)) {
        const std::string op = msg.header.str("op");
        if (op == "close") {
          write_message(fd, ok());
          return;
        }
        Json reply = ok();
        std::vector<std::vector<uint8_t>> out_blobs;
        try {
          if (op == "info") {
            reply.set("info", info());
          } else if (op == "upload") {
            if (msg.blobs.size() != 1)
              throw std::runtime_error("upload takes exactly one blob");
            std::string name = sanitize(msg.header.str("name"));
            std::string path = options_.workspace + "/" + name;
            FILE* f = std::fopen(path.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write " + path);
            std::fwrite(msg.blobs[0].data(), 1, msg.blobs[0].size(), f);
            std::fclose(f);
            reply.set("name", Json::String(name))
                .set("size",
                     Json::Int(static_cast<int64_t>(msg.blobs[0].size())));
          } else if (op == "load_model") {
            std::vector<uint8_t> bytes;
            if (!msg.blobs.empty()) {
              bytes = msg.blobs[0];
            } else {
              std::string path =
                  options_.workspace + "/" + sanitize(msg.header.str("name"));
              FILE* f = std::fopen(path.c_str(), "rb");
              if (!f)
                throw std::runtime_error("no such uploaded file '" +
                                         msg.header.str("name") + "'");
              std::fseek(f, 0, SEEK_END);
              bytes.resize(static_cast<size_t>(std::ftell(f)));
              std::fseek(f, 0, SEEK_SET);
              if (std::fread(bytes.data(), 1, bytes.size(), f) !=
                  bytes.size()) {
                std::fclose(f);
                throw std::runtime_error("short read");
              }
              std::fclose(f);
            }
            models[++counter] = load(bytes, msg.header);
            reply.set("handle", Json::Int(counter));
          } else if (op == "unload") {
            models.erase(msg.header.integer("handle"));
          } else if (op == "run" || op == "time") {
            auto it = models.find(msg.header.integer("handle"));
            if (it == models.end())
              throw std::runtime_error("no such model handle");
            Model& model = *it->second;
            if (op == "run") {
              Tensors out = execute(model, msg.header, msg.blobs, 0);
              Json specs = Json::Array();
              for (auto& s : out.specs) specs.push(s);
              reply.set("tensors", specs);
              out_blobs = std::move(out.blobs);
            } else {
              int64_t number =
                  std::max<int64_t>(msg.header.integer("number", 1), 1);
              int64_t repeat =
                  std::max<int64_t>(msg.header.integer("repeat", 1), 1);
              execute(model, msg.header, msg.blobs, 0);  // warm-up
              std::vector<double> times;
              Json results = Json::Array();
              for (int64_t r = 0; r < repeat; ++r) {
                auto start = std::chrono::steady_clock::now();
                for (int64_t n = 0; n < number; ++n)
                  execute(model, msg.header, msg.blobs, 0);
                double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count() /
                                 number;
                times.push_back(seconds);
                results.push(Json::Double(seconds));
              }
              std::sort(times.begin(), times.end());
              double median = times.size() % 2
                                  ? times[times.size() / 2]
                                  : 0.5 * (times[times.size() / 2 - 1] +
                                           times[times.size() / 2]);
              reply.set("results", results).set("median", Json::Double(median));
            }
          } else if (op == "run_once") {
            if (msg.blobs.empty())
              throw std::runtime_error("run_once needs the model in blob 0");
            auto model = load(msg.blobs[0], msg.header);
            Tensors out = execute(*model, msg.header, msg.blobs, 1);
            Json specs = Json::Array();
            for (auto& s : out.specs) specs.push(s);
            reply.set("tensors", specs);
            out_blobs = std::move(out.blobs);
          } else {
            throw std::runtime_error("unknown operation '" + op + "'");
          }
        } catch (const std::exception& e) {
          reply = fail(e.what());
          out_blobs.clear();
        }
        if (!write_message(fd, reply, out_blobs)) return;
      }
    } catch (const std::exception&) {
      return;
    }
  }

  Options options_;
  Ort::Env env_;
};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int k = 1; k < argc; ++k) {
    std::string arg = argv[k];
    auto value = [&]() -> std::string {
      if (k + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++k];
    };
    if (arg == "--host")
      options.host = value();
    else if (arg == "--port")
      options.port = std::atoi(value().c_str());
    else if (arg == "--key")
      options.key = value();
    else if (arg == "--workspace")
      options.workspace = value();
    else {
      std::fprintf(
          stderr,
          "usage: %s [--host H] [--port P] [--key K] [--workspace DIR]\n",
          argv[0]);
      return 2;
    }
  }
  std::string mkdir_cmd = "mkdir -p '" + options.workspace + "'";
  if (std::system(mkdir_cmd.c_str()) != 0) return 2;
  try {
    Server(options).serve();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
