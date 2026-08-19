#include "tensor_pool.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "tensor_pool_dtype.h"

// Platform-specific file-memory-mapping for LoadSafetensorsMmap. Everything
// falls back to LoadSafetensors's ordinary read on a platform none of these
// branches cover (mirrors profiler.cpp's ReadCurrentRssBytes pattern), so
// LoadSafetensorsMmap keeps working -- just without the mmap benefit -- on
// wasm/Emscripten, which has no real demand-paged file mapping.
#if defined(__EMSCRIPTEN__)
// No mapping support: TryMmapFile below always reports failure.
#elif defined(_WIN32)
// clang-format off
#include <windows.h>
// clang-format on
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace onnxsim {
namespace tensor_pool {

namespace {

// The file's 8-byte header-length field is little-endian by spec regardless
// of host byte order (see tensor_pool.h's "Byte order" note), so it is
// written/read one byte at a time rather than via a host uint64_t
// reinterpret_cast, which would be wrong on a big-endian host.
void WriteU64LE(std::ostream& out, uint64_t v) {
  char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<char>(v & 0xFF);
    v >>= 8;
  }
  out.write(buf, 8);
}

uint64_t ReadU64LE(std::istream& in) {
  unsigned char buf[8];
  in.read(reinterpret_cast<char*>(buf), 8);
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | buf[i];
  return v;
}

// Same decode as ReadU64LE, for the mmap path, which has no istream to read
// from -- just the mapped bytes themselves.
uint64_t DecodeU64LE(const char* buf) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) {
    v = (v << 8) | static_cast<unsigned char>(buf[i]);
  }
  return v;
}

#if defined(_WIN32)
// Owns the Windows handles/view backing a mapping; unmapped/closed together
// once the last Entry referencing it (via the aliasing shared_ptr below) is
// dropped.
struct FileMapping {
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
  void* view = nullptr;
  ~FileMapping() {
    if (view != nullptr) ::UnmapViewOfFile(view);
    if (mapping != nullptr) ::CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE) ::CloseHandle(file);
  }
};

// Memory-maps `path` read-only. Returns {nullptr, 0} on any failure --
// missing file, zero-length file (CreateFileMapping/MapViewOfFile reject
// those), or any other OS-level error -- so LoadSafetensorsMmap can fall
// back to an ordinary read.
std::pair<std::shared_ptr<const char[]>, uint64_t> TryMmapFile(
    const std::string& path) {
  HANDLE file =
      ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return {nullptr, 0};
  LARGE_INTEGER size;
  if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 0) {
    ::CloseHandle(file);
    return {nullptr, 0};
  }
  HANDLE mapping =
      ::CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    ::CloseHandle(file);
    return {nullptr, 0};
  }
  void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    ::CloseHandle(mapping);
    ::CloseHandle(file);
    return {nullptr, 0};
  }
  auto state = std::make_shared<FileMapping>();
  state->file = file;
  state->mapping = mapping;
  state->view = view;
  std::shared_ptr<const char[]> owner(state, static_cast<const char*>(view));
  return {owner, static_cast<uint64_t>(size.QuadPart)};
}
#elif !defined(__EMSCRIPTEN__)
// Owns the POSIX mapping; munmap'd once the last Entry referencing it (via
// the aliasing shared_ptr below) is dropped.
struct FileMapping {
  void* addr = MAP_FAILED;
  size_t length = 0;
  ~FileMapping() {
    if (addr != MAP_FAILED && length > 0) ::munmap(addr, length);
  }
};

// Memory-maps `path` read-only. Returns {nullptr, 0} on any failure --
// missing file, zero-length file (mmap() rejects a zero length), or any
// other OS-level error -- so LoadSafetensorsMmap can fall back to an
// ordinary read.
std::pair<std::shared_ptr<const char[]>, uint64_t> TryMmapFile(
    const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return {nullptr, 0};
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    return {nullptr, 0};
  }
  size_t length = static_cast<size_t>(st.st_size);
  void* addr = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
  // The mapping keeps the file's data reachable independent of the fd once
  // established, so the fd itself needn't (and, on some platforms, shouldn't)
  // outlive this call.
  ::close(fd);
  if (addr == MAP_FAILED) return {nullptr, 0};
  auto state = std::make_shared<FileMapping>();
  state->addr = addr;
  state->length = length;
  std::shared_ptr<const char[]> owner(state, static_cast<const char*>(addr));
  return {owner, static_cast<uint64_t>(length)};
}
#else
std::pair<std::shared_ptr<const char[]>, uint64_t> TryMmapFile(
    const std::string&) {
  return {nullptr, 0};
}
#endif

void AppendJsonEscaped(std::string& out, std::string_view s) {
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
}

// A minimal recursive-descent parser for exactly the shape a safetensors
// header takes: a flat JSON object whose values are either a tensor
// descriptor object ({"dtype", "shape", "data_offsets"}) or, for the single
// "__metadata__" key, an arbitrary value this pool ignores. It is not a
// general-purpose JSON parser -- SkipValue() below only needs to skip
// whatever "__metadata__" (or an unrecognized key inside a tensor
// descriptor) contains, well-formed or not.
class HeaderParser {
 public:
  explicit HeaderParser(std::string_view s) : s_(s) {}

  struct TensorHeader {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
  };

  std::vector<TensorHeader> Parse() {
    std::vector<TensorHeader> out;
    SkipWs();
    Expect('{');
    SkipWs();
    if (Peek() == '}') {
      Next();
      return out;
    }
    while (true) {
      SkipWs();
      std::string key = ParseString();
      SkipWs();
      Expect(':');
      SkipWs();
      if (key == "__metadata__") {
        SkipValue();
      } else {
        out.push_back(ParseTensorHeader(key));
      }
      SkipWs();
      char c = Next();
      if (c == ',') continue;
      if (c == '}') break;
      Fail("expected ',' or '}'");
    }
    return out;
  }

 private:
  std::string_view s_;
  size_t pos_ = 0;

  [[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error(
        "tensor_pool: malformed safetensors header at byte " +
        std::to_string(pos_) + ": " + what);
  }

  char Peek() {
    if (pos_ >= s_.size()) Fail("unexpected end of header");
    return s_[pos_];
  }

  char Next() {
    char c = Peek();
    ++pos_;
    return c;
  }

  void Expect(char c) {
    if (Next() != c) {
      Fail(std::string("expected '") + c + "'");
    }
  }

  void SkipWs() {
    while (pos_ < s_.size() &&
           std::isspace(static_cast<unsigned char>(s_[pos_]))) {
      ++pos_;
    }
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    while (true) {
      char c = Next();
      if (c == '"') break;
      if (c == '\\') {
        char e = Next();
        switch (e) {
          case '"':
            out += '"';
            break;
          case '\\':
            out += '\\';
            break;
          case '/':
            out += '/';
            break;
          case 'n':
            out += '\n';
            break;
          case 't':
            out += '\t';
            break;
          case 'r':
            out += '\r';
            break;
          case 'u': {
            // Header keys/dtype strings never contain non-ASCII escapes in
            // practice (tensor names are protobuf identifiers); decode only
            // the common case of a plain BMP code point being re-encoded as
            // ASCII when it fits, otherwise fall back to '?' rather than
            // implementing full UTF-16 surrogate pairing.
            if (pos_ + 4 > s_.size()) Fail("truncated \\u escape");
            unsigned cp = 0;
            for (int i = 0; i < 4; ++i) {
              char h = s_[pos_ + i];
              cp <<= 4;
              if (h >= '0' && h <= '9')
                cp |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f')
                cp |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F')
                cp |= static_cast<unsigned>(h - 'A' + 10);
              else
                Fail("invalid \\u escape");
            }
            pos_ += 4;
            out += (cp < 0x80) ? static_cast<char>(cp) : '?';
            break;
          }
          default:
            Fail("invalid escape");
        }
      } else {
        out += c;
      }
    }
    return out;
  }

  int64_t ParseInt() {
    size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size() &&
           std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
      ++pos_;
    }
    if (pos_ == start) Fail("expected integer");
    // Header ints (shape dims, data_offsets) are always plain integers, never
    // fractional/exponential, per the safetensors spec.
    return std::stoll(std::string(s_.substr(start, pos_ - start)));
  }

  std::vector<int64_t> ParseIntArray() {
    std::vector<int64_t> out;
    Expect('[');
    SkipWs();
    if (Peek() == ']') {
      Next();
      return out;
    }
    while (true) {
      SkipWs();
      out.push_back(ParseInt());
      SkipWs();
      char c = Next();
      if (c == ',') continue;
      if (c == ']') break;
      Fail("expected ',' or ']' in array");
    }
    return out;
  }

  TensorHeader ParseTensorHeader(std::string name) {
    TensorHeader th;
    th.name = std::move(name);
    bool have_offsets = false;
    Expect('{');
    SkipWs();
    if (Peek() == '}') {
      Next();
      Fail("tensor descriptor missing dtype/shape/data_offsets");
    }
    while (true) {
      SkipWs();
      std::string key = ParseString();
      SkipWs();
      Expect(':');
      SkipWs();
      if (key == "dtype") {
        th.dtype = ParseString();
      } else if (key == "shape") {
        th.shape = ParseIntArray();
      } else if (key == "data_offsets") {
        auto off = ParseIntArray();
        if (off.size() != 2 || off[0] < 0 || off[1] < off[0]) {
          Fail("data_offsets must be a [begin, end] pair with begin <= end");
        }
        th.begin = static_cast<uint64_t>(off[0]);
        th.end = static_cast<uint64_t>(off[1]);
        have_offsets = true;
      } else {
        SkipValue();
      }
      SkipWs();
      char c = Next();
      if (c == ',') continue;
      if (c == '}') break;
      Fail("expected ',' or '}' in tensor descriptor");
    }
    if (th.dtype.empty() || !have_offsets) {
      Fail("tensor descriptor missing dtype or data_offsets");
    }
    return th;
  }

  void SkipValue() {
    SkipWs();
    char c = Peek();
    if (c == '"') {
      ParseString();
    } else if (c == '{') {
      Next();
      SkipWs();
      if (Peek() == '}') {
        Next();
        return;
      }
      while (true) {
        SkipWs();
        ParseString();
        SkipWs();
        Expect(':');
        SkipValue();
        SkipWs();
        char cc = Next();
        if (cc == ',') continue;
        if (cc == '}') break;
        Fail("expected ',' or '}' while skipping object");
      }
    } else if (c == '[') {
      Next();
      SkipWs();
      if (Peek() == ']') {
        Next();
        return;
      }
      while (true) {
        SkipValue();
        SkipWs();
        char cc = Next();
        if (cc == ',') continue;
        if (cc == ']') break;
        Fail("expected ',' or ']' while skipping array");
      }
    } else if (c == 't' || c == 'f' || c == 'n') {
      // true / false / null
      while (pos_ < s_.size() &&
             std::isalpha(static_cast<unsigned char>(s_[pos_]))) {
        ++pos_;
      }
    } else {
      // number
      size_t start = pos_;
      while (pos_ < s_.size() &&
             (std::isdigit(static_cast<unsigned char>(s_[pos_])) ||
              s_[pos_] == '-' || s_[pos_] == '+' || s_[pos_] == '.' ||
              s_[pos_] == 'e' || s_[pos_] == 'E')) {
        ++pos_;
      }
      if (pos_ == start) Fail("unexpected character while skipping value");
    }
  }
};

}  // namespace

void TensorPool::Add(const std::string& name, int32_t dtype,
                     std::vector<int64_t> shape, std::string&& bytes) {
  auto owner = std::make_shared<std::string>(std::move(bytes));
  std::string_view view(*owner);
  // Aliasing constructor: shares `owner`'s control block (so the string
  // outlives every Entry referencing it) but points at its raw buffer,
  // giving Entry a uniform shared_ptr<const char[]> regardless of whether
  // its storage originated from a std::string (here) or a plain read buffer
  // (LoadSafetensors below).
  std::shared_ptr<const char[]> aliased(owner, view.data());
  entries_[name] = Entry{dtype, std::move(shape), std::move(aliased), view};
}

void TensorPool::Add(const std::string& name, int32_t dtype,
                     std::vector<int64_t> shape,
                     std::shared_ptr<const char[]> owner,
                     std::string_view data) {
  entries_[name] = Entry{dtype, std::move(shape), std::move(owner), data};
}

const Entry* TensorPool::Find(const std::string& name) const {
  auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

bool TensorPool::Erase(const std::string& name) {
  return entries_.erase(name) != 0;
}

std::string TensorPool::ContentHash(const std::string& name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    throw std::out_of_range("TensorPool::ContentHash: no entry named '" + name +
                            "'");
  }
  const Entry& entry = it->second;
  return ToHex(
      ContentDigest(hash_algorithm_, entry.dtype, entry.shape, entry.data));
}

std::map<std::string, std::string> TensorPool::AllContentHashes() const {
  std::map<std::string, std::string> out;
  for (const auto& [name, entry] : entries_) {
    out[name] = ToHex(
        ContentDigest(hash_algorithm_, entry.dtype, entry.shape, entry.data));
  }
  return out;
}

void TensorPool::SaveSafetensors(
    const std::string& path,
    std::map<std::string, std::pair<uint64_t, uint64_t>>* data_offsets_out)
    const {
  std::string header = "{";
  uint64_t offset = 0;
  bool first = true;
  if (data_offsets_out != nullptr) data_offsets_out->clear();
  for (const auto& [name, entry] : entries_) {
    std::string_view dt = ToSafetensors(entry.dtype);
    if (dt.empty()) {
      throw std::runtime_error("TensorPool::SaveSafetensors: tensor '" + name +
                               "' has an ONNX dtype (" +
                               std::to_string(entry.dtype) +
                               ") with no safetensors representation");
    }
    if (!first) header += ',';
    first = false;
    header += '"';
    AppendJsonEscaped(header, name);
    header += "\":{\"dtype\":\"";
    header += dt;
    header += "\",\"shape\":[";
    for (size_t i = 0; i < entry.shape.size(); ++i) {
      if (i != 0) header += ',';
      header += std::to_string(entry.shape[i]);
    }
    uint64_t begin = offset;
    uint64_t end = offset + entry.nbytes();
    header += "],\"data_offsets\":[";
    header += std::to_string(begin);
    header += ',';
    header += std::to_string(end);
    header += "]}";
    offset = end;
    if (data_offsets_out != nullptr) {
      (*data_offsets_out)[name] = {begin, end};
    }
  }
  header += '}';
  // The reference (Rust) safetensors writer pads the header with trailing
  // spaces (valid, semantically-inert JSON whitespace) so the data segment
  // starts 8-byte aligned -- friendlier to an mmap-based reader. Match that
  // convention for interop even though this reader doesn't require it.
  while (header.size() % 8 != 0) header += ' ';

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("TensorPool::SaveSafetensors: cannot open '" +
                             path + "' for writing");
  }
  WriteU64LE(out, header.size());
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& [name, entry] : entries_) {
    if (!entry.data.empty()) {
      out.write(entry.data.data(),
                static_cast<std::streamsize>(entry.data.size()));
    }
  }
  if (!out) {
    throw std::runtime_error("TensorPool::SaveSafetensors: write failed for '" +
                             path + "'");
  }
}

void TensorPool::LoadSafetensors(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("TensorPool::LoadSafetensors: cannot open '" +
                             path + "'");
  }
  in.seekg(0, std::ios::end);
  std::streamoff file_size = in.tellg();
  if (file_size < 8) {
    throw std::runtime_error("TensorPool::LoadSafetensors: '" + path +
                             "' is too small to be a safetensors file");
  }
  in.seekg(0, std::ios::beg);
  uint64_t header_len = ReadU64LE(in);
  if (!in || header_len > static_cast<uint64_t>(file_size) - 8) {
    throw std::runtime_error("TensorPool::LoadSafetensors: '" + path +
                             "' has an invalid header length");
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in) {
    throw std::runtime_error("TensorPool::LoadSafetensors: '" + path +
                             "' truncated while reading the header");
  }

  auto tensor_headers = HeaderParser(header).Parse();

  // Single read of the whole data segment: every Entry below aliases this
  // one buffer (via the shared_ptr aliasing constructor), so this is the
  // pool's only tensor-data copy, however many tensors the file holds.
  uint64_t data_size = static_cast<uint64_t>(file_size) - 8 - header_len;
  auto buffer = std::make_shared<std::string>(data_size, '\0');
  if (data_size > 0) {
    in.read(buffer->data(), static_cast<std::streamsize>(data_size));
    if (!in) {
      throw std::runtime_error("TensorPool::LoadSafetensors: '" + path +
                               "' truncated while reading tensor data");
    }
  }

  entries_.clear();
  for (auto& th : tensor_headers) {
    if (th.end > data_size) {
      throw std::runtime_error(
          "TensorPool::LoadSafetensors: '" + path + "' tensor '" + th.name +
          "' data_offsets [" + std::to_string(th.begin) + ", " +
          std::to_string(th.end) + ") exceed the file's data segment (" +
          std::to_string(data_size) + " bytes)");
    }
    int32_t dtype = FromSafetensors(th.dtype);
    if (dtype == ONNX_UNDEFINED) {
      throw std::runtime_error("TensorPool::LoadSafetensors: '" + path +
                               "' tensor '" + th.name +
                               "' has unsupported dtype '" + th.dtype + "'");
    }
    std::shared_ptr<const char[]> owner(buffer, buffer->data() + th.begin);
    std::string_view data(buffer->data() + th.begin, th.end - th.begin);
    entries_[th.name] =
        Entry{dtype, std::move(th.shape), std::move(owner), data};
  }
}

void TensorPool::LoadSafetensorsMmap(const std::string& path) {
  auto [mapped, file_size] = TryMmapFile(path);
  if (!mapped) {
    // No mapping support on this platform, or the mmap()/MapViewOfFile()
    // call itself failed -- fall back to the ordinary one-copy read rather
    // than surfacing a spurious failure for what may be a perfectly valid
    // file.
    LoadSafetensors(path);
    return;
  }

  if (file_size < 8) {
    throw std::runtime_error("TensorPool::LoadSafetensorsMmap: '" + path +
                             "' is too small to be a safetensors file");
  }
  uint64_t header_len = DecodeU64LE(mapped.get());
  if (header_len > file_size - 8) {
    throw std::runtime_error("TensorPool::LoadSafetensorsMmap: '" + path +
                             "' has an invalid header length");
  }

  // Unlike LoadSafetensors, the header itself doesn't need copying into a
  // std::string first -- HeaderParser only ever reads through a
  // std::string_view, and the mapped bytes already outlive this call via
  // `mapped`.
  std::string_view header(mapped.get() + 8, header_len);
  auto tensor_headers = HeaderParser(header).Parse();

  uint64_t data_start = 8 + header_len;
  uint64_t data_size = file_size - data_start;

  entries_.clear();
  for (auto& th : tensor_headers) {
    if (th.end > data_size) {
      throw std::runtime_error(
          "TensorPool::LoadSafetensorsMmap: '" + path + "' tensor '" + th.name +
          "' data_offsets [" + std::to_string(th.begin) + ", " +
          std::to_string(th.end) + ") exceed the file's data segment (" +
          std::to_string(data_size) + " bytes)");
    }
    int32_t dtype = FromSafetensors(th.dtype);
    if (dtype == ONNX_UNDEFINED) {
      throw std::runtime_error("TensorPool::LoadSafetensorsMmap: '" + path +
                               "' tensor '" + th.name +
                               "' has unsupported dtype '" + th.dtype + "'");
    }
    const char* base = mapped.get() + data_start + th.begin;
    std::shared_ptr<const char[]> owner(mapped, base);
    std::string_view data(base, th.end - th.begin);
    entries_[th.name] =
        Entry{dtype, std::move(th.shape), std::move(owner), data};
  }
}

uint64_t HeaderPrefixSize(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("tensor_pool::HeaderPrefixSize: cannot open '" +
                             path + "'");
  }
  uint64_t header_len = ReadU64LE(in);
  if (!in) {
    throw std::runtime_error("tensor_pool::HeaderPrefixSize: '" + path +
                             "' is too small to be a safetensors file");
  }
  return 8 + header_len;
}

}  // namespace tensor_pool
}  // namespace onnxsim
