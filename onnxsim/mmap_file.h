/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Read-only whole-file memory-mapping, shared by tensor_pool.cpp's
 * LoadSafetensorsMmap and tensor_pool_gguf.cpp's LoadGGUFMmap: both need the
 * exact same platform mapping/RAII-unmap mechanics regardless of which wire
 * format is being read afterwards, unlike the two codecs' actual parsing
 * logic, which stays deliberately separate (see tensor_pool_gguf.cpp's file
 * comment on why the two codecs otherwise duplicate rather than share).
 */
#ifndef ONNXSIM_MMAP_FILE_H_
#define ONNXSIM_MMAP_FILE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

// Platform-specific file-memory-mapping. Everything falls back to an
// ordinary read on a platform none of these branches cover (mirrors
// profiler.cpp's ReadCurrentRssBytes pattern), so callers keep working --
// just without the mmap benefit -- on wasm/Emscripten, which has no real
// demand-paged file mapping.
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

#if defined(_WIN32)
// Owns the Windows handles/view backing a mapping; unmapped/closed together
// once the last owner referencing it (via the aliasing shared_ptr TryMmapFile
// returns) is dropped.
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
// those), or any other OS-level error -- so callers can fall back to an
// ordinary read.
inline std::pair<std::shared_ptr<const char[]>, uint64_t> TryMmapFile(
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
// Owns the POSIX mapping; munmap'd once the last owner referencing it (via
// the aliasing shared_ptr TryMmapFile returns) is dropped.
struct FileMapping {
  void* addr = MAP_FAILED;
  size_t length = 0;
  ~FileMapping() {
    if (addr != MAP_FAILED && length > 0) ::munmap(addr, length);
  }
};

// Memory-maps `path` read-only. Returns {nullptr, 0} on any failure --
// missing file, zero-length file (mmap() rejects a zero length), or any
// other OS-level error -- so callers can fall back to an ordinary read.
inline std::pair<std::shared_ptr<const char[]>, uint64_t> TryMmapFile(
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
inline std::pair<std::shared_ptr<const char[]>, uint64_t> TryMmapFile(
    const std::string&) {
  return {nullptr, 0};
}
#endif

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_MMAP_FILE_H_
