#pragma once
// Platform abstraction layer — OS-specific paths and the memory-mapped-file primitive.
// Implemented in src/core/platform/{posix,windows}.cpp; no other file includes OS-specific headers.

#include <cstddef>
#include <filesystem>
#include <memory>

namespace rope::platform {

// Absolute, canonical path of the running executable.
std::filesystem::path exe_path();

// Default driver-cache directory: $XDG_CACHE_HOME/rope/drivers or ~/.cache/rope/drivers (Linux/macOS), %LOCALAPPDATA%\rope\drivers (Windows).
std::filesystem::path default_cache_dir();

// Default forecast-grid cache path (one slot, overwritten each `rope forecast`); same root as default_cache_dir().
std::filesystem::path default_forecast_cache_path();

// Read-only memory-mapped file. Move-only. A concurrent atomic replace (temp+rename) of the underlying path
// doesn't invalidate an open mapping — POSIX gets this for free; Windows reproduces it via FILE_SHARE_DELETE.
class MappedFile {
public:
    // Defined per-platform .cpp, not inline (needs Impl's complete type).
    MappedFile();
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();

    // Throws std::runtime_error on failure.
    static MappedFile open_readonly(const std::filesystem::path& path);

    const std::byte* data() const noexcept { return data_; }
    std::size_t       size() const noexcept { return size_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const std::byte* data_ = nullptr;
    std::size_t       size_ = 0;
};

} // namespace rope::platform
