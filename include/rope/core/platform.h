#pragma once
// Platform abstraction layer — OS-specific paths and the memory-mapped-file
// primitive.
//
// Implementations live in src/core/platform/posix.cpp (Linux/macOS) and
// src/core/platform/windows.cpp (Windows).  No other source file may include
// OS-specific headers directly.

#include <cstddef>
#include <filesystem>
#include <memory>

namespace rope::platform {

// ---------------------------------------------------------------------------
// Returns the absolute, canonical path of the currently running executable.
// ---------------------------------------------------------------------------
std::filesystem::path exe_path();

// ---------------------------------------------------------------------------
// Default directory for cached driver files written by DriverCacheManager.
//   Linux/macOS: $XDG_CACHE_HOME/rope/drivers or ~/.cache/rope/drivers
//   Windows:     %LOCALAPPDATA%\rope\drivers
// ---------------------------------------------------------------------------
std::filesystem::path default_cache_dir();

// ---------------------------------------------------------------------------
// Default per-user path for the single cached forecast-grid file written by
// `rope forecast` and read by `rope get` / `rope_open`. Exactly one slot —
// a new `rope forecast` overwrites this file. Uses the same persistent-cache
// root as default_cache_dir() (a runtime-dir root would be wrong for
// something meant to survive between sessions):
//   Linux/macOS: $XDG_CACHE_HOME/rope/forecast_grid.bin, else ~/.cache/rope/forecast_grid.bin
//   Windows:     %LOCALAPPDATA%\rope\forecast_grid.bin
// ---------------------------------------------------------------------------
std::filesystem::path default_forecast_cache_path();

// ---------------------------------------------------------------------------
// MappedFile — read-only memory-mapped file. Move-only.
//
// Lets a reader (rope_open / rope get) hold a multi-gigabyte forecast-grid
// cache file open without materializing it into process memory; the OS
// pages in only the bytes actually touched by queries.
//
// On POSIX, if the underlying file is atomically replaced (temp+rename)
// while still mapped, the mapping keeps serving the old data until it's
// released — rename/unlink never invalidates an existing mapping or fd.
// The Windows implementation opens with FILE_SHARE_DELETE to reproduce the
// same property.
// ---------------------------------------------------------------------------
class MappedFile {
public:
    // Declared (not defined inline) — the implicit exception-unwind path
    // through a default member initializer would need Impl's complete type,
    // which isn't visible in this header. Defined in each platform .cpp.
    MappedFile();
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();

    // Throws std::runtime_error on failure (file missing, cannot map, etc).
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
