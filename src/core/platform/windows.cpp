// Windows platform implementation.
#include "rope/core/platform.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace rope::platform {

// ---------------------------------------------------------------------------
// exe_path
// ---------------------------------------------------------------------------
std::filesystem::path exe_path() {
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return std::filesystem::path{buf};
}

// ---------------------------------------------------------------------------
// default_cache_dir
// ---------------------------------------------------------------------------
std::filesystem::path default_cache_dir() {
    if (const char* appdata = std::getenv("LOCALAPPDATA"))
        return std::filesystem::path{appdata} / "rope" / "drivers";
    return std::filesystem::temp_directory_path() / "rope" / "drivers";
}

// ---------------------------------------------------------------------------
// default_forecast_cache_path
// ---------------------------------------------------------------------------
std::filesystem::path default_forecast_cache_path() {
    if (const char* appdata = std::getenv("LOCALAPPDATA"))
        return std::filesystem::path{appdata} / "rope" / "forecast_grid.bin";
    return std::filesystem::temp_directory_path() / "rope" / "forecast_grid.bin";
}

// ---------------------------------------------------------------------------
// MappedFile — Windows CreateFileMapping/MapViewOfFile
// ---------------------------------------------------------------------------
struct MappedFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    void*  addr = nullptr;

    ~Impl() {
        if (addr) ::UnmapViewOfFile(addr);
        if (mapping) ::CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) ::CloseHandle(file);
    }
};

MappedFile::MappedFile() = default;

MappedFile::MappedFile(MappedFile&& other) noexcept
    : impl_(std::move(other.impl_)), data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}
MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}
MappedFile::~MappedFile() = default;

MappedFile MappedFile::open_readonly(const std::filesystem::path& path) {
    HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error(
            "MappedFile::open_readonly: CreateFile failed: " +
            std::to_string(GetLastError()));

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 0) {
        ::CloseHandle(file);
        throw std::runtime_error("MappedFile::open_readonly: file is empty: " + path.string());
    }

    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        int e = static_cast<int>(GetLastError());
        ::CloseHandle(file);
        throw std::runtime_error(
            "MappedFile::open_readonly: CreateFileMapping failed: " + std::to_string(e));
    }

    void* addr = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        int e = static_cast<int>(GetLastError());
        ::CloseHandle(mapping);
        ::CloseHandle(file);
        throw std::runtime_error(
            "MappedFile::open_readonly: MapViewOfFile failed: " + std::to_string(e));
    }

    MappedFile mf;
    mf.impl_ = std::make_unique<Impl>();
    mf.impl_->file    = file;
    mf.impl_->mapping = mapping;
    mf.impl_->addr    = addr;
    mf.data_ = static_cast<const std::byte*>(addr);
    mf.size_ = static_cast<std::size_t>(size.QuadPart);
    return mf;
}

} // namespace rope::platform
