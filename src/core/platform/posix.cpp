// POSIX platform implementation (Linux + macOS).
#include "rope/core/platform.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <climits>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef __APPLE__
#  include <mach-o/dyld.h>
#endif

namespace rope::platform {

// ---------------------------------------------------------------------------
// exe_path
// ---------------------------------------------------------------------------
std::filesystem::path exe_path() {
#ifdef __linux__
    return std::filesystem::canonical("/proc/self/exe");
#else
    // macOS: _NSGetExecutablePath returns the real path of the running binary.
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        std::error_code ec;
        auto p = std::filesystem::canonical(buf, ec);
        return ec ? std::filesystem::path{buf} : p;
    }
    throw std::runtime_error("exe_path: _NSGetExecutablePath failed");
#endif
}

// ---------------------------------------------------------------------------
// default_cache_dir
// ---------------------------------------------------------------------------
std::filesystem::path default_cache_dir() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"))
        return std::filesystem::path{xdg} / "rope" / "drivers";
    if (const char* home = std::getenv("HOME"))
        return std::filesystem::path{home} / ".cache" / "rope" / "drivers";
    return std::filesystem::temp_directory_path() / "rope" / "drivers";
}

// ---------------------------------------------------------------------------
// default_forecast_cache_path
// ---------------------------------------------------------------------------
std::filesystem::path default_forecast_cache_path() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"))
        return std::filesystem::path{xdg} / "rope" / "forecast_grid.bin";
    if (const char* home = std::getenv("HOME"))
        return std::filesystem::path{home} / ".cache" / "rope" / "forecast_grid.bin";
    return std::filesystem::temp_directory_path() / "rope" / "forecast_grid.bin";
}

// ---------------------------------------------------------------------------
// MappedFile — POSIX mmap
// ---------------------------------------------------------------------------
struct MappedFile::Impl {
    int fd = -1;
    void* addr = nullptr;
    std::size_t len = 0;

    ~Impl() {
        if (addr && addr != MAP_FAILED) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
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
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error(
            std::string("MappedFile::open_readonly: open(): ") + std::strerror(errno));

    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            std::string("MappedFile::open_readonly: fstat(): ") + std::strerror(e));
    }
    auto len = static_cast<std::size_t>(st.st_size);
    if (len == 0) {
        ::close(fd);
        throw std::runtime_error("MappedFile::open_readonly: file is empty: " + path.string());
    }

    void* addr = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            std::string("MappedFile::open_readonly: mmap(): ") + std::strerror(e));
    }

    MappedFile mf;
    mf.impl_ = std::make_unique<Impl>();
    mf.impl_->fd   = fd;
    mf.impl_->addr = addr;
    mf.impl_->len  = len;
    mf.data_ = static_cast<const std::byte*>(addr);
    mf.size_ = len;
    return mf;
}

} // namespace rope::platform
