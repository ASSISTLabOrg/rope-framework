#include "rope/io/mapped_forecast_grid.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace rope::io {

namespace {
constexpr std::uint32_t FG_MAGIC   = 0x52504647u;  // "RPFG" — matches forecast_grid_bin.cpp
constexpr std::uint32_t FG_VERSION = 1u;
constexpr std::size_t   HEADER_SIZE = 60;

template <class T>
T read_field(const std::byte* base, std::size_t offset) {
    T v;
    std::memcpy(&v, base + offset, sizeof(T));
    return v;
}
} // namespace

MappedForecastGrid MappedForecastGrid::open(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path))
        throw ForecastCacheMissingError(
            "MappedForecastGrid::open: no forecast cached at " + path.string());

    platform::MappedFile file;
    try {
        file = platform::MappedFile::open_readonly(path);
    } catch (const std::exception& e) {
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: " + std::string(e.what()));
    }

    if (file.size() < HEADER_SIZE)
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: file too short for a header: " + path.string());

    const std::byte* base = file.data();
    auto magic   = read_field<std::uint32_t>(base, 0);
    auto version = read_field<std::uint32_t>(base, 4);
    auto n_lst   = read_field<std::int32_t>(base, 8);
    auto n_lat   = read_field<std::int32_t>(base, 12);
    auto n_alt   = read_field<std::int32_t>(base, 16);
    auto H       = read_field<std::int32_t>(base, 20);
    auto lat_min_deg = read_field<double>(base, 24);
    auto lat_max_deg = read_field<double>(base, 32);
    auto alt_min_km  = read_field<double>(base, 40);
    auto alt_max_km  = read_field<double>(base, 48);

    if (magic != FG_MAGIC)
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: bad magic in " + path.string() +
            " (not a forecast-grid cache file)");
    if (version != FG_VERSION)
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: unsupported version " +
            std::to_string(version) + " in " + path.string());
    if (n_lst <= 0 || n_lat <= 1 || n_alt <= 1 || H <= 0)
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: implausible grid shape in " + path.string());
    if (lat_min_deg >= lat_max_deg || alt_min_km >= alt_max_km)
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: implausible grid range in " + path.string());

    MappedForecastGrid grid;
    grid.shape.n_lst       = n_lst;
    grid.shape.n_lat       = n_lat;
    grid.shape.n_alt       = n_alt;
    grid.shape.lat_min_deg = lat_min_deg;
    grid.shape.lat_max_deg = lat_max_deg;
    grid.shape.alt_min_km  = alt_min_km;
    grid.shape.alt_max_km  = alt_max_km;
    grid.H                 = H;

    const std::size_t voxels = static_cast<std::size_t>(grid.shape.voxels());
    const std::size_t times_bytes       = static_cast<std::size_t>(H) * sizeof(std::int64_t);
    const std::size_t density_bytes     = static_cast<std::size_t>(H) * voxels * sizeof(float);
    const std::size_t uncertainty_bytes = density_bytes;
    const std::size_t expected_size = HEADER_SIZE + times_bytes + density_bytes + uncertainty_bytes;

    if (file.size() < expected_size)
        throw ForecastCacheCorruptError(
            "MappedForecastGrid::open: truncated file (expected more data) in " + path.string());

    grid.times.resize(static_cast<std::size_t>(H));
    std::memcpy(grid.times.data(), base + HEADER_SIZE, times_bytes);

    grid.density_offset_     = HEADER_SIZE + times_bytes;
    grid.uncertainty_offset_ = grid.density_offset_ + density_bytes;
    grid.file_ = std::move(file);
    return grid;
}

const float* MappedForecastGrid::density_at(int t) const noexcept {
    const std::size_t voxels = static_cast<std::size_t>(shape.voxels());
    const std::byte* p = file_.data() + density_offset_ +
                          static_cast<std::size_t>(t) * voxels * sizeof(float);
    return reinterpret_cast<const float*>(p);
}

const float* MappedForecastGrid::uncertainty_at(int t) const noexcept {
    const std::size_t voxels = static_cast<std::size_t>(shape.voxels());
    const std::byte* p = file_.data() + uncertainty_offset_ +
                          static_cast<std::size_t>(t) * voxels * sizeof(float);
    return reinterpret_cast<const float*>(p);
}

} // namespace rope::io
