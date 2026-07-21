#include "rope/io/forecast_grid_bin.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

namespace rope::io {

static constexpr std::uint32_t FG_MAGIC   = 0x52504647u;  // "RPFG"
static constexpr std::uint32_t FG_VERSION = 1u;

namespace {

std::filesystem::path make_temp_path(const std::filesystem::path& path) {
    // Same directory as `path` — rename() is only guaranteed atomic within
    // one filesystem/volume; a cross-filesystem rename can silently
    // degrade to a non-atomic copy+delete in some stdlib implementations.
    auto ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::random_device rd;
    auto suffix = path.filename().string() + ".tmp-" +
                  std::to_string(ticks) + "-" + std::to_string(rd());
    return path.parent_path() / suffix;
}

} // namespace

ForecastGrid ForecastGridBin::load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path))
        throw ForecastCacheMissingError(
            "ForecastGridBin::load: no forecast cached at " + path.string());

    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: cannot open " + path.string());

    std::uint32_t magic, version, reserved;
    std::int32_t  n_lst, n_lat, n_alt, H;
    double        lat_min_deg, lat_max_deg, alt_min_km, alt_max_km;

    f.read(reinterpret_cast<char*>(&magic),       4);
    f.read(reinterpret_cast<char*>(&version),     4);
    f.read(reinterpret_cast<char*>(&n_lst),       4);
    f.read(reinterpret_cast<char*>(&n_lat),       4);
    f.read(reinterpret_cast<char*>(&n_alt),       4);
    f.read(reinterpret_cast<char*>(&H),           4);
    f.read(reinterpret_cast<char*>(&lat_min_deg), 8);
    f.read(reinterpret_cast<char*>(&lat_max_deg), 8);
    f.read(reinterpret_cast<char*>(&alt_min_km),  8);
    f.read(reinterpret_cast<char*>(&alt_max_km),  8);
    f.read(reinterpret_cast<char*>(&reserved),    4);

    if (!f)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: failed to read header from " + path.string());
    if (magic != FG_MAGIC)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: bad magic in " + path.string() +
            " (not a forecast-grid cache file)");
    if (version != FG_VERSION)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: unsupported version " +
            std::to_string(version) + " in " + path.string());
    if (n_lst <= 0 || n_lat <= 1 || n_alt <= 1 || H <= 0)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: implausible grid shape in " + path.string());
    if (lat_min_deg >= lat_max_deg || alt_min_km >= alt_max_km)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: implausible grid range in " + path.string());

    ForecastGrid grid;
    grid.shape.n_lst       = n_lst;
    grid.shape.n_lat       = n_lat;
    grid.shape.n_alt       = n_alt;
    grid.shape.lat_min_deg = lat_min_deg;
    grid.shape.lat_max_deg = lat_max_deg;
    grid.shape.alt_min_km  = alt_min_km;
    grid.shape.alt_max_km  = alt_max_km;
    grid.H                 = H;

    const std::size_t voxels = static_cast<std::size_t>(grid.shape.voxels());
    grid.times.resize(static_cast<std::size_t>(H));
    grid.density.resize(static_cast<std::size_t>(H) * voxels);
    grid.uncertainty.resize(static_cast<std::size_t>(H) * voxels);

    f.read(reinterpret_cast<char*>(grid.times.data()),
           static_cast<std::streamsize>(grid.times.size() * sizeof(std::int64_t)));
    f.read(reinterpret_cast<char*>(grid.density.data()),
           static_cast<std::streamsize>(grid.density.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(grid.uncertainty.data()),
           static_cast<std::streamsize>(grid.uncertainty.size() * sizeof(float)));

    if (!f)
        throw ForecastCacheCorruptError(
            "ForecastGridBin::load: truncated file (expected more data) in " + path.string());

    return grid;
}

void ForecastGridBin::save(const ForecastGrid& grid, const std::filesystem::path& path) {
    if (grid.H <= 0)
        throw std::runtime_error("ForecastGridBin::save: grid.H must be > 0");
    const std::size_t voxels = static_cast<std::size_t>(grid.shape.voxels());
    if (grid.times.size()       != static_cast<std::size_t>(grid.H) ||
        grid.density.size()     != static_cast<std::size_t>(grid.H) * voxels ||
        grid.uncertainty.size() != static_cast<std::size_t>(grid.H) * voxels)
        throw std::runtime_error(
            "ForecastGridBin::save: grid array sizes are inconsistent with H * shape.voxels()");

    std::filesystem::create_directories(path.parent_path());
    auto tmp = make_temp_path(path);

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f)
            throw std::runtime_error("ForecastGridBin::save: cannot open " + tmp.string());

        const std::uint32_t reserved = 0u;
        const std::int32_t  n_lst = grid.shape.n_lst, n_lat = grid.shape.n_lat,
                             n_alt = grid.shape.n_alt, H = grid.H;

        f.write(reinterpret_cast<const char*>(&FG_MAGIC),   4);
        f.write(reinterpret_cast<const char*>(&FG_VERSION), 4);
        f.write(reinterpret_cast<const char*>(&n_lst),      4);
        f.write(reinterpret_cast<const char*>(&n_lat),      4);
        f.write(reinterpret_cast<const char*>(&n_alt),      4);
        f.write(reinterpret_cast<const char*>(&H),          4);
        f.write(reinterpret_cast<const char*>(&grid.shape.lat_min_deg), 8);
        f.write(reinterpret_cast<const char*>(&grid.shape.lat_max_deg), 8);
        f.write(reinterpret_cast<const char*>(&grid.shape.alt_min_km),  8);
        f.write(reinterpret_cast<const char*>(&grid.shape.alt_max_km),  8);
        f.write(reinterpret_cast<const char*>(&reserved),    4);

        f.write(reinterpret_cast<const char*>(grid.times.data()),
                static_cast<std::streamsize>(grid.times.size() * sizeof(std::int64_t)));
        f.write(reinterpret_cast<const char*>(grid.density.data()),
                static_cast<std::streamsize>(grid.density.size() * sizeof(float)));
        f.write(reinterpret_cast<const char*>(grid.uncertainty.data()),
                static_cast<std::streamsize>(grid.uncertainty.size() * sizeof(float)));

        if (!f) {
            std::filesystem::remove(tmp);
            throw std::runtime_error("ForecastGridBin::save: write failed for " + tmp.string());
        }
    }

    std::filesystem::rename(tmp, path);
}

// ---------------------------------------------------------------------------
// ForecastGridBinWriter
// ---------------------------------------------------------------------------

ForecastGridBinWriter ForecastGridBinWriter::open(
    const GridSpec& shape, int H, const std::filesystem::path& path)
{
    if (H <= 0)
        throw std::runtime_error("ForecastGridBinWriter::open: H must be > 0");

    std::filesystem::create_directories(path.parent_path());

    ForecastGridBinWriter w;
    w.tmp_path_   = make_temp_path(path);
    w.final_path_ = path;
    w.H_          = H;
    w.voxels_     = shape.voxels();

    w.file_.open(w.tmp_path_, std::ios::binary | std::ios::trunc);
    if (!w.file_)
        throw std::runtime_error("ForecastGridBinWriter::open: cannot open " + w.tmp_path_.string());

    const std::uint32_t reserved = 0u;
    const std::int32_t  n_lst = shape.n_lst, n_lat = shape.n_lat, n_alt = shape.n_alt, H32 = H;

    w.file_.write(reinterpret_cast<const char*>(&FG_MAGIC),   4);
    w.file_.write(reinterpret_cast<const char*>(&FG_VERSION), 4);
    w.file_.write(reinterpret_cast<const char*>(&n_lst),      4);
    w.file_.write(reinterpret_cast<const char*>(&n_lat),      4);
    w.file_.write(reinterpret_cast<const char*>(&n_alt),      4);
    w.file_.write(reinterpret_cast<const char*>(&H32),        4);
    w.file_.write(reinterpret_cast<const char*>(&shape.lat_min_deg), 8);
    w.file_.write(reinterpret_cast<const char*>(&shape.lat_max_deg), 8);
    w.file_.write(reinterpret_cast<const char*>(&shape.alt_min_km),  8);
    w.file_.write(reinterpret_cast<const char*>(&shape.alt_max_km),  8);
    w.file_.write(reinterpret_cast<const char*>(&reserved),   4);

    if (!w.file_) {
        std::filesystem::remove(w.tmp_path_);
        throw std::runtime_error("ForecastGridBinWriter::open: header write failed for " + w.tmp_path_.string());
    }

    constexpr std::size_t HEADER_BYTES = 60;
    w.times_base_       = HEADER_BYTES;
    w.density_base_     = w.times_base_ + static_cast<std::size_t>(H) * sizeof(std::int64_t);
    w.uncertainty_base_ = w.density_base_ + static_cast<std::size_t>(H) * static_cast<std::size_t>(w.voxels_) * sizeof(float);

    return w;
}

void ForecastGridBinWriter::write_chunk(
    int t_offset, std::span<const std::int64_t> times,
    std::span<const float> density, std::span<const float> uncertainty)
{
    const std::size_t count = times.size();
    if (t_offset != next_expected_t_)
        throw std::logic_error(
            "ForecastGridBinWriter::write_chunk: expected t_offset=" +
            std::to_string(next_expected_t_) + ", got " + std::to_string(t_offset));
    if (density.size()     != count * static_cast<std::size_t>(voxels_) ||
        uncertainty.size() != count * static_cast<std::size_t>(voxels_))
        throw std::runtime_error(
            "ForecastGridBinWriter::write_chunk: array sizes inconsistent with times.size() * voxels");
    if (static_cast<std::size_t>(t_offset) + count > static_cast<std::size_t>(H_))
        throw std::runtime_error("ForecastGridBinWriter::write_chunk: chunk exceeds H");

    const std::size_t voxels_sz = static_cast<std::size_t>(voxels_);

    file_.seekp(static_cast<std::streamoff>(
        times_base_ + static_cast<std::size_t>(t_offset) * sizeof(std::int64_t)));
    file_.write(reinterpret_cast<const char*>(times.data()),
                static_cast<std::streamsize>(count * sizeof(std::int64_t)));

    file_.seekp(static_cast<std::streamoff>(
        density_base_ + static_cast<std::size_t>(t_offset) * voxels_sz * sizeof(float)));
    file_.write(reinterpret_cast<const char*>(density.data()),
                static_cast<std::streamsize>(density.size() * sizeof(float)));

    file_.seekp(static_cast<std::streamoff>(
        uncertainty_base_ + static_cast<std::size_t>(t_offset) * voxels_sz * sizeof(float)));
    file_.write(reinterpret_cast<const char*>(uncertainty.data()),
                static_cast<std::streamsize>(uncertainty.size() * sizeof(float)));

    if (!file_)
        throw std::runtime_error("ForecastGridBinWriter::write_chunk: write failed for " + tmp_path_.string());

    next_expected_t_ += static_cast<int>(count);
}

void ForecastGridBinWriter::close() {
    if (closed_)
        throw std::logic_error("ForecastGridBinWriter::close: already closed");
    if (next_expected_t_ != H_)
        throw std::logic_error(
            "ForecastGridBinWriter::close: only " + std::to_string(next_expected_t_) +
            " of " + std::to_string(H_) + " timesteps were written");

    file_.flush();
    if (!file_)
        throw std::runtime_error("ForecastGridBinWriter::close: flush failed for " + tmp_path_.string());
    file_.close();

    std::filesystem::rename(tmp_path_, final_path_);
    closed_ = true;
}

ForecastGridBinWriter::~ForecastGridBinWriter() {
    if (!closed_ && !tmp_path_.empty()) {
        if (file_.is_open()) file_.close();
        std::error_code ec;
        std::filesystem::remove(tmp_path_, ec);
    }
}

} // namespace rope::io
