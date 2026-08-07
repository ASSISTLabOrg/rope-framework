#include "rope/io/driver_bin.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace rope::io {

static constexpr std::uint32_t SW_MAGIC       = 0x52505357u;  // "RPSW"
static constexpr std::uint32_t SW_VERSION     = 2u;
static constexpr std::uint32_t SW_MAX_NAME_LEN = 256u;

SpaceWeatherDB SpaceWeatherBin::load(const std::filesystem::path& bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::load: cannot open " + bin_path.string());

    std::uint32_t magic, version, nrows, ncols;
    f.read(reinterpret_cast<char*>(&magic),   4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&nrows),   4);
    f.read(reinterpret_cast<char*>(&ncols),   4);

    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::load: failed to read header from " +
            bin_path.string());
    if (magic != SW_MAGIC)
        throw std::runtime_error(
            "SpaceWeatherBin::load: bad magic in " + bin_path.string() +
            " (not a .swbin file)");
    if (version != SW_VERSION)
        throw std::runtime_error(
            "SpaceWeatherBin::load: unsupported version " +
            std::to_string(version) + " in " + bin_path.string());

    std::vector<std::string> raw_names(ncols);
    for (std::uint32_t j = 0; j < ncols; ++j) {
        std::uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        if (!f)
            throw std::runtime_error(
                "SpaceWeatherBin::load: unexpected EOF reading name table in " +
                bin_path.string());
        if (name_len == 0 || name_len > SW_MAX_NAME_LEN)
            throw std::runtime_error(
                "SpaceWeatherBin::load: implausible column name length " +
                std::to_string(name_len) + " in " + bin_path.string());

        std::string name(name_len, '\0');
        f.read(name.data(), name_len);
        if (!f)
            throw std::runtime_error(
                "SpaceWeatherBin::load: unexpected EOF reading name table in " +
                bin_path.string());
        raw_names[j] = std::move(name);
    }

    std::vector<TimePoint>          times(nrows);
    std::vector<std::vector<float>> raw_data(ncols, std::vector<float>(nrows));

    std::vector<float> row_buf(ncols);
    for (std::uint32_t i = 0; i < nrows; ++i) {
        std::int64_t tp_raw;
        f.read(reinterpret_cast<char*>(&tp_raw), 8);
        if (ncols > 0)
            f.read(reinterpret_cast<char*>(row_buf.data()),
                   static_cast<std::streamsize>(ncols * 4));

        if (!f)
            throw std::runtime_error(
                "SpaceWeatherBin::load: unexpected EOF at record " +
                std::to_string(i) + " in " + bin_path.string());

        times[i] = static_cast<TimePoint>(tp_raw);
        for (std::uint32_t j = 0; j < ncols; ++j)
            raw_data[j][i] = row_buf[j];
    }

    return SpaceWeatherDB{std::move(times), std::move(raw_names), std::move(raw_data)};
}

void SpaceWeatherBin::save(const SpaceWeatherDB& db,
                           const std::filesystem::path& bin_path) {
    std::ofstream f(bin_path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::save: cannot open " + bin_path.string());

    const auto nrows = static_cast<std::uint32_t>(db.times_.size());
    const auto ncols = static_cast<std::uint32_t>(db.raw_names_.size());

    f.write(reinterpret_cast<const char*>(&SW_MAGIC),   4);
    f.write(reinterpret_cast<const char*>(&SW_VERSION), 4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);
    f.write(reinterpret_cast<const char*>(&ncols),      4);

    for (std::uint32_t j = 0; j < ncols; ++j) {
        const auto& name     = db.raw_names_[j];
        const auto  name_len = static_cast<std::uint32_t>(name.size());
        f.write(reinterpret_cast<const char*>(&name_len), 4);
        f.write(name.data(), static_cast<std::streamsize>(name_len));
    }

    for (std::uint32_t i = 0; i < nrows; ++i) {
        auto tp_raw = static_cast<std::int64_t>(db.times_[i]);
        f.write(reinterpret_cast<const char*>(&tp_raw), 8);
        for (std::uint32_t j = 0; j < ncols; ++j) {
            float v = db.raw_data_[j][i];
            f.write(reinterpret_cast<const char*>(&v), 4);
        }
    }

    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::save: write failed for " + bin_path.string());
}

} // namespace rope::io
