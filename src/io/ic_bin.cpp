#include "rope/io/ic_bin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rope::io {

static constexpr std::uint32_t IC_MAGIC        = 0x52504943u;  // "RPIC"
static constexpr std::uint32_t IC_VERSION      = 2u;
static constexpr std::uint32_t IC_MAX_NAME_LEN = 256u;

static std::string read_axis_name(std::ifstream& f, const std::filesystem::path& bin_path) {
    std::uint32_t name_len;
    f.read(reinterpret_cast<char*>(&name_len), 4);
    if (!f)
        throw std::runtime_error(
            "IcBin::load: unexpected EOF reading axis name in " + bin_path.string());
    if (name_len == 0 || name_len > IC_MAX_NAME_LEN)
        throw std::runtime_error(
            "IcBin::load: implausible axis name length " + std::to_string(name_len) +
            " in " + bin_path.string());

    std::string name(name_len, '\0');
    f.read(name.data(), name_len);
    if (!f)
        throw std::runtime_error(
            "IcBin::load: unexpected EOF reading axis name in " + bin_path.string());
    return name;
}

ICTable IcBin::load(const std::filesystem::path& bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f)
        throw std::runtime_error(
            "IcBin::load: cannot open " + bin_path.string());

    std::uint32_t magic, version, nrows, latent_dim;
    f.read(reinterpret_cast<char*>(&magic),      4);
    f.read(reinterpret_cast<char*>(&version),    4);
    f.read(reinterpret_cast<char*>(&nrows),      4);
    f.read(reinterpret_cast<char*>(&latent_dim), 4);

    if (!f)
        throw std::runtime_error(
            "IcBin::load: failed to read header from " + bin_path.string());
    if (magic != IC_MAGIC)
        throw std::runtime_error(
            "IcBin::load: bad magic in " + bin_path.string() +
            " (not a .icbin file)");
    if (version != IC_VERSION)
        throw std::runtime_error(
            "IcBin::load: unsupported version " +
            std::to_string(version) + " in " + bin_path.string());
    if (latent_dim == 0)
        throw std::runtime_error(
            "IcBin::load: latent_dim=0 in " + bin_path.string());

    const int K = static_cast<int>(latent_dim);

    std::vector<std::string> axis_names = {
        read_axis_name(f, bin_path),
        read_axis_name(f, bin_path)
    };

    std::vector<float> pts_axis0(nrows), pts_axis1(nrows);
    std::vector<float> vals(static_cast<std::size_t>(nrows) * K);

    for (std::uint32_t i = 0; i < nrows; ++i) {
        f.read(reinterpret_cast<char*>(&pts_axis0[i]), 4);
        f.read(reinterpret_cast<char*>(&pts_axis1[i]), 4);
        f.read(reinterpret_cast<char*>(&vals[i * K]), static_cast<std::streamsize>(K * 4));

        if (!f)
            throw std::runtime_error(
                "IcBin::load: unexpected EOF at record " +
                std::to_string(i) + " in " + bin_path.string());
    }

    // Rebuild axes and grid index (same logic as CSV constructor).
    auto unique_sorted = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end(),
                            [](float a, float b){ return std::abs(a - b) < 1e-4f; }),
                v.end());
        return v;
    };

    std::vector<float> axis0_grid = unique_sorted(pts_axis0);
    std::vector<float> axis1_grid = unique_sorted(pts_axis1);
    const std::size_t  n0 = axis0_grid.size();
    const std::size_t  n1 = axis1_grid.size();

    std::vector<int> grid_idx(n0 * n1, -1);
    const float EPS = 1e-4f;
    for (std::uint32_t i = 0; i < nrows; ++i) {
        int i0 = static_cast<int>(
            std::lower_bound(axis0_grid.begin(), axis0_grid.end(),
                             pts_axis0[i] - EPS) - axis0_grid.begin());
        int i1 = static_cast<int>(
            std::lower_bound(axis1_grid.begin(), axis1_grid.end(),
                             pts_axis1[i] - EPS) - axis1_grid.begin());
        if (i0 < static_cast<int>(n0) && i1 < static_cast<int>(n1))
            grid_idx[i0 * n1 + i1] = static_cast<int>(i);
    }

    return ICTable{K, std::move(axis_names),
                   std::move(pts_axis0), std::move(pts_axis1),
                   std::move(vals),
                   std::move(axis0_grid), std::move(axis1_grid),
                   std::move(grid_idx), n1};
}

void IcBin::save(const ICTable& table,
                 const std::filesystem::path& bin_path) {
    std::ofstream f(bin_path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error(
            "IcBin::save: cannot open " + bin_path.string());

    const auto nrows      = static_cast<std::uint32_t>(table.pts_axis0_.size());
    const auto latent_dim = static_cast<std::uint32_t>(table.k_);

    f.write(reinterpret_cast<const char*>(&IC_MAGIC),   4);
    f.write(reinterpret_cast<const char*>(&IC_VERSION), 4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);
    f.write(reinterpret_cast<const char*>(&latent_dim), 4);

    for (const auto& name : table.axis_names_) {
        const auto name_len = static_cast<std::uint32_t>(name.size());
        f.write(reinterpret_cast<const char*>(&name_len), 4);
        f.write(name.data(), static_cast<std::streamsize>(name_len));
    }

    for (std::uint32_t i = 0; i < nrows; ++i) {
        f.write(reinterpret_cast<const char*>(&table.pts_axis0_[i]), 4);
        f.write(reinterpret_cast<const char*>(&table.pts_axis1_[i]), 4);
        f.write(reinterpret_cast<const char*>(&table.vals_[i * table.k_]),
                static_cast<std::streamsize>(table.k_ * 4));
    }

    if (!f)
        throw std::runtime_error(
            "IcBin::save: write failed for " + bin_path.string());
}

} // namespace rope::io
