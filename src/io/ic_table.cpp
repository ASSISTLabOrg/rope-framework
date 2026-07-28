#include "rope/io/ic_table.h"
#include "rope/io/ic_bin.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace rope::io {

// ---------------------------------------------------------------------------
// from_file() — dispatch on extension
// ---------------------------------------------------------------------------
ICTable ICTable::from_file(const std::filesystem::path& path) {
    if (path.extension() == ".icbin")
        return IcBin::load(path);
    return ICTable{path};
}

// ---------------------------------------------------------------------------
// load_from_dir() — probe for binary then CSV
// ---------------------------------------------------------------------------
ICTable ICTable::load_from_dir(const std::filesystem::path& dir) {
    auto bin = dir / "ic_table.icbin";
    auto csv = dir / "ic_table.csv";
    if (std::filesystem::exists(bin)) return IcBin::load(bin);
    if (std::filesystem::exists(csv)) return ICTable{csv};
    throw std::runtime_error(
        "ICTable: no IC table found in " + dir.string() +
        " (expected ic_table.icbin or ic_table.csv)");
}

// ---------------------------------------------------------------------------
// Private constructor (used by IcBin::load)
// ---------------------------------------------------------------------------
ICTable::ICTable(int k,
                 std::vector<std::string> axis_names,
                 std::vector<float> pts_axis0,
                 std::vector<float> pts_axis1,
                 std::vector<float> vals,
                 std::vector<float> axis0_grid,
                 std::vector<float> axis1_grid,
                 std::vector<int>   grid_idx,
                 std::size_t        n_axis1)
    : k_(k)
    , axis_names_(std::move(axis_names))
    , pts_axis0_(std::move(pts_axis0))
    , pts_axis1_(std::move(pts_axis1))
    , vals_(std::move(vals))
    , axis0_grid_(std::move(axis0_grid))
    , axis1_grid_(std::move(axis1_grid))
    , grid_idx_(std::move(grid_idx))
    , n_axis1_(n_axis1)
{}

namespace {
bool is_y_column(const std::string& name) {
    if (name.size() < 2 || name[0] != 'y') return false;
    return std::all_of(name.begin() + 1, name.end(),
        [](unsigned char c) { return std::isdigit(c) != 0; });
}
} // namespace

// ---------------------------------------------------------------------------
// CSV constructor — auto-detects the 2 axis columns and K from y1, y2, … columns
// ---------------------------------------------------------------------------
ICTable::ICTable(const std::filesystem::path& csv_path) {
    CsvReader csv(csv_path);
    const std::size_t N = csv.nrows();

    // Auto-detect K: count consecutive y1, y2, ... columns.
    k_ = 0;
    while (csv.has_column("y" + std::to_string(k_ + 1))) ++k_;
    if (k_ == 0)
        throw std::runtime_error(
            "ICTable: no 'y1' column found in " + csv_path.string());

    // Auto-detect the 2 axis columns: any header that isn't "y<N>".
    for (const auto& name : csv.column_names())
        if (!is_y_column(name)) axis_names_.push_back(name);
    if (axis_names_.size() != 2)
        throw std::runtime_error(
            "ICTable: expected exactly 2 axis columns (non-'y<N>' headers) in " +
            csv_path.string() + ", found " + std::to_string(axis_names_.size()));

    pts_axis0_.resize(N);
    pts_axis1_.resize(N);
    vals_.resize(N * k_);

    for (std::size_t i = 0; i < N; ++i) {
        pts_axis0_[i] = csv.get_float(axis_names_[0], i);
        pts_axis1_[i] = csv.get_float(axis_names_[1], i);
        for (int k = 0; k < k_; ++k)
            vals_[i * k_ + k] = csv.get_float("y" + std::to_string(k + 1), i);
    }

    axis0_grid_ = unique_sorted(pts_axis0_);
    axis1_grid_ = unique_sorted(pts_axis1_);

    const float EPS      = 1e-4f;
    const std::size_t n0 = axis0_grid_.size();
    const std::size_t n1 = axis1_grid_.size();
    n_axis1_ = n1;
    grid_idx_.assign(n0 * n1, -1);

    for (std::size_t i = 0; i < N; ++i) {
        int i0 = static_cast<int>(
            std::lower_bound(axis0_grid_.begin(), axis0_grid_.end(),
                             pts_axis0_[i] - EPS) - axis0_grid_.begin());
        int i1 = static_cast<int>(
            std::lower_bound(axis1_grid_.begin(), axis1_grid_.end(),
                             pts_axis1_[i] - EPS) - axis1_grid_.begin());
        if (i0 < static_cast<int>(n0) && i1 < static_cast<int>(n1))
            grid_idx_[i0 * n1 + i1] = static_cast<int>(i);
    }
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------
std::vector<float> ICTable::get_latent_coeffs(std::span<const float> axis_values) const {
    if (axis_values.size() != axis_names_.size())
        throw std::runtime_error(
            "ICTable::get_latent_coeffs: expected " + std::to_string(axis_names_.size()) +
            " axis values, got " + std::to_string(axis_values.size()));

    std::vector<float> out(k_, 0.0f);
    if (bilinear(axis_values[0], axis_values[1], out.data())) return out;
    nearest(axis_values[0], axis_values[1], out.data());
    return out;
}

std::vector<float> ICTable::unique_sorted(const std::vector<float>& v) {
    std::vector<float> u = v;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end(),
                        [](float a, float b){ return std::abs(a - b) < 1e-4f; }),
            u.end());
    return u;
}

const float* ICTable::row_ptr(int i0, int i1) const noexcept {
    if (i0 < 0 || i0 >= static_cast<int>(axis0_grid_.size()) ||
        i1 < 0 || i1 >= static_cast<int>(n_axis1_))
        return nullptr;
    int idx = grid_idx_[i0 * n_axis1_ + i1];
    if (idx < 0) return nullptr;
    return vals_.data() + idx * k_;
}

bool ICTable::bilinear(float a0, float a1, float* out) const {
    auto a0_it = std::lower_bound(axis0_grid_.begin(), axis0_grid_.end(), a0);
    auto a1_it = std::lower_bound(axis1_grid_.begin(), axis1_grid_.end(), a1);
    int i01 = static_cast<int>(a0_it - axis0_grid_.begin());
    int i11 = static_cast<int>(a1_it - axis1_grid_.begin());
    int i00 = i01 - 1, i10 = i11 - 1;

    if (i00 < 0 || i01 >= static_cast<int>(axis0_grid_.size()) ||
        i10 < 0 || i11 >= static_cast<int>(n_axis1_))
        return false;

    const float* v00 = row_ptr(i00, i10);
    const float* v01 = row_ptr(i00, i11);
    const float* v10 = row_ptr(i01, i10);
    const float* v11 = row_ptr(i01, i11);
    if (!v00 || !v01 || !v10 || !v11) return false;

    float ta0 = (a0 - axis0_grid_[i00]) / (axis0_grid_[i01] - axis0_grid_[i00]);
    float ta1 = (a1 - axis1_grid_[i10]) / (axis1_grid_[i11] - axis1_grid_[i10]);
    float w00 = (1-ta0)*(1-ta1), w01 = (1-ta0)*ta1;
    float w10 = ta0*(1-ta1),     w11 = ta0*ta1;

    for (int k = 0; k < k_; ++k)
        out[k] = w00*v00[k] + w01*v01[k] + w10*v10[k] + w11*v11[k];
    return true;
}

void ICTable::nearest(float a0, float a1, float* out) const {
    float best = std::numeric_limits<float>::max();
    int   best_i = 0;
    for (std::size_t i = 0; i < pts_axis0_.size(); ++i) {
        float d0 = pts_axis0_[i] - a0;
        float d1 = pts_axis1_[i] - a1;
        float d  = d0*d0 + d1*d1;
        if (d < best) { best = d; best_i = static_cast<int>(i); }
    }
    const float* src = vals_.data() + best_i * k_;
    for (int k = 0; k < k_; ++k) out[k] = src[k];
}

} // namespace rope::io
