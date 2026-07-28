#pragma once
// IC lookup table: 2-axis grid of latent initial-condition coefficients.
//
// Columns: <axis0_name>, <axis1_name>, y1 … yK. Axis names are not fixed —
// auto-detected from the CSV header (any column that isn't "y<N>") or read
// from the .icbin file's own name table; cross-checked against the
// manifest's ic.params.grid_axes by the caller (see forecast::make_ic_source
// and StackedEnsemblePipeline::load_ic_source).
//
// Interpolation: bilinear on the 2-axis grid; nearest-neighbour fallback
// when the query is outside the convex hull.

#include "rope/io/csv_reader.h"

#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace rope::io {

class IcBin;  // forward-declare for friend

class ICTable {
public:
    // Load from file; format auto-detected from extension.
    // ".icbin" → binary; anything else → CSV.
    static ICTable from_file(const std::filesystem::path& path);

    // Probe dir for ic_table.icbin then ic_table.csv; throws if neither exists.
    static ICTable load_from_dir(const std::filesystem::path& dir);

    // Auto-detects the table's 2 axis names from the CSV header (any column
    // that isn't "y<N>"); throws unless exactly 2 such columns are present.
    explicit ICTable(const std::filesystem::path& csv_path);

    // axis_values.size() must equal axis_names().size() (2 today); throws otherwise.
    std::vector<float> get_latent_coeffs(std::span<const float> axis_values) const;

    const std::vector<std::string>& axis_names() const noexcept { return axis_names_; }
    int latent_dim() const noexcept { return k_; }

private:
    friend class IcBin;

    // Private constructor used by IcBin::load().
    ICTable(int k,
            std::vector<std::string> axis_names,
            std::vector<float> pts_axis0,
            std::vector<float> pts_axis1,
            std::vector<float> vals,
            std::vector<float> axis0_grid,
            std::vector<float> axis1_grid,
            std::vector<int>   grid_idx,
            std::size_t        n_axis1);

    int                       k_ = 0;  // latent dimension (runtime)
    std::vector<std::string> axis_names_;  // exactly 2 today, in axis order
    std::vector<float>       pts_axis0_, pts_axis1_;
    std::vector<float>       vals_;       // (N, k_) row-major
    std::vector<float>       axis0_grid_, axis1_grid_;
    std::vector<int>         grid_idx_;   // -1 = missing cell
    std::size_t              n_axis1_ = 0;

    static std::vector<float> unique_sorted(const std::vector<float>& v);
    const float* row_ptr(int i0, int i1) const noexcept;
    bool bilinear(float a0, float a1, float* out) const;
    void nearest(float a0, float a1, float* out) const;
};

} // namespace rope::io
