#pragma once
// IC table: (F10, Kp) → 10-dimensional latent initial condition coefficients.
//
// Columns: F10, Kp, y1 … y10.
//
// Interpolation: bilinear on the (F10, Kp) grid; nearest-neighbour fallback
// when the query is outside the convex hull.

#include "rope/io/csv_reader.h"

#include <filesystem>
#include <limits>
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

    explicit ICTable(const std::filesystem::path& csv_path);

    // Returns latent_dim() coefficients for (f10, kp).
    std::vector<float> get_latent_coeffs(float f10, float kp) const;

    int latent_dim() const noexcept { return k_; }

private:
    friend class IcBin;

    // Private constructor used by IcBin::load().
    ICTable(int k,
            std::vector<float> pts_f10,
            std::vector<float> pts_kp,
            std::vector<float> vals,
            std::vector<float> f10_axis,
            std::vector<float> kp_axis,
            std::vector<int>   grid_idx,
            std::size_t        n_kp);

    int                k_ = 10;  // latent dimension (runtime)
    std::vector<float> pts_f10_, pts_kp_;
    std::vector<float> vals_;       // (N, k_) row-major
    std::vector<float> f10_axis_, kp_axis_;
    std::vector<int>   grid_idx_;   // -1 = missing cell
    std::size_t        n_kp_ = 0;

    static std::vector<float> unique_sorted(const std::vector<float>& v);
    const float* row_ptr(int fi, int ki) const noexcept;
    bool bilinear(float f10, float kp, float* out) const;
    void nearest(float f10, float kp, float* out) const;
};

} // namespace rope::io
