#pragma once
// Shared plain-data types used across all modules.

#include <cstdint>
#include <vector>

namespace rope {

// Physical shape of a model's output grid; per-model (io::ModelManifest::grid), not a fixed global shape.
struct GridSpec {
    int n_lst = 0;
    int n_lat = 0;
    int n_alt = 0;
    double lat_min_deg = 0.0;
    double lat_max_deg = 0.0;
    double alt_min_km  = 0.0;
    double alt_max_km  = 0.0;

    int voxels() const noexcept { return n_lst * n_lat * n_alt; }
};

// In-memory forecast (produced by forecast/, consumed by interpolate/ and capi/). density/uncertainty: row-major [t, lst, lat, alt], kg/m³; times: UTC seconds.
struct ForecastGrid {
    GridSpec                   shape;
    std::vector<float>        density;      // H * shape.voxels() floats
    std::vector<float>        uncertainty;  // H * shape.voxels() floats
    std::vector<std::int64_t> times;        // H timestamps (seconds since epoch)
    int H = 0;

    const float* density_at(int t) const noexcept {
        return density.data() + static_cast<std::size_t>(t) * shape.voxels();
    }
    const float* uncertainty_at(int t) const noexcept {
        return uncertainty.data() + static_cast<std::size_t>(t) * shape.voxels();
    }
};

} // namespace rope
