#pragma once
// Shared plain-data types used across all modules.

#include <cstdint>
#include <vector>

namespace rope {

// ---------------------------------------------------------------------------
// GridSpec — physical shape of a model's output grid: bin counts on all
// three axes, plus the physical range of lat/alt (LST's range is always the
// full 24h cycle by definition of what LST is, so only its count varies).
// Declared per model in model_manifest.json (see io::ModelManifest::grid) —
// different trained models may target different physical grids, so this is
// no longer a single fixed shape shared by every model.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// ForecastGrid — in-memory forecast produced by forecast/ and consumed by
// interpolate/ and capi/.
//
// Both density and uncertainty are stored row-major [t, lst, lat, alt];
// alt is the fastest axis.  Units: kg/m³.
// Times are UTC seconds since the Unix epoch, one per forecast hour.
// ---------------------------------------------------------------------------
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
