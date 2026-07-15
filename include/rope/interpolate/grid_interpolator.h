#pragma once
// GridInterpolator — spatiotemporal density and uncertainty query on a ForecastGrid.
//
// Grid axes are per-model, declared in model_manifest.json (rope::GridSpec)
// and carried on the queried ForecastGrid — not fixed constants. LST is
// always uniform over the full 24h cycle (n_lst bins); lat and alt are
// uniform over [lat_min_deg, lat_max_deg] / [alt_min_km, alt_max_km].
//
// Spatial interpolation is performed in log10 space, then exponentiated.
// Both density and uncertainty use the same spatial weights.
//
// Time modes:
//   HOLD   — snap to the next model hour; no temporal blending.
//   INTERP — trilinear spatial at both bracket hours, then linear time blend.

#include "rope/core/datetime.h"
#include "rope/core/types.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace rope::interpolate {

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------
struct TimeOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct SpatialOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------
struct InterpolationResult {
    double density;      // kg/m³
    double uncertainty;  // kg/m³
};

// ---------------------------------------------------------------------------
// GridInterpolator<Grid> — templated (not virtual) so both the vector-backed
// ForecastGrid and the mmap-backed MappedForecastGrid (rope/io/mapped_forecast_grid.h)
// can be interpolated through identical code with zero dispatch overhead —
// query_hold/query_interp is a hot path (called once per orbit-integration
// timestep), and CLAUDE.md says to avoid virtual dispatch there.
//
// Grid must expose: `GridSpec shape`, `int H`, `std::vector<std::int64_t> times`,
// `const float* density_at(int t) const`, `const float* uncertainty_at(int t) const`
// — the exact shape ForecastGrid already has. Explicitly instantiated for
// both grid types in grid_interpolator.cpp.
// ---------------------------------------------------------------------------
template <class Grid>
class GridInterpolator {
public:
    explicit GridInterpolator(const Grid& grid);

    TimePoint time_min() const noexcept { return times_.front(); }
    TimePoint time_max() const noexcept { return times_.back();  }

    // Hold: snap to the next model hour, then spatial interpolate.
    InterpolationResult query_hold(
        TimePoint time_unix, double lst, double lat, double alt_km) const;

    // Interp: spatial interp at both bracket hours, then linear time blend.
    InterpolationResult query_interp(
        TimePoint time_unix, double lst, double lat, double alt_km) const;

private:
    const Grid&            grid_;
    std::vector<TimePoint> times_;
    int                    H_;
    std::vector<double>    lst_ax_, lat_ax_, alt_ax_;
    double                 lst_step_, lat_step_, alt_step_;

    void check_spatial(double lst, double lat, double alt_km) const;
    void check_time(TimePoint tp) const;
    std::pair<int, int> bracket(TimePoint tp) const;

    // Trilinear interpolation in log10 space at time step t, for a given field.
    double spatial_interp(const float* time_slice,
                          double lst, double lat, double alt_km) const;

    // Interpolate both density and uncertainty at a fixed time step.
    InterpolationResult spatial_both(int t,
                                     double lst, double lat, double alt_km) const;

    static double lerp(double a, double b, double w) noexcept { return a + w*(b-a); }
    static int    lower_idx(const std::vector<double>& ax, double v) noexcept;
};

} // namespace rope::interpolate
