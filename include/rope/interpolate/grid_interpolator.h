#pragma once
// GridInterpolator — spatiotemporal density/uncertainty query on a ForecastGrid.
// Grid axes are per-model (rope::GridSpec), not fixed constants. Spatial interpolation is trilinear in log10 space.
// HOLD: snap to next model hour. INTERP: trilinear at both bracket hours, then linear time blend.
// Latitude beyond [lat_min_deg, lat_max_deg] blends toward a polar-cap average out to +/-90.
// Altitude above alt_max_km optionally extrapolates log-linearly out to a hard ceiling; see ExtrapolationOptions.

#include "rope/core/datetime.h"
#include "rope/core/types.h"
#include "rope/io/config_reader.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace rope::interpolate {

struct TimeOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct SpatialOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct InterpolationResult {
    double density;      // kg/m³
    double uncertainty;  // kg/m³
};

// Controls log-linear extrapolation past alt_max_km. Does not affect the alt_min_km floor,
// latitude, or the hard 2000 km ceiling (GridInterpolator::kMaxExtrapolationAltKm).
struct ExtrapolationOptions {
    bool extrapolate_altitude = true;
    int  n_etp_pts = 8;  // near-boundary altitude bins used to fit the local log-linear slope
};

// Reads [interpolation] extrapolate_altitude/n_etp_pts; falls back to ExtrapolationOptions{} defaults when absent.
ExtrapolationOptions options_from_reader(const io::ConfigReader& config);

// Templated, not virtual, for zero-dispatch-overhead queries over either ForecastGrid or MappedForecastGrid.
// Grid must expose: GridSpec shape, int H, times, density_at(t), uncertainty_at(t). Explicitly instantiated in grid_interpolator.cpp.
template <class Grid>
class GridInterpolator {
public:
    // Hard ceiling for altitude extrapolation, regardless of ExtrapolationOptions; never configurable.
    static constexpr double kMaxExtrapolationAltKm = 2000.0;

    explicit GridInterpolator(const Grid& grid, ExtrapolationOptions opts = {});

    TimePoint time_min() const noexcept { return times_.front(); }
    TimePoint time_max() const noexcept { return times_.back();  }

    // Reconfigures altitude extrapolation; throws std::invalid_argument if n_etp_pts is out of [2, n_alt].
    // Not safe to call concurrently with query_hold/query_interp on the same instance.
    void set_extrapolation_options(ExtrapolationOptions opts);
    const ExtrapolationOptions& extrapolation_options() const noexcept { return opts_; }

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
    ExtrapolationOptions   opts_;

    // Polar-cap averages (log10 space), one per (t, alt); precomputed once at construction.
    std::vector<float> north_cap_density_, south_cap_density_;
    std::vector<float> north_cap_uncertainty_, south_cap_uncertainty_;

    // Per-column altitude-extrapolation fit (log10 space, slope in /km), one per (t, lst, lat).
    std::vector<float> slope_density_, intercept_density_;
    std::vector<float> slope_uncertainty_, intercept_uncertainty_;

    // Polar-cap altitude-extrapolation fit (log10 space), one per t; LST/lat-independent like the caps themselves.
    std::vector<float> north_cap_slope_density_, north_cap_intercept_density_;
    std::vector<float> south_cap_slope_density_, south_cap_intercept_density_;
    std::vector<float> north_cap_slope_uncertainty_, north_cap_intercept_uncertainty_;
    std::vector<float> south_cap_slope_uncertainty_, south_cap_intercept_uncertainty_;

    void check_spatial(double lst, double lat, double alt_km) const;
    void check_time(TimePoint tp) const;
    std::pair<int, int> bracket(TimePoint tp) const;

    // Fills north/south_cap_* from the grid's boundary latitude rows.
    void precompute_polar_caps();

    // Fills the slope_*/intercept_* and north/south_cap_slope_*/intercept_* arrays from opts_.n_etp_pts.
    void precompute_altitude_extrapolation();

    // Zonal mean of log10(field) around latitude row `lat_row` at altitude z, for one timestep.
    static float polar_cap_log10(const float* field, int lat_row, int z,
                                 int n_lst, int n_lat, int n_alt) noexcept;

    // One field's per-timestep data needed by spatial_interp: real grid values, polar-cap values,
    // and both fields' altitude-extrapolation fits (real per-column and polar-cap).
    struct FieldView {
        const float* base;
        const float* north_cap;
        const float* south_cap;
        const float* slope;
        const float* intercept;
        float north_slope, north_intercept, south_slope, south_intercept;
    };

    // Trilinear interpolation in log10 space for one field; extrapolates past alt_max_km via f's fit when in range.
    double spatial_interp(const FieldView& f, double lst, double lat, double alt_km) const;

    // Interpolate both density and uncertainty at a fixed time step.
    InterpolationResult spatial_both(int t,
                                     double lst, double lat, double alt_km) const;

    static double lerp(double a, double b, double w) noexcept { return a + w*(b-a); }
    static int    lower_idx(const std::vector<double>& ax, double v) noexcept;
};

} // namespace rope::interpolate
