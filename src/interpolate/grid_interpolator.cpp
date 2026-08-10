#include "rope/interpolate/grid_interpolator.h"
#include "rope/io/mapped_forecast_grid.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <tuple>

namespace rope::interpolate {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

static std::vector<double> make_lst_axis(const GridSpec& shape) {
    std::vector<double> v(shape.n_lst);
    for (int i = 0; i < shape.n_lst; ++i)
        v[i] = i * (24.0 / shape.n_lst);
    return v;
}
static std::vector<double> make_lat_axis(const GridSpec& shape) {
    std::vector<double> v(shape.n_lat);
    const double span = shape.lat_max_deg - shape.lat_min_deg;
    for (int i = 0; i < shape.n_lat; ++i)
        v[i] = shape.lat_min_deg + i * (span / (shape.n_lat - 1));
    return v;
}
static std::vector<double> make_alt_axis(const GridSpec& shape) {
    std::vector<double> v(shape.n_alt);
    const double span = shape.alt_max_km - shape.alt_min_km;
    for (int i = 0; i < shape.n_alt; ++i)
        v[i] = shape.alt_min_km + i * (span / (shape.n_alt - 1));
    return v;
}

template <class Grid>
GridInterpolator<Grid>::GridInterpolator(const Grid& grid, ExtrapolationOptions opts)
    : grid_(grid)
    , times_(grid.times)
    , H_(grid.H)
    , lst_ax_(make_lst_axis(grid.shape))
    , lat_ax_(make_lat_axis(grid.shape))
    , alt_ax_(make_alt_axis(grid.shape))
    , lst_step_(24.0 / grid.shape.n_lst)
    , lat_step_((grid.shape.lat_max_deg - grid.shape.lat_min_deg) / (grid.shape.n_lat - 1))
    , alt_step_((grid.shape.alt_max_km  - grid.shape.alt_min_km)  / (grid.shape.n_alt - 1))
{
    if (H_ == 0)
        throw std::runtime_error("GridInterpolator: empty grid");
    precompute_polar_caps();
    set_extrapolation_options(opts);
}

// ---------------------------------------------------------------------------
// Polar caps
// ---------------------------------------------------------------------------

template <class Grid>
float GridInterpolator<Grid>::polar_cap_log10(const float* field, int lat_row, int z,
                                              int n_lst, int n_lat, int n_alt) noexcept {
    double sum = 0.0;
    for (int l = 0; l < n_lst; ++l) {
        std::size_t i = static_cast<std::size_t>(l) * (n_lat * n_alt)
                       + static_cast<std::size_t>(lat_row) * n_alt + z;
        float v = field[i];
        sum += (v > 0.0f) ? std::log10(static_cast<double>(v)) : -300.0;
    }
    return static_cast<float>(sum / n_lst);
}

template <class Grid>
void GridInterpolator<Grid>::precompute_polar_caps() {
    const int n_lst = grid_.shape.n_lst;
    const int n_lat = grid_.shape.n_lat;
    const int n_alt = grid_.shape.n_alt;

    north_cap_density_.resize(static_cast<std::size_t>(H_) * n_alt);
    south_cap_density_.resize(static_cast<std::size_t>(H_) * n_alt);
    north_cap_uncertainty_.resize(static_cast<std::size_t>(H_) * n_alt);
    south_cap_uncertainty_.resize(static_cast<std::size_t>(H_) * n_alt);

    for (int t = 0; t < H_; ++t) {
        const float* d   = grid_.density_at(t);
        const float* u   = grid_.uncertainty_at(t);
        std::size_t  off = static_cast<std::size_t>(t) * n_alt;
        for (int z = 0; z < n_alt; ++z) {
            north_cap_density_[off + z]     = polar_cap_log10(d, n_lat - 1, z, n_lst, n_lat, n_alt);
            south_cap_density_[off + z]     = polar_cap_log10(d, 0,         z, n_lst, n_lat, n_alt);
            north_cap_uncertainty_[off + z] = polar_cap_log10(u, n_lat - 1, z, n_lst, n_lat, n_alt);
            south_cap_uncertainty_[off + z] = polar_cap_log10(u, 0,         z, n_lst, n_lat, n_alt);
        }
    }
}

// ---------------------------------------------------------------------------
// Altitude extrapolation
// ---------------------------------------------------------------------------

template <class Grid>
void GridInterpolator<Grid>::set_extrapolation_options(ExtrapolationOptions opts) {
    const int n_alt = grid_.shape.n_alt;
    if (opts.n_etp_pts < 2 || opts.n_etp_pts > n_alt)
        throw std::invalid_argument(
            "GridInterpolator: n_etp_pts must be in [2, n_alt=" + std::to_string(n_alt) + "]");
    opts_ = opts;
    precompute_altitude_extrapolation();
}

// Least-squares slope/intercept of (x[z], y(z)) over the fit window; x-stats are shared across
// every column since they all fit the same trailing altitude bins.
template <class Grid>
void GridInterpolator<Grid>::precompute_altitude_extrapolation() {
    const int n_lst = grid_.shape.n_lst;
    const int n_lat = grid_.shape.n_lat;
    const int n_alt = grid_.shape.n_alt;
    const int N  = opts_.n_etp_pts;
    const int z0 = n_alt - N;

    double Sx = 0.0, Sxx = 0.0;
    for (int z = z0; z < n_alt; ++z) {
        Sx  += alt_ax_[z];
        Sxx += alt_ax_[z] * alt_ax_[z];
    }
    const double denom = N * Sxx - Sx * Sx;

    auto fit = [&](auto y_at) -> std::pair<float, float> {
        double Sy = 0.0, Sxy = 0.0;
        for (int z = z0; z < n_alt; ++z) {
            double y = y_at(z);
            Sy  += y;
            Sxy += alt_ax_[z] * y;
        }
        double slope     = (N * Sxy - Sx * Sy) / denom;
        double intercept = (Sy - slope * Sx) / N;
        return {static_cast<float>(slope), static_cast<float>(intercept)};
    };
    auto logv_floor = [](float v) -> double {
        return (v > 0.0f) ? std::log10(static_cast<double>(v)) : -300.0;
    };

    const std::size_t n_cols = static_cast<std::size_t>(H_) * n_lst * n_lat;
    slope_density_.resize(n_cols);       intercept_density_.resize(n_cols);
    slope_uncertainty_.resize(n_cols);   intercept_uncertainty_.resize(n_cols);
    north_cap_slope_density_.resize(H_);      north_cap_intercept_density_.resize(H_);
    south_cap_slope_density_.resize(H_);      south_cap_intercept_density_.resize(H_);
    north_cap_slope_uncertainty_.resize(H_);  north_cap_intercept_uncertainty_.resize(H_);
    south_cap_slope_uncertainty_.resize(H_);  south_cap_intercept_uncertainty_.resize(H_);

    for (int t = 0; t < H_; ++t) {
        const float* d = grid_.density_at(t);
        const float* u = grid_.uncertainty_at(t);
        const std::size_t col_off = static_cast<std::size_t>(t) * n_lst * n_lat;

        for (int l = 0; l < n_lst; ++l) {
            for (int a = 0; a < n_lat; ++a) {
                std::size_t out = col_off + static_cast<std::size_t>(l) * n_lat + a;
                auto d_at = [&](int z) {
                    return logv_floor(d[static_cast<std::size_t>(l) * (n_lat * n_alt)
                                        + static_cast<std::size_t>(a) * n_alt + z]);
                };
                auto u_at = [&](int z) {
                    return logv_floor(u[static_cast<std::size_t>(l) * (n_lat * n_alt)
                                        + static_cast<std::size_t>(a) * n_alt + z]);
                };
                std::tie(slope_density_[out], intercept_density_[out]) = fit(d_at);
                std::tie(slope_uncertainty_[out], intercept_uncertainty_[out]) = fit(u_at);
            }
        }

        // The pole caps are already log10-space per-altitude values; fit them directly, no floor.
        const std::size_t cap_off = static_cast<std::size_t>(t) * n_alt;
        auto cap_at = [](const std::vector<float>& cap, std::size_t off) {
            return [&cap, off](int z) { return static_cast<double>(cap[off + z]); };
        };
        std::tie(north_cap_slope_density_[t], north_cap_intercept_density_[t])
            = fit(cap_at(north_cap_density_, cap_off));
        std::tie(south_cap_slope_density_[t], south_cap_intercept_density_[t])
            = fit(cap_at(south_cap_density_, cap_off));
        std::tie(north_cap_slope_uncertainty_[t], north_cap_intercept_uncertainty_[t])
            = fit(cap_at(north_cap_uncertainty_, cap_off));
        std::tie(south_cap_slope_uncertainty_[t], south_cap_intercept_uncertainty_[t])
            = fit(cap_at(south_cap_uncertainty_, cap_off));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <class Grid>
void GridInterpolator<Grid>::check_spatial(double lst, double lat, double alt_km) const {
    (void)lst;  // LST is periodic — any value is valid
    // Only latitudes outside the true physical range +/-90 are rejected here.
    if (lat < -90.0 || lat > 90.0) {
        std::ostringstream oss;
        oss << "Requested latitude " << lat
            << " is outside the physical range [-90, 90]";
        throw SpatialOutOfRangeError(oss.str());
    }
    // The floor at alt_min_km is always hard, regardless of extrapolation.
    if (alt_km < alt_ax_.front()) {
        std::ostringstream oss;
        oss << "Requested altitude " << alt_km
            << " km is below the ROPE grid floor of " << alt_ax_.front() << " km";
        throw SpatialOutOfRangeError(oss.str());
    }
    if (alt_km > alt_ax_.back()) {
        if (!opts_.extrapolate_altitude) {
            std::ostringstream oss;
            oss << "Requested altitude " << alt_km
                << " km is above the ROPE grid bound of " << alt_ax_.back()
                << " km and altitude extrapolation is disabled";
            throw SpatialOutOfRangeError(oss.str());
        }
        if (alt_km > kMaxExtrapolationAltKm) {
            std::ostringstream oss;
            oss << "Requested altitude " << alt_km
                << " km exceeds the extrapolation ceiling of " << kMaxExtrapolationAltKm << " km";
            throw SpatialOutOfRangeError(oss.str());
        }
    }
}

template <class Grid>
void GridInterpolator<Grid>::check_time(TimePoint tp) const {
    if (tp < times_.front() || tp > times_.back()) {
        std::ostringstream oss;
        oss << "Requested time " << format_iso(tp)
            << " is outside the forecast window ["
            << format_iso(times_.front()) << ", "
            << format_iso(times_.back()) << "]";
        throw TimeOutOfRangeError(oss.str());
    }
}

template <class Grid>
std::pair<int, int> GridInterpolator<Grid>::bracket(TimePoint tp) const {
    auto it = std::lower_bound(times_.begin(), times_.end(), tp);
    int i1 = static_cast<int>(it - times_.begin());
    if (it != times_.end() && *it == tp) return {i1, i1};
    return {i1 - 1, i1};
}

template <class Grid>
int GridInterpolator<Grid>::lower_idx(const std::vector<double>& ax, double v) noexcept {
    auto it = std::lower_bound(ax.begin(), ax.end(), v);
    int i   = static_cast<int>(it - ax.begin());
    return std::max(0, std::min(i - 1, static_cast<int>(ax.size()) - 2));
}

// ---------------------------------------------------------------------------
// Trilinear interpolation in log10 space
// ---------------------------------------------------------------------------

template <class Grid>
double GridInterpolator<Grid>::spatial_interp(const FieldView& f,
                                               double lst, double lat, double alt_km) const {
    const int n_lst = grid_.shape.n_lst;
    const int n_lat = grid_.shape.n_lat;
    const int n_alt = grid_.shape.n_alt;

    // --- LST: uniform step, periodic ---
    lst = lst - std::floor(lst / 24.0) * 24.0;  // normalize to [0, 24)
    int li0 = static_cast<int>(lst / lst_step_);
    if (li0 >= n_lst) li0 = n_lst - 1;
    int    li1  = (li0 + 1) % n_lst;             // wraps n_lst-1 → 0
    double lst1 = (li1 == 0) ? 24.0 : lst_ax_[li1];
    double wl   = (lst - lst_ax_[li0]) / (lst1 - lst_ax_[li0]);

    // --- Latitude: uniform in-grid, polar-cap beyond ---
    int ai0, ai1;
    double a_lo, a_hi;
    if (lat > lat_ax_.back()) {
        ai0 = n_lat - 1; ai1 = n_lat;             // ai1 is the virtual north-pole slot
        a_lo = lat_ax_.back(); a_hi = 90.0;
    } else if (lat < lat_ax_.front()) {
        ai0 = -1; ai1 = 0;                        // ai0 is the virtual south-pole slot
        a_lo = -90.0; a_hi = lat_ax_.front();
    } else {
        ai0 = static_cast<int>((lat - lat_ax_.front()) / lat_step_);
        ai0 = std::max(0, std::min(ai0, n_lat - 2));
        ai1 = ai0 + 1;
        a_lo = lat_ax_[ai0]; a_hi = lat_ax_[ai1];
    }
    double wa = (lat - a_lo) / (a_hi - a_lo);

    if (alt_km > alt_ax_.back()) {
        // Extrapolation: each (lat slot, LST) corner's own local log-linear fit evaluated at
        // alt_km, blended with the same LST-then-latitude weights as the in-grid case below --
        // valid because the fit is linear in its parameters, so blend-then-evaluate and
        // evaluate-then-blend agree exactly.
        auto logv_extrap = [&](int a, int l) -> double {
            if (a < 0)      return f.south_intercept + f.south_slope * alt_km;
            if (a >= n_lat) return f.north_intercept + f.north_slope * alt_km;
            std::size_t i = static_cast<std::size_t>(l) * n_lat + a;
            return f.intercept[i] + f.slope[i] * alt_km;
        };
        double c_a0 = lerp(logv_extrap(ai0,li0), logv_extrap(ai0,li1), wl);
        double c_a1 = lerp(logv_extrap(ai1,li0), logv_extrap(ai1,li1), wl);
        return std::pow(10.0, lerp(c_a0, c_a1, wa));
    }

    // --- Altitude: uniform ---
    int zi0 = static_cast<int>((alt_km - alt_ax_.front()) / alt_step_);
    zi0 = std::max(0, std::min(zi0, n_alt - 2));
    int    zi1 = zi0 + 1;
    double wz  = (alt_km - alt_ax_[zi0]) / (alt_ax_[zi1] - alt_ax_[zi0]);

    auto idx = [n_lat, n_alt](int l, int a, int z) -> std::size_t {
        return static_cast<std::size_t>(l) * (n_lat * n_alt)
             + static_cast<std::size_t>(a) * n_alt
             + z;
    };

    // Log10 value at lat slot `a`, floored at -300; a pole slot (a<0 or a>=n_lat) is LST-invariant.
    auto logv = [&](int a, int l, int z) -> double {
        if (a < 0)      return f.south_cap[z];
        if (a >= n_lat) return f.north_cap[z];
        float v = f.base[idx(l, a, z)];
        return (v > 0.0f) ? std::log10(static_cast<double>(v)) : -300.0;
    };

    double c00 = lerp(logv(ai0,li0,zi0), logv(ai0,li1,zi0), wl);
    double c01 = lerp(logv(ai0,li0,zi1), logv(ai0,li1,zi1), wl);
    double c10 = lerp(logv(ai1,li0,zi0), logv(ai1,li1,zi0), wl);
    double c11 = lerp(logv(ai1,li0,zi1), logv(ai1,li1,zi1), wl);
    double c0  = lerp(c00, c10, wa);
    double c1  = lerp(c01, c11, wa);
    return std::pow(10.0, lerp(c0, c1, wz));
}

template <class Grid>
InterpolationResult GridInterpolator<Grid>::spatial_both(
    int t, double lst, double lat, double alt_km) const
{
    const int n_lst = grid_.shape.n_lst;
    const int n_lat = grid_.shape.n_lat;
    const std::size_t cap_off = static_cast<std::size_t>(t) * grid_.shape.n_alt;
    const std::size_t col_off = static_cast<std::size_t>(t) * n_lst * n_lat;

    FieldView fd{
        grid_.density_at(t),
        &north_cap_density_[cap_off], &south_cap_density_[cap_off],
        &slope_density_[col_off],     &intercept_density_[col_off],
        north_cap_slope_density_[t],  north_cap_intercept_density_[t],
        south_cap_slope_density_[t],  south_cap_intercept_density_[t],
    };
    FieldView fu{
        grid_.uncertainty_at(t),
        &north_cap_uncertainty_[cap_off], &south_cap_uncertainty_[cap_off],
        &slope_uncertainty_[col_off],     &intercept_uncertainty_[col_off],
        north_cap_slope_uncertainty_[t],  north_cap_intercept_uncertainty_[t],
        south_cap_slope_uncertainty_[t],  south_cap_intercept_uncertainty_[t],
    };

    return {
        spatial_interp(fd, lst, lat, alt_km),
        spatial_interp(fu, lst, lat, alt_km),
    };
}

// ---------------------------------------------------------------------------
// Public query methods
// ---------------------------------------------------------------------------

template <class Grid>
InterpolationResult GridInterpolator<Grid>::query_hold(
    TimePoint time_unix, double lst, double lat, double alt_km) const
{
    check_time(time_unix);
    check_spatial(lst, lat, alt_km);

    auto [i0, i1] = bracket(time_unix);
    int use_i     = (times_[i0] == time_unix) ? i0 : i1;  // ceil
    return spatial_both(use_i, lst, lat, alt_km);
}

template <class Grid>
InterpolationResult GridInterpolator<Grid>::query_interp(
    TimePoint time_unix, double lst, double lat, double alt_km) const
{
    check_time(time_unix);
    check_spatial(lst, lat, alt_km);

    auto [i0, i1] = bracket(time_unix);

    if (i0 == i1)
        return spatial_both(i0, lst, lat, alt_km);

    double span = static_cast<double>(times_[i1] - times_[i0]);
    double off  = static_cast<double>(time_unix   - times_[i0]);
    double w    = off / span;

    auto r0 = spatial_both(i0, lst, lat, alt_km);
    auto r1 = spatial_both(i1, lst, lat, alt_km);
    return {
        lerp(r0.density,     r1.density,     w),
        lerp(r0.uncertainty, r1.uncertainty, w),
    };
}

// ---------------------------------------------------------------------------
// Explicit instantiation — the only two grid backings that exist.
// ---------------------------------------------------------------------------
template class GridInterpolator<ForecastGrid>;
template class GridInterpolator<io::MappedForecastGrid>;

} // namespace rope::interpolate
