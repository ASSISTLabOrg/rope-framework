#include "rope/interpolate/grid_interpolator.h"
#include "rope/io/mapped_forecast_grid.h"

#include <algorithm>
#include <cmath>
#include <sstream>

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
GridInterpolator<Grid>::GridInterpolator(const Grid& grid)
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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <class Grid>
void GridInterpolator<Grid>::check_spatial(double lst, double lat, double alt_km) const {
    (void)lst;  // LST is periodic — any value is valid
    if (lat < lat_ax_.front() || lat > lat_ax_.back()) {
        std::ostringstream oss;
        oss << "Requested latitude " << lat
            << " is outside the ROPE grid bounds ["
            << lat_ax_.front() << ", " << lat_ax_.back() << "]";
        throw SpatialOutOfRangeError(oss.str());
    }
    if (alt_km < alt_ax_.front() || alt_km > alt_ax_.back()) {
        std::ostringstream oss;
        oss << "Requested altitude " << alt_km
            << " km is outside the ROPE grid bounds ["
            << alt_ax_.front() << ", " << alt_ax_.back() << "] km";
        throw SpatialOutOfRangeError(oss.str());
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
double GridInterpolator<Grid>::spatial_interp(const float* base,
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

    // --- Latitude: uniform ---
    int ai0 = static_cast<int>((lat - lat_ax_.front()) / lat_step_);
    ai0 = std::max(0, std::min(ai0, n_lat - 2));
    int    ai1 = ai0 + 1;
    double wa  = (lat - lat_ax_[ai0]) / (lat_ax_[ai1] - lat_ax_[ai0]);

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

    // Interpolate in log10 space; floor at -300 to guard against underflow zeros.
    auto logv = [&](int l, int a, int z) -> double {
        float v = base[idx(l, a, z)];
        return (v > 0.0f) ? std::log10(static_cast<double>(v)) : -300.0;
    };

    double c00 = lerp(logv(li0,ai0,zi0), logv(li1,ai0,zi0), wl);
    double c01 = lerp(logv(li0,ai0,zi1), logv(li1,ai0,zi1), wl);
    double c10 = lerp(logv(li0,ai1,zi0), logv(li1,ai1,zi0), wl);
    double c11 = lerp(logv(li0,ai1,zi1), logv(li1,ai1,zi1), wl);
    double c0  = lerp(c00, c10, wa);
    double c1  = lerp(c01, c11, wa);
    return std::pow(10.0, lerp(c0, c1, wz));
}

template <class Grid>
InterpolationResult GridInterpolator<Grid>::spatial_both(
    int t, double lst, double lat, double alt_km) const
{
    const float* d = grid_.density_at(t);
    const float* u = grid_.uncertainty_at(t);
    return {
        spatial_interp(d, lst, lat, alt_km),
        spatial_interp(u, lst, lat, alt_km),
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
