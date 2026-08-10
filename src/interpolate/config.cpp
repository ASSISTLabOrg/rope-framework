#include "rope/interpolate/grid_interpolator.h"

namespace rope::interpolate {

ExtrapolationOptions options_from_reader(const io::ConfigReader& config) {
    ExtrapolationOptions opts;
    opts.extrapolate_altitude = config.get("interpolation.extrapolate_altitude", "true") == "true";
    opts.n_etp_pts = config.get_int("interpolation.n_etp_pts", opts.n_etp_pts);
    return opts;
}

} // namespace rope::interpolate
