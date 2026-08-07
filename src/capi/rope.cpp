// C API implementation — wraps io::MappedForecastGrid + interpolate for
// in-process queries.
//
// Exception boundary: every public function catches all exceptions and
// converts them to return codes + err_buf.  No C++ exception may cross
// the C boundary.

#include "rope/capi/rope.h"

#include "rope/core/datetime.h"
#include "rope/core/platform.h"
#include "rope/interpolate/grid_interpolator.h"
#include "rope/io/mapped_forecast_grid.h"
#include "rope/io/model_manifest.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

struct rope_interp {
    rope::io::MappedForecastGrid                          grid;
    rope::interpolate::GridInterpolator<rope::io::MappedForecastGrid> interp;

    explicit rope_interp(rope::io::MappedForecastGrid g)
        : grid(std::move(g)), interp(grid) {}
};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

namespace {

void fill_err(char* err_buf, int err_len, const char* msg) noexcept {
    if (err_buf && err_len > 0) {
        std::strncpy(err_buf, msg, static_cast<std::size_t>(err_len) - 1);
        err_buf[err_len - 1] = '\0';
    }
}

int classify_exception(const std::exception* e) noexcept {
    if (!e) return ROPE_ERR_INTERNAL;
    if (dynamic_cast<const rope::io::ForecastCacheMissingError*>(e)) return ROPE_ERR_NO_FORECAST;
    if (dynamic_cast<const rope::io::ForecastCacheCorruptError*>(e)) return ROPE_ERR_CACHE_CORRUPT;
    return ROPE_ERR_INTERNAL;
}

} // namespace

// ---------------------------------------------------------------------------
// rope_open
// ---------------------------------------------------------------------------

rope_interp_t* rope_open(const char* cache_path, char* err_buf, int err_len) {
    try {
        auto path = cache_path
            ? std::filesystem::path{cache_path}
            : rope::platform::default_forecast_cache_path();

        rope::io::MappedForecastGrid grid = rope::io::MappedForecastGrid::open(path);
        return new rope_interp{std::move(grid)};
    } catch (const std::exception& e) {
        fill_err(err_buf, err_len, e.what());
    } catch (...) {
        fill_err(err_buf, err_len, "rope_open: unknown error");
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// rope_query
// ---------------------------------------------------------------------------

int rope_query(rope_interp_t* interp,
               int mode,
               double time_unix,
               double lst, double lat, double alt_km,
               double* density, double* uncertainty,
               char* err_buf, int err_len) {
    if (!interp || !density || !uncertainty) {
        fill_err(err_buf, err_len, "rope_query: NULL argument");
        return ROPE_ERR_BAD_ARG;
    }
    try {
        auto tp = static_cast<rope::TimePoint>(time_unix);
        rope::interpolate::InterpolationResult r;
        if (mode == ROPE_HOLD)
            r = interp->interp.query_hold(tp, lst, lat, alt_km);
        else
            r = interp->interp.query_interp(tp, lst, lat, alt_km);
        *density     = r.density;
        *uncertainty = r.uncertainty;
        return ROPE_OK;
    } catch (const rope::interpolate::TimeOutOfRangeError& e) {
        fill_err(err_buf, err_len, e.what());
        return ROPE_ERR_TIME_RANGE;
    } catch (const rope::interpolate::SpatialOutOfRangeError& e) {
        fill_err(err_buf, err_len, e.what());
        return ROPE_ERR_SPATIAL_RANGE;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_len, e.what());
        return classify_exception(&e);
    } catch (...) {
        fill_err(err_buf, err_len, "rope_query: unknown error");
        return ROPE_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// rope_query_batch
// ---------------------------------------------------------------------------

int rope_query_batch(rope_interp_t* interp,
                     int mode, int n,
                     const double* times_unix,
                     const double* lst,
                     const double* lat,
                     const double* alt_km,
                     double* density,
                     double* uncertainty,
                     char* err_buf, int err_len) {
    if (!interp || n < 0 || !times_unix || !lst || !lat || !alt_km
                 || !density || !uncertainty) {
        fill_err(err_buf, err_len, "rope_query_batch: NULL or invalid argument");
        return ROPE_ERR_BAD_ARG;
    }
    try {
        for (int i = 0; i < n; ++i) {
            auto tp = static_cast<rope::TimePoint>(times_unix[i]);
            rope::interpolate::InterpolationResult r;
            if (mode == ROPE_HOLD)
                r = interp->interp.query_hold(tp, lst[i], lat[i], alt_km[i]);
            else
                r = interp->interp.query_interp(tp, lst[i], lat[i], alt_km[i]);
            density[i]     = r.density;
            uncertainty[i] = r.uncertainty;
        }
        return ROPE_OK;
    } catch (const rope::interpolate::TimeOutOfRangeError& e) {
        fill_err(err_buf, err_len, e.what());
        return ROPE_ERR_TIME_RANGE;
    } catch (const rope::interpolate::SpatialOutOfRangeError& e) {
        fill_err(err_buf, err_len, e.what());
        return ROPE_ERR_SPATIAL_RANGE;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_len, e.what());
        return classify_exception(&e);
    } catch (...) {
        fill_err(err_buf, err_len, "rope_query_batch: unknown error");
        return ROPE_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// rope_close
// ---------------------------------------------------------------------------

void rope_close(rope_interp_t* interp) {
    delete interp;
}

// ---------------------------------------------------------------------------
// rope_get_manifest_info
// ---------------------------------------------------------------------------

int rope_get_manifest_info(const char* exported_dir,
                           char* buf, int buf_len,
                           char* err_buf, int err_len) {
    if (!exported_dir || !buf || buf_len <= 0) {
        fill_err(err_buf, err_len, "rope_get_manifest_info: NULL or invalid argument");
        return ROPE_ERR_BAD_ARG;
    }
    try {
        auto manifest = rope::io::ModelManifest::load(std::filesystem::path{exported_dir});
        std::string summary = manifest.to_summary_json();

        if (static_cast<int>(summary.size()) >= buf_len) {
            fill_err(err_buf, err_len,
                     ("rope_get_manifest_info: buffer too small, need at least " +
                      std::to_string(summary.size() + 1) + " bytes").c_str());
            return ROPE_ERR_BUFFER_TOO_SMALL;
        }
        std::memcpy(buf, summary.data(), summary.size());
        buf[summary.size()] = '\0';
        return ROPE_OK;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_len, e.what());
        return ROPE_ERR_INTERNAL;
    } catch (...) {
        fill_err(err_buf, err_len, "rope_get_manifest_info: unknown error");
        return ROPE_ERR_INTERNAL;
    }
}
