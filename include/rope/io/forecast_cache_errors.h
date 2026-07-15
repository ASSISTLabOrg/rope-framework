#pragma once
// Exceptions shared by ForecastGridBin (src/io/forecast_grid_bin.h) and
// MappedForecastGrid (src/io/mapped_forecast_grid.h) — both readers of the
// same on-disk forecast-grid cache file fail the same way.

#include <stdexcept>

namespace rope::io {

// The cache file does not exist — no forecast has been run yet, or the
// configured cache path is wrong. Distinguishes "never forecast" from
// "forecast, but the file is corrupt" (ForecastCacheCorruptError).
struct ForecastCacheMissingError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The cache file exists but has a bad magic, unsupported version, an
// implausible grid shape, or is truncated/short relative to its own header.
struct ForecastCacheCorruptError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace rope::io
