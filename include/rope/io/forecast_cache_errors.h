#pragma once
// Exceptions shared by ForecastGridBin and MappedForecastGrid readers of the forecast-grid cache file.

#include <stdexcept>

namespace rope::io {

// No forecast has been run yet (or the cache path is wrong).
struct ForecastCacheMissingError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Cache file exists but is corrupt: bad magic/version, implausible shape, or truncated.
struct ForecastCacheCorruptError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace rope::io
