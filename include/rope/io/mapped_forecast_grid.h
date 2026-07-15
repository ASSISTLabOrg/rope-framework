#pragma once
// MappedForecastGrid — memory-mapped read of the forecast-grid cache file
// (see forecast_grid_bin.h for the exact binary format).
//
// Unlike ForecastGridBin::load() (which fully materializes density/
// uncertainty into owned std::vectors), this maps the file and returns raw
// pointers into it — memory footprint is whatever the OS pages in, not the
// whole file. This is what `rope get` and `rope_open()` use, since a
// multi-year forecast's grid can be tens of GB.
//
// Exposes the same field/method shape as ForecastGrid (`shape`, `times`,
// `H`, `density_at(t)`, `uncertainty_at(t)`) so GridInterpolator<Grid> can
// be instantiated over either type identically. H/shape/times are tiny
// (H int64s, at most) and are copied into regular owned members; only the
// two large arrays stay mapped.

#include "rope/core/platform.h"
#include "rope/core/types.h"
#include "rope/io/forecast_cache_errors.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rope::io {

class MappedForecastGrid {
public:
    // Throws ForecastCacheMissingError if the file doesn't exist,
    // ForecastCacheCorruptError on bad magic/version/shape/truncation.
    static MappedForecastGrid open(const std::filesystem::path& path);

    GridSpec                   shape;
    std::vector<std::int64_t> times;
    int H = 0;

    const float* density_at(int t) const noexcept;
    const float* uncertainty_at(int t) const noexcept;

private:
    platform::MappedFile file_;
    std::size_t density_offset_     = 0;
    std::size_t uncertainty_offset_ = 0;
};

} // namespace rope::io
