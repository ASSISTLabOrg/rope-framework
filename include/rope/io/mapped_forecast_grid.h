#pragma once
// Memory-mapped read of the forecast-grid cache file (see forecast_grid_bin.h for the format).
// Unlike ForecastGridBin::load(), doesn't materialize density/uncertainty — only OS-paged bytes count against memory.
// Used by `rope get`/`rope_open()`. Mirrors ForecastGrid's field/method shape so GridInterpolator<Grid> works over either.

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
