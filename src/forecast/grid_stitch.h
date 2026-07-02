#pragma once
#include <algorithm>
#include "rope/core/types.h"

namespace rope::forecast {

// src: (H, LST, LAT, n_alt); dst: (H, LST, LAT, GRID_ALT), caller zero-init.
inline void stitch_altitude_range(float* dst, const float* src, int H,
                                   int alt_start, int alt_end)
{
    const int n_alt = alt_end - alt_start;
    for (int t = 0; t < H; ++t)
        for (int lst = 0; lst < GRID_LST; ++lst)
            for (int lat = 0; lat < GRID_LAT; ++lat) {
                const float* s = src
                    + (static_cast<std::size_t>(t) * GRID_LST + lst) * GRID_LAT * n_alt
                    + static_cast<std::size_t>(lat) * n_alt;
                float* d = dst
                    + static_cast<std::size_t>(t) * GRID_VOXELS
                    + (static_cast<std::size_t>(lst) * GRID_LAT + lat) * GRID_ALT
                    + alt_start;
                std::copy(s, s + n_alt, d);
            }
}

} // namespace rope::forecast
