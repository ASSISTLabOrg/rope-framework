#pragma once
#include <algorithm>
#include "rope/core/types.h"

namespace rope::forecast {

// src: (H, LST, LAT, n_alt); dst: (H, LST, LAT, shape.n_alt), caller zero-init.
inline void stitch_altitude_range(float* dst, const float* src, int H,
                                   int alt_start, int alt_end,
                                   const GridSpec& shape)
{
    const int n_alt = alt_end - alt_start;
    for (int t = 0; t < H; ++t)
        for (int lst = 0; lst < shape.n_lst; ++lst)
            for (int lat = 0; lat < shape.n_lat; ++lat) {
                const float* s = src
                    + (static_cast<std::size_t>(t) * shape.n_lst + lst) * shape.n_lat * n_alt
                    + static_cast<std::size_t>(lat) * n_alt;
                float* d = dst
                    + static_cast<std::size_t>(t) * shape.voxels()
                    + (static_cast<std::size_t>(lst) * shape.n_lat + lat) * shape.n_alt
                    + alt_start;
                std::copy(s, s + n_alt, d);
            }
}

} // namespace rope::forecast
