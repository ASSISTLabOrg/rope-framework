#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

#include "rope/core/types.h"
#include "../../src/forecast/grid_stitch.h"

using namespace rope;
using namespace rope::forecast;

TEST_CASE("stitch_altitude_range: full range is identity copy") {
    const int H = 2;
    const std::size_t N = static_cast<std::size_t>(H) * GRID_VOXELS;
    std::vector<float> src(N);
    std::iota(src.begin(), src.end(), 0.0f);

    std::vector<float> dst(N, -1.0f);
    stitch_altitude_range(dst.data(), src.data(), H, 0, GRID_ALT);

    CHECK(dst == src);
}

TEST_CASE("stitch_altitude_range: two partial stages land in correct altitude slots") {
    const int H   = 1;
    const int MID = 20;

    std::vector<float> src1(static_cast<std::size_t>(H) * GRID_LST * GRID_LAT * MID,             1.0f);
    std::vector<float> src2(static_cast<std::size_t>(H) * GRID_LST * GRID_LAT * (GRID_ALT - MID), 2.0f);

    std::vector<float> dst(static_cast<std::size_t>(H) * GRID_VOXELS, 0.0f);
    stitch_altitude_range(dst.data(), src1.data(), H, 0,   MID);
    stitch_altitude_range(dst.data(), src2.data(), H, MID, GRID_ALT);

    bool ok = true;
    for (int lst = 0; lst < GRID_LST && ok; ++lst)
        for (int lat = 0; lat < GRID_LAT && ok; ++lat)
            for (int alt = 0; alt < GRID_ALT && ok; ++alt) {
                float expected = (alt < MID) ? 1.0f : 2.0f;
                float actual   = dst[(lst * GRID_LAT + lat) * GRID_ALT + alt];
                if (actual != expected) ok = false;
            }
    CHECK(ok);
}

TEST_CASE("stitch_altitude_range: H dimension is honored") {
    const int H   = 3;
    const int MID = 15;

    const std::size_t frame1 = static_cast<std::size_t>(GRID_LST) * GRID_LAT * MID;
    const std::size_t frame2 = static_cast<std::size_t>(GRID_LST) * GRID_LAT * (GRID_ALT - MID);

    std::vector<float> src1(static_cast<std::size_t>(H) * frame1);
    std::vector<float> src2(static_cast<std::size_t>(H) * frame2);
    for (int t = 0; t < H; ++t) {
        std::fill(src1.begin() + t * frame1, src1.begin() + (t + 1) * frame1,
                  static_cast<float>(t + 1));
        std::fill(src2.begin() + t * frame2, src2.begin() + (t + 1) * frame2,
                  static_cast<float>(-(t + 1)));
    }

    std::vector<float> dst(static_cast<std::size_t>(H) * GRID_VOXELS, 0.0f);
    stitch_altitude_range(dst.data(), src1.data(), H, 0,   MID);
    stitch_altitude_range(dst.data(), src2.data(), H, MID, GRID_ALT);

    for (int t = 0; t < H; ++t) {
        const float* frame = dst.data() + static_cast<std::size_t>(t) * GRID_VOXELS;
        for (int lst = 0; lst < GRID_LST; ++lst)
            for (int lat = 0; lat < GRID_LAT; ++lat) {
                for (int alt = 0; alt < MID; ++alt)
                    CHECK(frame[(lst * GRID_LAT + lat) * GRID_ALT + alt] == static_cast<float>(t + 1));
                for (int alt = MID; alt < GRID_ALT; ++alt)
                    CHECK(frame[(lst * GRID_LAT + lat) * GRID_ALT + alt] == static_cast<float>(-(t + 1)));
            }
    }
}

TEST_CASE("stitch_altitude_range: non-overlapping stages leave zero in untouched slots") {
    const int H   = 1;
    const int LO  = 10;
    const int HI  = 30;

    std::vector<float> src(static_cast<std::size_t>(H) * GRID_LST * GRID_LAT * (HI - LO), 7.0f);
    std::vector<float> dst(static_cast<std::size_t>(H) * GRID_VOXELS, 0.0f);
    stitch_altitude_range(dst.data(), src.data(), H, LO, HI);

    for (int lst = 0; lst < GRID_LST; ++lst)
        for (int lat = 0; lat < GRID_LAT; ++lat)
            for (int alt = 0; alt < GRID_ALT; ++alt) {
                float v = dst[(lst * GRID_LAT + lat) * GRID_ALT + alt];
                if (alt >= LO && alt < HI)
                    CHECK(v == 7.0f);
                else
                    CHECK(v == 0.0f);
            }
}
