#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

#include "rope/core/types.h"
#include "../../src/forecast/backends/grid_stitch.h"

using namespace rope;
using namespace rope::forecast;

// Matches the shape the hardcoded constants used to assume: 72x36x45.
static GridSpec test_shape() {
    GridSpec s;
    s.n_lst = 72; s.n_lat = 36; s.n_alt = 45;
    s.lat_min_deg = -87.5; s.lat_max_deg = 87.5;
    s.alt_min_km  = 100.0; s.alt_max_km  = 980.0;
    return s;
}
static const GridSpec GRID = test_shape();
static const int GRID_LST    = GRID.n_lst;
static const int GRID_LAT    = GRID.n_lat;
static const int GRID_ALT    = GRID.n_alt;
static const int GRID_VOXELS = GRID.voxels();

TEST_CASE("stitch_altitude_range: full range is identity copy") {
    const int H = 2;
    const std::size_t N = static_cast<std::size_t>(H) * GRID_VOXELS;
    std::vector<float> src(N);
    std::iota(src.begin(), src.end(), 0.0f);

    std::vector<float> dst(N, -1.0f);
    stitch_altitude_range(dst.data(), src.data(), H, 0, GRID_ALT, GRID);

    CHECK(dst == src);
}

TEST_CASE("stitch_altitude_range: two partial stages land in correct altitude slots") {
    const int H   = 1;
    const int MID = 20;

    std::vector<float> src1(static_cast<std::size_t>(H) * GRID_LST * GRID_LAT * MID,             1.0f);
    std::vector<float> src2(static_cast<std::size_t>(H) * GRID_LST * GRID_LAT * (GRID_ALT - MID), 2.0f);

    std::vector<float> dst(static_cast<std::size_t>(H) * GRID_VOXELS, 0.0f);
    stitch_altitude_range(dst.data(), src1.data(), H, 0,   MID,      GRID);
    stitch_altitude_range(dst.data(), src2.data(), H, MID, GRID_ALT, GRID);

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
    stitch_altitude_range(dst.data(), src1.data(), H, 0,   MID,      GRID);
    stitch_altitude_range(dst.data(), src2.data(), H, MID, GRID_ALT, GRID);

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
    stitch_altitude_range(dst.data(), src.data(), H, LO, HI, GRID);

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

TEST_CASE("stitch_altitude_range: honors a non-default grid shape") {
    // A smaller, differently-ranged grid than the 72x36x45 default, proving
    // stitch_altitude_range doesn't assume any particular fixed shape.
    GridSpec small;
    small.n_lst = 4; small.n_lat = 3; small.n_alt = 5;
    small.lat_min_deg = -60.0; small.lat_max_deg = 60.0;
    small.alt_min_km  = 150.0; small.alt_max_km  = 400.0;

    const int H = 1;
    std::vector<float> src(static_cast<std::size_t>(H) * small.voxels(), 9.0f);
    std::vector<float> dst(static_cast<std::size_t>(H) * small.voxels(), 0.0f);
    stitch_altitude_range(dst.data(), src.data(), H, 0, small.n_alt, small);

    CHECK(dst == src);
}
