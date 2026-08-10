#include <catch2/catch_test_macros.hpp>
#include "rope/io/forecast_grid_bin.h"
#include "rope/io/mapped_forecast_grid.h"
#include "rope/interpolate/grid_interpolator.h"

#include <filesystem>
#include <fstream>

#if defined(__unix__) || defined(__APPLE__)
#define ROPE_TEST_HAS_POSIX_RENAME_SEMANTICS 1
#endif

namespace fs = std::filesystem;
using rope::ForecastGrid;
using rope::GridSpec;
using rope::io::ForecastGridBin;
using rope::io::MappedForecastGrid;
using rope::io::ForecastCacheMissingError;
using rope::io::ForecastCacheCorruptError;
using rope::interpolate::GridInterpolator;

static GridSpec small_shape() {
    GridSpec s;
    s.n_lst = 4; s.n_lat = 3; s.n_alt = 5;
    s.lat_min_deg = -60.0; s.lat_max_deg = 60.0;
    s.alt_min_km  = 150.0; s.alt_max_km  = 500.0;
    return s;
}

static ForecastGrid make_grid(int H, float den, float unc) {
    ForecastGrid g;
    g.shape = small_shape();
    g.H = H;
    g.density.assign(static_cast<std::size_t>(H) * g.shape.voxels(), den);
    g.uncertainty.assign(static_cast<std::size_t>(H) * g.shape.voxels(), unc);
    g.times.resize(static_cast<std::size_t>(H));
    for (int t = 0; t < H; ++t) g.times[t] = 1700000000 + t * 3600;
    return g;
}

TEST_CASE("MappedForecastGrid: accessors match the source grid exactly") {
    auto path = fs::temp_directory_path() / "rope_mapped_fg_accessors.bin";
    fs::remove(path);

    auto grid = make_grid(2, 3.5f, 0.75f);
    ForecastGridBin::save(grid, path);

    auto mapped = MappedForecastGrid::open(path);
    CHECK(mapped.H == grid.H);
    CHECK(mapped.shape.n_lst == grid.shape.n_lst);
    CHECK(mapped.shape.n_lat == grid.shape.n_lat);
    CHECK(mapped.shape.n_alt == grid.shape.n_alt);
    CHECK(mapped.times == grid.times);

    const int voxels = grid.shape.voxels();
    for (int t = 0; t < grid.H; ++t) {
        const float* d = mapped.density_at(t);
        const float* u = mapped.uncertainty_at(t);
        for (int v = 0; v < voxels; ++v) {
            CHECK(d[v] == grid.density_at(t)[v]);
            CHECK(u[v] == grid.uncertainty_at(t)[v]);
        }
    }

    fs::remove(path);
}

TEST_CASE("MappedForecastGrid: interpolates identically to the vector-backed grid") {
    auto path = fs::temp_directory_path() / "rope_mapped_fg_interp.bin";
    fs::remove(path);

    auto grid = make_grid(2, 2.0f, 0.5f);
    ForecastGridBin::save(grid, path);
    auto mapped = MappedForecastGrid::open(path);

    // small_shape() has n_alt=5, below the default n_etp_pts=8 -- pass an explicit value that fits.
    rope::interpolate::ExtrapolationOptions small_grid_opts{true, 2};
    GridInterpolator<ForecastGrid>        gi_vec(grid, small_grid_opts);
    GridInterpolator<MappedForecastGrid>  gi_map(mapped, small_grid_opts);

    auto r_vec = gi_vec.query_interp(grid.times[0], 6.0, 0.0, 300.0);
    auto r_map = gi_map.query_interp(grid.times[0], 6.0, 0.0, 300.0);
    CHECK(r_vec.density == r_map.density);
    CHECK(r_vec.uncertainty == r_map.uncertainty);

    fs::remove(path);
}

TEST_CASE("MappedForecastGrid: open on a nonexistent path throws ForecastCacheMissingError") {
    auto path = fs::temp_directory_path() / "rope_mapped_fg_missing.bin";
    fs::remove(path);
    REQUIRE_THROWS_AS(MappedForecastGrid::open(path), ForecastCacheMissingError);
}

TEST_CASE("MappedForecastGrid: bad magic throws ForecastCacheCorruptError") {
    auto path = fs::temp_directory_path() / "rope_mapped_fg_badmagic.bin";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        std::uint32_t bad_magic = 0xDEADBEEFu;
        char padding[56] = {};
        f.write(reinterpret_cast<const char*>(&bad_magic), 4);
        f.write(padding, sizeof(padding));
    }
    REQUIRE_THROWS_AS(MappedForecastGrid::open(path), ForecastCacheCorruptError);
    fs::remove(path);
}

TEST_CASE("MappedForecastGrid: two concurrent read-only mappings of the same file are safe") {
    auto path = fs::temp_directory_path() / "rope_mapped_fg_concurrent.bin";
    fs::remove(path);
    auto grid = make_grid(1, 4.0f, 0.4f);
    ForecastGridBin::save(grid, path);

    auto m1 = MappedForecastGrid::open(path);
    auto m2 = MappedForecastGrid::open(path);
    CHECK(m1.density_at(0)[0] == m2.density_at(0)[0]);

    fs::remove(path);
}

#ifdef ROPE_TEST_HAS_POSIX_RENAME_SEMANTICS
TEST_CASE("MappedForecastGrid: an open mapping keeps serving old data after the file is atomically replaced") {
    auto path = fs::temp_directory_path() / "rope_mapped_fg_snapshot.bin";
    fs::remove(path);

    auto gridA = make_grid(1, 1.0f, 0.1f);
    ForecastGridBin::save(gridA, path);
    auto mapped = MappedForecastGrid::open(path);  // still holds gridA's data

    auto gridB = make_grid(1, 99.0f, 9.9f);
    ForecastGridBin::save(gridB, path);  // atomically replaces path

    // The already-open mapping must still see gridA's values, not gridB's —
    // POSIX rename never invalidates an existing mapping/fd.
    CHECK(mapped.density_at(0)[0] == 1.0f);

    auto reopened = MappedForecastGrid::open(path);
    CHECK(reopened.density_at(0)[0] == 99.0f);

    fs::remove(path);
}
#endif
