#include <catch2/catch_test_macros.hpp>
#include "rope/forecast/pipeline.h"
#include "rope/core/types.h"
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static rope::forecast::Config make_test_config() {
    rope::forecast::Config cfg;
    cfg.exported_dir          = fs::path(ROPE_FIXTURE_MODELS);
    cfg.driver_path           = fs::path(ROPE_FIXTURE_DIR) / "sw_test.csv";
    cfg.intra_threads_base    = 1;
    cfg.intra_threads_meta    = 1;
    cfg.intra_threads_decoder = 1;
    cfg.compute_uncertainty   = true;
    return cfg;
}

static const char* TEST_START = "2024-01-01 00:00:00";
static constexpr int TEST_HORIZON = 3;

TEST_CASE("Pipeline: loads successfully from synthetic fixtures") {
    auto pipe = rope::forecast::load(make_test_config());
    REQUIRE(pipe != nullptr);
}

TEST_CASE("Pipeline: run() returns correctly shaped ForecastGrid") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);

    CHECK(grid.H == TEST_HORIZON);
    CHECK(static_cast<int>(grid.density.size())     == TEST_HORIZON * rope::GRID_VOXELS);
    CHECK(static_cast<int>(grid.uncertainty.size()) == TEST_HORIZON * rope::GRID_VOXELS);
    CHECK(static_cast<int>(grid.times.size())       == TEST_HORIZON);
}

TEST_CASE("Pipeline: density is positive everywhere") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (float d : grid.density)
        CHECK(d > 0.0f);
}

TEST_CASE("Pipeline: uncertainty is non-negative everywhere") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (float u : grid.uncertainty)
        CHECK(u >= 0.0f);
}

TEST_CASE("Pipeline: time steps are monotonically increasing") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (int i = 1; i < grid.H; ++i)
        CHECK(grid.times[i] > grid.times[i - 1]);
}

TEST_CASE("Pipeline: uncertainty disabled sets it to zero") {
    auto cfg = make_test_config();
    cfg.compute_uncertainty = false;
    auto pipe = rope::forecast::load(cfg);
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (float u : grid.uncertainty)
        CHECK(u == 0.0f);
    for (float d : grid.density)
        CHECK(d > 0.0f);
}

TEST_CASE("Pipeline: run() is deterministic - repeated calls produce identical output") {
    auto pipe  = rope::forecast::load(make_test_config());
    auto grid1 = pipe->run(TEST_START, TEST_HORIZON);
    auto grid2 = pipe->run(TEST_START, TEST_HORIZON);

    CHECK(grid1.H == grid2.H);
    CHECK(grid1.times == grid2.times);
    CHECK(grid1.density == grid2.density);
    CHECK(grid1.uncertainty == grid2.uncertainty);
}

TEST_CASE("Pipeline: M=3 small ensemble produces correctly shaped ForecastGrid") {
    rope::forecast::Config cfg;
    cfg.exported_dir          = fs::path(ROPE_FIXTURE_DIR) / "test_models_m3";
    cfg.driver_path           = fs::path(ROPE_FIXTURE_DIR) / "sw_test.csv";
    cfg.intra_threads_base    = 1;
    cfg.intra_threads_meta    = 1;
    cfg.intra_threads_decoder = 1;
    cfg.compute_uncertainty   = false;

    auto pipe = rope::forecast::load(cfg);
    REQUIRE(pipe != nullptr);

    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    CHECK(grid.H == TEST_HORIZON);
    CHECK(static_cast<int>(grid.density.size())     == TEST_HORIZON * rope::GRID_VOXELS);
    CHECK(static_cast<int>(grid.uncertainty.size()) == TEST_HORIZON * rope::GRID_VOXELS);
    CHECK(static_cast<int>(grid.times.size())       == TEST_HORIZON);
    for (float d : grid.density)
        CHECK(d > 0.0f);
}

TEST_CASE("Pipeline: split-decoder stages land in correct altitude ranges") {
    constexpr int SPLIT = 22;

    rope::forecast::Config cfg;
    cfg.exported_dir          = fs::path(ROPE_FIXTURE_DIR) / "test_models_split";
    cfg.driver_path           = fs::path(ROPE_FIXTURE_DIR) / "sw_test.csv";
    cfg.intra_threads_base    = 1;
    cfg.intra_threads_meta    = 1;
    cfg.intra_threads_decoder = 1;
    cfg.compute_uncertainty   = false;

    auto pipe = rope::forecast::load(cfg);
    REQUIRE(pipe != nullptr);

    auto grid = pipe->run(TEST_START, 1);  // H=1; one frame is enough
    REQUIRE(grid.H == 1);
    REQUIRE(static_cast<int>(grid.density.size()) == rope::GRID_VOXELS);

    bool lo_ok = true, hi_ok = true;
    for (int lst = 0; lst < rope::GRID_LST && (lo_ok || hi_ok); ++lst)
        for (int lat = 0; lat < rope::GRID_LAT && (lo_ok || hi_ok); ++lat) {
            const float* col = grid.density.data()
                               + (lst * rope::GRID_LAT + lat) * rope::GRID_ALT;
            for (int alt = 0;     alt < SPLIT;         ++alt)
                if (col[alt] != 1.0f)  { lo_ok = false; break; }
            for (int alt = SPLIT; alt < rope::GRID_ALT; ++alt)
                if (col[alt] != 10.0f) { hi_ok = false; break; }
        }
    CHECK(lo_ok);
    CHECK(hi_ok);
}
