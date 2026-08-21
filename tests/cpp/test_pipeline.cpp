#include <catch2/catch_test_macros.hpp>
#include "rope/forecast/pipeline.h"
#include "rope/core/types.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

// Copies ROPE_FIXTURE_MODELS into a temp dir and patches top-level uncert_scale_factor into its manifest.
static fs::path fixture_models_with_uncert_scale(double factor, const char* dir_name) {
    fs::path src_dir = fs::path(ROPE_FIXTURE_MODELS);
    fs::path dst_dir = fs::temp_directory_path() / dir_name;
    fs::remove_all(dst_dir);
    fs::create_directories(dst_dir);
    fs::copy(src_dir, dst_dir, fs::copy_options::recursive | fs::copy_options::overwrite_existing);

    fs::path manifest_path = dst_dir / "model_manifest.json";
    std::ifstream in(manifest_path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string manifest = ss.str();

    const std::string needle = "\"validated\": false,";
    auto pos = manifest.find(needle);
    if (pos == std::string::npos)
        throw std::runtime_error("fixture_models_with_uncert_scale: validated field not found in manifest");
    manifest.replace(pos, needle.size(),
                     needle + " \"uncert_scale_factor\": " + std::to_string(factor) + ",");

    std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
    out << manifest;
    return dst_dir;
}

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
// run()'s output spans hour 0 through hour TEST_HORIZON, inclusive.
static constexpr int TEST_H_LAT = TEST_HORIZON + 1;

TEST_CASE("Pipeline: loads successfully from synthetic fixtures") {
    auto pipe = rope::forecast::load(make_test_config());
    REQUIRE(pipe != nullptr);
}

TEST_CASE("Pipeline: run() returns correctly shaped ForecastGrid") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);

    CHECK(grid.H == TEST_H_LAT);
    CHECK(static_cast<int>(grid.density.size())     == TEST_H_LAT * grid.shape.voxels());
    CHECK(static_cast<int>(grid.uncertainty.size()) == TEST_H_LAT * grid.shape.voxels());
    CHECK(static_cast<int>(grid.times.size())       == TEST_H_LAT);
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
    CHECK(grid.H == TEST_H_LAT);
    CHECK(static_cast<int>(grid.density.size())     == TEST_H_LAT * grid.shape.voxels());
    CHECK(static_cast<int>(grid.uncertainty.size()) == TEST_H_LAT * grid.shape.voxels());
    CHECK(static_cast<int>(grid.times.size())       == TEST_H_LAT);
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

    auto grid = pipe->run(TEST_START, 1);  // horizon=1; hours 0..1, two frames
    REQUIRE(grid.H == 2);
    REQUIRE(static_cast<int>(grid.density.size()) == 2 * grid.shape.voxels());

    const int n_lst = grid.shape.n_lst, n_lat = grid.shape.n_lat, n_alt = grid.shape.n_alt;
    const int voxels = grid.shape.voxels();
    bool lo_ok = true, hi_ok = true;
    for (int t = 0; t < grid.H && (lo_ok || hi_ok); ++t)
        for (int lst = 0; lst < n_lst && (lo_ok || hi_ok); ++lst)
            for (int lat = 0; lat < n_lat && (lo_ok || hi_ok); ++lat) {
                const float* col = grid.density.data()
                                   + t * voxels + (lst * n_lat + lat) * n_alt;
                for (int alt = 0;     alt < SPLIT;  ++alt)
                    if (col[alt] != 1.0f)  { lo_ok = false; break; }
                for (int alt = SPLIT; alt < n_alt;   ++alt)
                    if (col[alt] != 10.0f) { hi_ok = false; break; }
            }
    CHECK(lo_ok);
    CHECK(hi_ok);
}

TEST_CASE("Pipeline: uncert_scale_factor scales uncertainty but not density") {
    constexpr float FACTOR = 2.5f;

    auto baseline_pipe = rope::forecast::load(make_test_config());
    auto baseline       = baseline_pipe->run(TEST_START, TEST_HORIZON);

    auto scaled_dir = fixture_models_with_uncert_scale(FACTOR, "rope_ptest_uncert_scale");
    auto cfg         = make_test_config();
    cfg.exported_dir = scaled_dir;
    auto scaled_pipe = rope::forecast::load(cfg);
    auto scaled      = scaled_pipe->run(TEST_START, TEST_HORIZON);

    REQUIRE(scaled.uncertainty.size() == baseline.uncertainty.size());
    CHECK(scaled.density == baseline.density);
    // Byte-identical models + thread counts make pre-scale uncertainty bit-reproducible (determinism invariant).
    for (std::size_t i = 0; i < baseline.uncertainty.size(); ++i)
        CHECK(scaled.uncertainty[i] == baseline.uncertainty[i] * FACTOR);
}

TEST_CASE("Pipeline: default uncert_scale_factor (1.0, explicit) matches the implicit default") {
    auto baseline_pipe = rope::forecast::load(make_test_config());
    auto baseline       = baseline_pipe->run(TEST_START, TEST_HORIZON);

    auto explicit_dir = fixture_models_with_uncert_scale(1.0, "rope_ptest_uncert_default_explicit");
    auto cfg           = make_test_config();
    cfg.exported_dir   = explicit_dir;
    auto explicit_pipe = rope::forecast::load(cfg);
    auto explicit_grid = explicit_pipe->run(TEST_START, TEST_HORIZON);

    CHECK(explicit_grid.density == baseline.density);
    CHECK(explicit_grid.uncertainty == baseline.uncertainty);
}
