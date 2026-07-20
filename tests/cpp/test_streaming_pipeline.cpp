#include <catch2/catch_test_macros.hpp>
#include "rope/forecast/pipeline.h"
#include "rope/core/types.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using rope::forecast::GridChunkSink;
using rope::forecast::LatentSink;

namespace {

rope::forecast::Config make_config(bool compute_uncertainty) {
    rope::forecast::Config cfg;
    cfg.exported_dir          = fs::path(ROPE_FIXTURE_MODELS);
    cfg.driver_path           = fs::path(ROPE_FIXTURE_DIR) / "sw_test_long.csv";
    cfg.intra_threads_base    = 1;
    cfg.intra_threads_meta    = 1;
    cfg.intra_threads_decoder = 1;
    cfg.compute_uncertainty   = compute_uncertainty;
    return cfg;
}

static const char* TEST_START = "2024-01-01 00:00:00";
constexpr int TEST_HORIZON = 20;

// Collects run_streaming()'s chunks into one flat ForecastGrid.
// horizon is hours beyond the given IC; output spans hour 0..horizon
// inclusive, i.e. horizon+1 timesteps.
rope::ForecastGrid collect(rope::forecast::Pipeline& pipe, const std::string& start,
                          int horizon, int chunk_hours,
                          std::vector<float>* latent_mean_out = nullptr,
                          std::vector<std::pair<int,int>>* chunk_log = nullptr)
{
    const int H_lat = horizon + 1;
    rope::ForecastGrid grid;
    grid.shape = pipe.grid_shape();
    grid.H = H_lat;
    const std::size_t voxels = static_cast<std::size_t>(grid.shape.voxels());
    grid.times.resize(static_cast<std::size_t>(H_lat));
    grid.density.resize(static_cast<std::size_t>(H_lat) * voxels);
    grid.uncertainty.resize(static_cast<std::size_t>(H_lat) * voxels);

    pipe.run_streaming(start, horizon, chunk_hours,
        [&](int t_offset, std::span<const std::int64_t> times,
            std::span<const float> density, std::span<const float> uncertainty) {
            if (chunk_log) chunk_log->emplace_back(t_offset, static_cast<int>(times.size()));
            std::copy(times.begin(), times.end(), grid.times.begin() + t_offset);
            std::copy(density.begin(), density.end(), grid.density.begin() + t_offset * voxels);
            std::copy(uncertainty.begin(), uncertainty.end(), grid.uncertainty.begin() + t_offset * voxels);
        },
        latent_mean_out
            ? LatentSink{[latent_mean_out](std::span<const float> lm) {
                  latent_mean_out->assign(lm.begin(), lm.end());
              }}
            : LatentSink{});
    return grid;
}

} // namespace

TEST_CASE("Streaming: run() and run_streaming() at several chunk_hours values are byte-identical (golden equality)") {
    for (bool uncertainty_on : {true, false}) {
        auto pipe = rope::forecast::load(make_config(uncertainty_on));
        auto reference = pipe->run(TEST_START, TEST_HORIZON);

        for (int chunk_hours : {1, 3, 21, 72, 0}) {
            INFO("uncertainty_on=" << uncertainty_on << " chunk_hours=" << chunk_hours);
            auto grid = collect(*pipe, TEST_START, TEST_HORIZON, chunk_hours);
            CHECK(grid.H == reference.H);
            CHECK(grid.times == reference.times);
            CHECK(grid.density == reference.density);
            CHECK(grid.uncertainty == reference.uncertainty);
        }
    }
}

TEST_CASE("Streaming: off-by-one chunk boundaries produce correct, contiguous t_offset/count") {
    auto pipe = rope::forecast::load(make_config(false));
    std::vector<std::pair<int,int>> chunks;
    collect(*pipe, TEST_START, 10, 3, nullptr, &chunks);

    int total = 0;
    int next_expected = 0;
    for (auto [t_offset, count] : chunks) {
        CHECK(t_offset == next_expected);
        next_expected += count;
        total += count;
    }
    CHECK(total == 11);
}

TEST_CASE("Streaming: chunk_hours=1 covers every hour exactly once") {
    auto pipe = rope::forecast::load(make_config(false));
    std::vector<std::pair<int,int>> chunks;
    auto grid = collect(*pipe, TEST_START, 5, 1, nullptr, &chunks);

    CHECK(chunks.size() == 6);
    int next_expected = 0;
    for (auto [t_offset, count] : chunks) {
        CHECK(t_offset == next_expected);
        CHECK(count == 1);
        next_expected += count;
    }
    CHECK(grid.H == 6);
    for (float d : grid.density) CHECK(d > 0.0f);
}

TEST_CASE("Streaming: latent_sink delivers the (H_lat, K) fused latent trajectory exactly once, before any chunk") {
    auto pipe = rope::forecast::load(make_config(true));
    std::vector<float> latent_mean;
    int latent_calls = 0;
    bool latent_seen_before_first_chunk = false;
    bool any_chunk_seen = false;

    pipe->run_streaming(TEST_START, TEST_HORIZON, 3,
        [&](int, std::span<const std::int64_t>, std::span<const float>, std::span<const float>) {
            any_chunk_seen = true;
        },
        [&](std::span<const float> lm) {
            ++latent_calls;
            latent_seen_before_first_chunk = !any_chunk_seen;
            latent_mean.assign(lm.begin(), lm.end());
        });

    CHECK(latent_calls == 1);
    CHECK(latent_seen_before_first_chunk);
    CHECK(static_cast<int>(latent_mean.size()) == (TEST_HORIZON + 1) * pipe->latent_dim());
}

TEST_CASE("Streaming: latent_mean via run_streaming() matches Pipeline::run()'s latent_mean_out") {
    auto pipe = rope::forecast::load(make_config(true));

    std::vector<float> via_run;
    pipe->run(TEST_START, TEST_HORIZON, &via_run);

    std::vector<float> via_streaming;
    collect(*pipe, TEST_START, TEST_HORIZON, 4, &via_streaming);

    CHECK(via_run == via_streaming);
}
