#pragma once
// forecast/pipeline.h — public interface for the ROPE forecast pipeline.
//
// Usage:
//   auto cfg = rope::forecast::Config{...};
//   auto pipe = rope::forecast::load(cfg);
//   ForecastGrid grid = pipe->run("2024-02-09 00:00:00", 120);

#include "rope/core/types.h"
#include "rope/io/config_reader.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rope::forecast {

// ---------------------------------------------------------------------------
// Config — all paths and tuning parameters needed to construct a Pipeline.
// ---------------------------------------------------------------------------
struct Config {
    // Directory produced by export_models.py.
    std::filesystem::path exported_dir;

    // Explicit driver file (.swbin/.csv). Empty: use driver_config.json + DriverCacheManager.
    std::filesystem::path driver_path;

    // Cached .swbin directory. Empty: platform cache root.
    std::filesystem::path cache_dir;

    // Max cached driver file age before refresh.
    int cache_max_age_hours = 24;

    // ORT intra-op threads for the 15 base models.
    int intra_threads_base    = 1;

    // ORT intra-op threads for the meta model. 0 = hardware_concurrency().
    int intra_threads_meta    = 0;

    // ORT intra-op threads for the COAE decoder. 0 = hardware_concurrency().
    int intra_threads_decoder = 0;

    // LibTorch device string for the COAE decoder.
    std::string decoder_device = "cpu";

    // When false, skip the Unscented Transform; uncertainty is 0.
    bool compute_uncertainty = true;

    // Max forecast-hours of decoded voxel data held at once. <= 0: one chunk (whole horizon).
    int decode_chunk_hours = 72;

    // Load-progress callback. Default (nullptr): silent.
    std::function<void(std::string_view)> log;
};

// ---------------------------------------------------------------------------
// GridChunkSink — one contiguous slice of the grid, increasing t_offset order.
// LatentSink — the full (H, latent_dim) latent trajectory, called once.
// ---------------------------------------------------------------------------
using GridChunkSink = std::function<void(int t_offset,
                                          std::span<const std::int64_t> times,
                                          std::span<const float> density,
                                          std::span<const float> uncertainty)>;
using LatentSink = std::function<void(std::span<const float> latent_mean)>;

// ---------------------------------------------------------------------------
// Pipeline — abstract interface; constructed by load().
// ---------------------------------------------------------------------------
class Pipeline {
public:
    virtual ~Pipeline() = default;

    // Collects run_streaming() into one in-memory ForecastGrid.
    // Output spans hour 0 (start_iso) through hour `horizon`, inclusive —
    // H_lat = horizon + 1 timesteps total.
    // latent_mean_out, when non-null, is filled with the (H_lat, latent_dim) latent trajectory.
    ForecastGrid run(const std::string& start_iso, int horizon,
                     std::vector<float>* latent_mean_out = nullptr)
    {
        const int H_lat = horizon + 1;
        ForecastGrid grid;
        grid.shape = grid_shape();
        grid.H = H_lat;
        const std::size_t voxels = static_cast<std::size_t>(grid.shape.voxels());
        grid.times.resize(static_cast<std::size_t>(H_lat));
        grid.density.resize(static_cast<std::size_t>(H_lat) * voxels);
        grid.uncertainty.resize(static_cast<std::size_t>(H_lat) * voxels);

        run_streaming(start_iso, horizon, /*chunk_hours=*/H_lat,
            [&](int t_offset, std::span<const std::int64_t> times,
                std::span<const float> density, std::span<const float> uncertainty) {
                std::copy(times.begin(), times.end(), grid.times.begin() + t_offset);
                std::copy(density.begin(), density.end(),
                          grid.density.begin() + static_cast<std::size_t>(t_offset) * voxels);
                std::copy(uncertainty.begin(), uncertainty.end(),
                          grid.uncertainty.begin() + static_cast<std::size_t>(t_offset) * voxels);
            },
            latent_mean_out
                ? LatentSink{[latent_mean_out](std::span<const float> lm) {
                      latent_mean_out->assign(lm.begin(), lm.end());
                  }}
                : LatentSink{});
        return grid;
    }

    virtual void run_streaming(const std::string& start_iso, int horizon,
                                int chunk_hours, const GridChunkSink& sink,
                                const LatentSink& latent_sink = nullptr) = 0;

    virtual GridSpec grid_shape() const = 0;
    virtual std::string model_kind() const = 0;
    virtual int latent_dim() const = 0;
};

// ---------------------------------------------------------------------------
// load() — construct and initialize the pipeline from cfg. Throws on error.
// ---------------------------------------------------------------------------
std::unique_ptr<Pipeline> load(const Config& cfg);

// ---------------------------------------------------------------------------
// config_from_reader() — build a Config from a parsed rope.conf.
// Relative paths resolve against config_dir.
// ---------------------------------------------------------------------------
Config config_from_reader(const io::ConfigReader& config,
                          const std::filesystem::path& config_dir);

} // namespace rope::forecast
