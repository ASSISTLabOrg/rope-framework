#pragma once
// forecast/pipeline.h — public interface for the ROPE forecast pipeline.
//
// Usage:
//   auto cfg = rope::forecast::Config{...};
//   auto pipe = rope::forecast::load(cfg);
//   ForecastGrid grid = pipe->run("2024-02-09 00:00:00", 120);

#include "rope/core/types.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace rope::forecast {

// ---------------------------------------------------------------------------
// Config — all paths and tuning parameters needed to construct a Pipeline.
// ---------------------------------------------------------------------------
struct Config {
    // Directory produced by export_models.py (contains *.onnx, *.bin,
    // driver_config.json, ic_config.json, ic_table.icbin or ic_table.csv).
    std::filesystem::path exported_dir;

    // Explicit driver data file (.swbin or .csv, format auto-detected).
    // When set, bypasses the cache manager entirely — no network access.
    // Leave empty to let driver_config.json + DriverCacheManager supply data.
    std::filesystem::path driver_path;

    // Directory for cached .swbin files written by DriverCacheManager.
    // Defaults to the platform cache root when empty.
    std::filesystem::path cache_dir;

    // How old a cached driver file may be before refresh is attempted.
    int cache_max_age_hours = 24;

    // ORT intra-op threads for the 15 base models.
    int intra_threads_base    = 1;

    // ORT intra-op threads for the meta model.
    // 0 = std::thread::hardware_concurrency().
    int intra_threads_meta    = 0;

    // ORT intra-op threads for the COAE decoder.
    // 0 = std::thread::hardware_concurrency().
    int intra_threads_decoder = 0;

    // PyTorch device string for the COAE decoder when LibTorch is linked.
    std::string decoder_device = "cpu";

    // When false, skip the Unscented Transform; uncertainty is set to 0.
    bool compute_uncertainty = true;

    // Progress callback invoked during pipeline load (model names, timing, etc.).
    // Called on the constructor thread. Default (nullptr): silent.
    std::function<void(std::string_view)> log;
};

// ---------------------------------------------------------------------------
// Pipeline — abstract interface; constructed by load().
// ---------------------------------------------------------------------------
class Pipeline {
public:
    virtual ~Pipeline() = default;

    // Run the full ROPE forecast pipeline.
    //   start_iso : "YYYY-MM-DD HH:MM:SS" UTC (rounded to the hour internally)
    //   horizon   : forecast length in hours (H)
    // Returns a ForecastGrid with H time steps, density and uncertainty fields.
    virtual ForecastGrid run(const std::string& start_iso, int horizon) = 0;
};

// ---------------------------------------------------------------------------
// load() — construct and initialize the pipeline from cfg.
// Loads all models and data tables from disk.  Throws on any error.
// ---------------------------------------------------------------------------
std::unique_ptr<Pipeline> load(const Config& cfg);

} // namespace rope::forecast
