#include "rope/forecast/pipeline.h"

namespace rope::forecast {

Config config_from_reader(const io::ConfigReader& config,
                          const std::filesystem::path& config_dir)
{
    // Paths in the config file may be relative; resolve them against the
    // directory that contains the config file.
    auto resolve = [&](const std::string& p) -> std::filesystem::path {
        std::filesystem::path fp{p};
        return fp.is_absolute() ? fp : config_dir / fp;
    };

    Config fcfg;
    fcfg.exported_dir        = resolve(config.get("paths.exported_dir"));
    // driver_path is optional: when absent, the cache manager takes over.
    if (config.has("paths.driver_path"))
        fcfg.driver_path = resolve(config.get("paths.driver_path"));
    // cache_dir is optional: defaults to platform cache root.
    if (config.has("driver_cache.cache_dir"))
        fcfg.cache_dir = resolve(config.get("driver_cache.cache_dir"));
    fcfg.cache_max_age_hours   = config.get_int("driver_cache.max_age_hours", 24);
    fcfg.intra_threads_base    = config.get_int("threads.intra_threads_base", 1);
    fcfg.intra_threads_meta    = config.get_int("threads.intra_threads_meta", 0);
    fcfg.intra_threads_decoder = config.get_int("threads.intra_threads_decoder", 0);
    fcfg.decoder_device        = config.get("decoder.device", "cpu");
    fcfg.compute_uncertainty   = config.get("forecast.compute_uncertainty", "true") == "true";
    fcfg.decode_chunk_hours    = config.get_int("forecast.decode_chunk_hours", 72);

    return fcfg;
}

} // namespace rope::forecast
