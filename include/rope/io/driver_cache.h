#pragma once
#include <chrono>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace rope::io {

struct DriverSource {
    std::string url;
    std::string description;
};

// Known online sources, keyed by driver_config.json's "source" field. Edit driver_cache.cpp to add entries.
const std::unordered_map<std::string, DriverSource>& known_sources();

// Local cache of driver .swbin files refreshed from online sources; wired into Pipeline::load() for auto-refresh.
class DriverCacheManager {
public:
    DriverCacheManager(std::filesystem::path cache_dir,
                       int max_age_hours = 24);

    // Fresh .swbin path for `source`; refreshes if stale, falls back to a stale file on refresh failure.
    std::filesystem::path get_path(const std::string& source);

private:
    bool is_stale(const std::filesystem::path& path) const;
    void refresh(const std::string& source, const std::filesystem::path& dest);

    // TODO: implement via cpp-httplib + OpenSSL.
    std::string download(const std::string& url);

    // CelesTrak CSV (DATE, F10.7_OBS, F10.7_OBS_CENTER81, AP_AVG, AP1..AP8) -> hourly (datetime, f10, kp).
    // TODO: implement Ap->Kp conversion and 3-hourly->hourly interpolation.
    void convert_and_write(const std::string& raw_csv,
                           const std::filesystem::path& dest);

    std::filesystem::path cache_dir_;
    std::chrono::seconds  max_age_;
};

} // namespace rope::io
