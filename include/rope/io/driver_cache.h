#pragma once
#include "rope/net/http_client.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

namespace rope::io {

struct DriverSource {
    std::string url;
    std::string description;
};

// Known online sources, keyed by model_manifest.json's top-level drivers.source field. Edit driver_cache.cpp to add entries.
const std::unordered_map<std::string, DriverSource>& known_sources();

// CelesTrak CSV (DATE, F10.7_OBS, KP1..KP8, AP1..AP8) -> hourly .swbin (f10, kp, ap) via PCHIP interpolation.
// Free function (no DriverCacheManager instance needed) so both the live download path and `rope convert-sw`
// can convert a raw CelesTrak CSV — the latter from a local file, without ever touching the network.
void convert_celestrak_csv_to_swbin(const std::string& raw_csv, const std::filesystem::path& dest);

// Local cache of driver .swbin files refreshed from online sources; wired into Pipeline::load() for auto-refresh.
class DriverCacheManager {
public:
    // `http` defaults to the production HTTPS client; tests inject a fake to exercise refresh() without the network.
    explicit DriverCacheManager(std::filesystem::path cache_dir,
                                int max_age_hours = 24,
                                std::unique_ptr<net::IHttpClient> http = net::make_http_client());

    // Fresh .swbin path for `source`; refreshes if stale, falls back to a stale file on refresh failure.
    std::filesystem::path get_path(const std::string& source);

private:
    bool is_stale(const std::filesystem::path& path) const;
    void refresh(const std::string& source, const std::filesystem::path& dest);

    // Delegates to the injected IHttpClient; throws std::runtime_error on any transport or HTTP-level failure.
    std::string download(const std::string& url);

    std::filesystem::path cache_dir_;
    std::chrono::seconds  max_age_;
    std::unique_ptr<net::IHttpClient> http_;
};

} // namespace rope::io
