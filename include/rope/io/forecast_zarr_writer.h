#pragma once
// Optional Zarr v2 export of forecast output ('rope forecast --zarr <path>').
// Write-only. `path` is a container directory, not the store itself: the
// store lands at <path>/forecast_<start>_H<horizon>/.

#include "rope/core/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace rope::io {

class ForecastZarrWriter {
public:
    // Throws if path exists and isn't a directory, or the store subdirectory already exists.
    static ForecastZarrWriter open(const GridSpec& shape, int H, int K,
                                   const std::string& model_kind,
                                   const std::string& forecast_start,
                                   const std::filesystem::path& path);

    ForecastZarrWriter(ForecastZarrWriter&&) noexcept;
    ForecastZarrWriter& operator=(ForecastZarrWriter&&) noexcept;
    ForecastZarrWriter(const ForecastZarrWriter&) = delete;
    ForecastZarrWriter& operator=(const ForecastZarrWriter&) = delete;
    ~ForecastZarrWriter();

    const std::filesystem::path& store_path() const noexcept;

    // t_offset must arrive in non-overlapping, increasing order.
    void write_chunk(int t_offset, std::span<const std::int64_t> times,
                     std::span<const float> density, std::span<const float> uncertainty);

    // Call once, with the full (H, K) latent trajectory.
    void write_latent(std::span<const float> mu_lat);

    // Call once, after all chunks succeeded.
    void close();

private:
    ForecastZarrWriter() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rope::io
