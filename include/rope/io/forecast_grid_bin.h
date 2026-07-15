#pragma once
// Binary forecast-grid cache format (.bin) — written once by `rope forecast`,
// read by `rope get` and the C API. Replaces the old socket-based server cache.
//
// Header (60 bytes, little-endian):
//   uint32  magic        = 0x52504647  ("RPFG")
//   uint32  version      = 1
//   int32   n_lst, n_lat, n_alt   (this forecast's GridSpec bin counts)
//   int32   H                     (forecast hours)
//   double  lat_min_deg, lat_max_deg
//   double  alt_min_km,  alt_max_km
//   uint32  reserved     = 0
//
// Body (flat, fixed-stride — no index; any timestep's offset is directly
// computable, which is what makes MappedForecastGrid's mmap-based reads
// trivial):
//   int64   times[H]
//   float32 density[H * n_lst*n_lat*n_alt]
//   float32 uncertainty[H * n_lst*n_lat*n_alt]
//
// The header carries the full GridSpec (counts *and* physical ranges), not
// just counts, because a reader of this file never has access to the model
// manifest that produced it.

#include "rope/core/types.h"
#include "rope/io/forecast_cache_errors.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>

namespace rope::io {

class ForecastGridBin {
public:
    // Throws ForecastCacheMissingError if the file doesn't exist,
    // ForecastCacheCorruptError on bad magic/version/shape/truncation.
    static ForecastGrid load(const std::filesystem::path& path);

    // Atomically replaces `path` (write to a temp file in the same
    // directory, then rename) — a reader never observes a partial file.
    // Throws std::runtime_error if `grid` is malformed (H<=0, or array
    // sizes inconsistent with H * grid.shape.voxels()).
    static void save(const ForecastGrid& grid, const std::filesystem::path& path);
};

// Streaming counterpart to ForecastGridBin::save() -- produces a byte-identical file.
class ForecastGridBinWriter {
public:
    static ForecastGridBinWriter open(const GridSpec& shape, int H,
                                      const std::filesystem::path& path);

    ForecastGridBinWriter(ForecastGridBinWriter&&) = default;
    ForecastGridBinWriter& operator=(ForecastGridBinWriter&&) = default;
    ForecastGridBinWriter(const ForecastGridBinWriter&) = delete;
    ForecastGridBinWriter& operator=(const ForecastGridBinWriter&) = delete;

    // t_offset must arrive in non-overlapping, increasing order.
    void write_chunk(int t_offset, std::span<const std::int64_t> times,
                     std::span<const float> density, std::span<const float> uncertainty);

    // Call once, after every chunk succeeded.
    void close();

    // Discards the temp file if close() was never reached.
    ~ForecastGridBinWriter();

private:
    ForecastGridBinWriter() = default;

    std::ofstream file_;
    std::filesystem::path tmp_path_, final_path_;
    int H_ = 0;
    int voxels_ = 0;
    std::size_t times_base_ = 0, density_base_ = 0, uncertainty_base_ = 0;
    int next_expected_t_ = 0;
    bool closed_ = false;
};

} // namespace rope::io
