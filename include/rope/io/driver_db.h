#pragma once
// Space-weather database (O(log N) lookup) and hourly driver-window builder.

#include "rope/core/datetime.h"
#include "rope/io/csv_reader.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rope::io {

struct DriverRow {
    TimePoint tp;
    float t1, t2, t3, t4;  // sin/cos harmonic features, always derived from tp
    float f10_41day_avg = std::numeric_limits<float>::quiet_NaN();  // fallback only; ignored when raw carries f10_41day_avg directly
    std::vector<std::pair<std::string, float>> raw;
    float get(std::string_view name) const;
};

inline std::vector<std::string> known_derived_driver_names() {
    return {"t1", "t2", "t3", "t4", "doy", "hour_int", "f10_41day_avg"};
}

class SpaceWeatherBin;  // forward-declare for friend

class SpaceWeatherDB {
public:
    // Format auto-detected from extension (".swbin" -> binary, else CSV).
    static SpaceWeatherDB from_file(const std::filesystem::path& path);

    explicit SpaceWeatherDB(const std::filesystem::path& csv_path);

    // Look up a single TimePoint; throws if not found.
    DriverRow lookup(TimePoint tp) const;

    TimePoint time_min() const { return times_.front(); }
    TimePoint time_max() const { return times_.back(); }
    std::size_t size()   const { return times_.size(); }

private:
    friend class SpaceWeatherBin;

    // Private constructor used by SpaceWeatherBin::load().
    SpaceWeatherDB(std::vector<TimePoint>          times,
                   std::vector<std::string>        raw_names,
                   std::vector<std::vector<float>> raw_data);
    static void reject_reserved_names(const std::vector<std::string>& names);

    std::vector<TimePoint>          times_;
    std::vector<std::string>        raw_names_;
    std::vector<std::vector<float>> raw_data_;  // raw_data_[j][i] = raw_names_[j]'s value at row i

    DriverRow make_row(std::size_t idx) const;
    float compute_f10_41day_avg(std::size_t idx) const;
};

class DriverWindowBuilder {
public:
    static std::vector<DriverRow> build(const SpaceWeatherDB& db,
                                        std::string_view      start_iso,
                                        int horizon,
                                        int seq_len);
};

} // namespace rope::io
