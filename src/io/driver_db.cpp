#include "rope/io/driver_db.h"
#include "rope/io/driver_bin.h"

#include "rope/core/text.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rope::io {

// ---------------------------------------------------------------------------
// DriverRow::get() — column name matching is case-insensitive (ASCII).
// ---------------------------------------------------------------------------
float DriverRow::get(std::string_view name) const {
    const std::string key = core::to_lower_ascii(name);

    if (key == "t1") return t1;
    if (key == "t2") return t2;
    if (key == "t3") return t3;
    if (key == "t4") return t4;

    for (const auto& [n, v] : raw)
        if (n == key) return v;

    if (key == "doy" || key == "hour_int") {
        int h, doy, yr;
        unpack(tp, h, doy, yr);
        return key == "hour_int" ? static_cast<float>(h)
                                  : static_cast<float>(doy) + h / 24.0f;
    }

    if (key == "f10_41day_avg") {
        if (std::isnan(f10_41day_avg))
            throw std::runtime_error(
                "DriverRow::get: 'f10_41day_avg' requires a raw 'f10' column, none found");
        return f10_41day_avg;
    }

    throw std::runtime_error("DriverRow::get: unknown column '" + std::string(name) + "'");
}

// ---------------------------------------------------------------------------
// from_file() — dispatch on extension
// ---------------------------------------------------------------------------
SpaceWeatherDB SpaceWeatherDB::from_file(const std::filesystem::path& path) {
    if (path.extension() == ".swbin")
        return SpaceWeatherBin::load(path);
    return SpaceWeatherDB{path};
}

// ---------------------------------------------------------------------------
// reject_reserved_names()
// ---------------------------------------------------------------------------
void SpaceWeatherDB::reject_reserved_names(const std::vector<std::string>& names) {
    // t1-t4 are pure functions of tp; f10_41day_avg is raw-if-present, else derived (not reserved).
    static const std::vector<std::string> reserved = {"t1", "t2", "t3", "t4"};
    for (const auto& n : names) {
        if (std::find(reserved.begin(), reserved.end(), n) != reserved.end())
            throw std::runtime_error(
                "SpaceWeatherDB: '" + n + "' is a reserved, always-derived driver name "
                "and cannot be supplied by a raw data source");
    }
}

// ---------------------------------------------------------------------------
// Private constructor (used by SpaceWeatherBin::load)
// ---------------------------------------------------------------------------
SpaceWeatherDB::SpaceWeatherDB(std::vector<TimePoint>          times,
                               std::vector<std::string>        raw_names,
                               std::vector<std::vector<float>> raw_data)
    : times_(std::move(times))
    , raw_names_(std::move(raw_names))
    , raw_data_(std::move(raw_data))
{
    // raw_names_ is the canonical (lowercase) name every later lookup matches against.
    for (auto& n : raw_names_) n = core::to_lower_ascii(n);
    reject_reserved_names(raw_names_);
}

// ---------------------------------------------------------------------------
// CSV constructor
// ---------------------------------------------------------------------------
SpaceWeatherDB::SpaceWeatherDB(const std::filesystem::path& csv_path) {
    CsvReader csv(csv_path);
    const std::size_t N = csv.nrows();

    times_.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        times_.push_back(parse_datetime(csv.get("datetime", i)));

    // raw_names_ is lowercased for matching; csv_headers keeps the CSV's own case for CsvReader.
    std::vector<std::string> csv_headers;
    for (const auto& name : csv.column_names())
        if (name != "datetime") {
            raw_names_.push_back(core::to_lower_ascii(name));
            csv_headers.push_back(name);
        }
    reject_reserved_names(raw_names_);

    raw_data_.resize(raw_names_.size());
    for (std::size_t j = 0; j < raw_names_.size(); ++j) {
        raw_data_[j].reserve(N);
        for (std::size_t i = 0; i < N; ++i)
            raw_data_[j].push_back(csv.get_float(csv_headers[j], i));
    }

    // Ensure sorted by time.
    if (!std::is_sorted(times_.begin(), times_.end())) {
        std::vector<std::size_t> idx(N);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b){ return times_[a] < times_[b]; });
        auto reorder = [&](std::vector<TimePoint>& v){
            auto tmp = v;
            for (std::size_t i = 0; i < N; ++i) v[i] = tmp[idx[i]];
        };
        reorder(times_);
        for (auto& col : raw_data_) {
            auto tmp = col;
            for (std::size_t i = 0; i < N; ++i) col[i] = tmp[idx[i]];
        }
    }
}

DriverRow SpaceWeatherDB::lookup(TimePoint tp) const {
    auto it = std::lower_bound(times_.begin(), times_.end(), tp);
    if (it == times_.end() || *it != tp)
        throw std::runtime_error(
            "SpaceWeatherDB: no entry for " + format_iso(tp));
    return make_row(static_cast<std::size_t>(it - times_.begin()));
}

DriverRow SpaceWeatherDB::make_row(std::size_t idx) const {
    DriverRow r;
    r.tp = times_[idx];
    float cont_doy;
    harmonics(r.tp, r.t1, r.t2, r.t3, r.t4, cont_doy);

    r.raw.reserve(raw_names_.size());
    for (std::size_t j = 0; j < raw_names_.size(); ++j)
        r.raw.emplace_back(raw_names_[j], raw_data_[j][idx]);

    // get()'s raw scan returns a supplied f10_41day_avg column before this field is ever consulted.
    const bool has_raw_f10_41day_avg =
        std::find(raw_names_.begin(), raw_names_.end(), "f10_41day_avg") != raw_names_.end();
    r.f10_41day_avg = has_raw_f10_41day_avg
        ? std::numeric_limits<float>::quiet_NaN()
        : compute_f10_41day_avg(idx);

    return r;
}

float SpaceWeatherDB::compute_f10_41day_avg(std::size_t idx) const {
    static constexpr std::size_t WINDOW_HOURS = 41 * 24;  // 984

    auto it = std::find(raw_names_.begin(), raw_names_.end(), "f10");
    if (it == raw_names_.end())
        return std::numeric_limits<float>::quiet_NaN();

    const auto& f10 = raw_data_[static_cast<std::size_t>(it - raw_names_.begin())];

    // Trailing window is by row count, not elapsed time (matches pandas .tail(984) on sorted data); fewer than WINDOW_HOURS rows near the start averages over what's available.
    std::size_t window_start = (idx + 1 >= WINDOW_HOURS) ? (idx + 1 - WINDOW_HOURS) : 0;
    std::size_t n = idx - window_start + 1;

    double sum = 0.0;  // double accumulator for precision over up to 984 samples
    for (std::size_t i = window_start; i <= idx; ++i)
        sum += f10[i];
    return static_cast<float>(sum / static_cast<double>(n));
}

std::vector<DriverRow> DriverWindowBuilder::build(const SpaceWeatherDB& db,
                                                   std::string_view      start_iso,
                                                   int horizon,
                                                   int seq_len) {
    TimePoint start      = floor_hour(parse_datetime(start_iso));
    int total            = (seq_len - 1) + horizon;
    TimePoint hist_start = start - static_cast<TimePoint>(seq_len - 1) * 3600;

    std::vector<DriverRow> rows;
    rows.reserve(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i)
        rows.push_back(db.lookup(hist_start + static_cast<TimePoint>(i) * 3600));
    return rows;
}

} // namespace rope::io
