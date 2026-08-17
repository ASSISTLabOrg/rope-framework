#include "rope/io/driver_cache.h"
#include "rope/core/datetime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rope::io {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Source registry
// ---------------------------------------------------------------------------
const std::unordered_map<std::string, DriverSource>& known_sources() {
    static const std::unordered_map<std::string, DriverSource> s = {
        {
            "celestrak_sw",
            {
                "https://celestrak.org/SpaceData/SW-Last5Years.csv",
                "CelesTrak space weather, last 5 years (hourly)"
            }
        },
        {
            "celestrak_sw_all",
            {
                "https://celestrak.org/SpaceData/SW-All.csv",
                "CelesTrak space weather, full historical record (hourly)"
            }
        },
    };
    return s;
}

// ---------------------------------------------------------------------------
// DriverCacheManager
// ---------------------------------------------------------------------------
DriverCacheManager::DriverCacheManager(fs::path cache_dir, int max_age_hours,
                                       std::unique_ptr<net::IHttpClient> http)
    : cache_dir_(std::move(cache_dir))
    , max_age_(std::chrono::seconds(
          static_cast<std::chrono::seconds::rep>(max_age_hours) * 3600))
    , http_(std::move(http))
{}

fs::path DriverCacheManager::get_path(const std::string& source) {
    const auto& sources = known_sources();
    if (sources.find(source) == sources.end())
        throw std::runtime_error(
            "DriverCacheManager: unknown source '" + source + "'. "
            "Known sources: celestrak_sw, celestrak_sw_all");

    fs::path dest = cache_dir_ / (source + ".swbin");

    if (!fs::exists(dest) || is_stale(dest)) {
        try {
            refresh(source, dest);
        } catch (const std::exception& e) {
            if (fs::exists(dest)) {
                // Fall back to stale cache rather than failing hard.
                return dest;
            }
            throw;
        }
    }
    return dest;
}

bool DriverCacheManager::is_stale(const fs::path& path) const {
    using Clock = std::chrono::file_clock;
    auto mtime  = fs::last_write_time(path);
    auto age    = Clock::now() - mtime;
    return age > max_age_;
}

void DriverCacheManager::refresh(const std::string& source,
                                 const fs::path& dest) {
    std::error_code ec;
    fs::create_directories(dest.parent_path(), ec);

    const auto& entry = known_sources().at(source);
    std::string raw   = download(entry.url);
    convert_celestrak_csv_to_swbin(raw, dest);
}

std::string DriverCacheManager::download(const std::string& url) {
    return http_->get(url);
}

// ─── PCHIP interpolation helpers (Fritsch-Carlson algorithm) ─────────────────
// Matches scipy.interpolate.PchipInterpolator behaviour: extrapolate=False
// returns NaN for query points outside [x.front(), x.back()].

namespace {

static double pchip_endpoint(double h0, double h1, double m0, double m1) {
    double d = ((2.0*h0 + h1)*m0 - h0*m1) / (h0 + h1);
    if (std::signbit(d) != std::signbit(m0))
        d = 0.0;
    if (std::signbit(m0) != std::signbit(m1) && std::abs(d) > 3.0*std::abs(m0))
        d = 3.0*m0;
    return d;
}

// Compute PCHIP monotone slopes at each knot.
static std::vector<double> pchip_slopes(const std::vector<double>& x,
                                         const std::vector<double>& y) {
    const int n = static_cast<int>(x.size());
    std::vector<double> d(n, 0.0);
    if (n < 2) return d;
    if (n == 2) {
        double m = (y[1] - y[0]) / (x[1] - x[0]);
        d[0] = d[1] = m;
        return d;
    }

    std::vector<double> h(n-1), delta(n-1);
    for (int i = 0; i < n-1; ++i) {
        h[i]     = x[i+1] - x[i];
        delta[i] = (y[i+1] - y[i]) / h[i];
    }

    // Interior slopes: weighted harmonic mean (preserves monotonicity).
    for (int i = 1; i < n-1; ++i) {
        if (delta[i-1] * delta[i] <= 0.0) {
            d[i] = 0.0;
        } else {
            double w1 = 2.0*h[i] + h[i-1];
            double w2 = h[i] + 2.0*h[i-1];
            d[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i]);
        }
    }
    // Endpoint slopes (n >= 3 guaranteed since n=2 returned early above).
    d[0]   = pchip_endpoint(h[0],   h[1],   delta[0],   delta[1]);
    d[n-1] = pchip_endpoint(h[n-2], h[n-3], delta[n-2], delta[n-3]);
    return d;
}

// Evaluate PCHIP at xq; returns NaN if out of [x[0], x[n-1]].
static double pchip_eval(const std::vector<double>& x,
                          const std::vector<double>& y,
                          const std::vector<double>& d,
                          double xq) {
    const int n = static_cast<int>(x.size());
    if (n < 1) return std::numeric_limits<double>::quiet_NaN();
    if (xq < x[0] || xq > x[n-1])
        return std::numeric_limits<double>::quiet_NaN();

    // Binary search for interval.
    int k = static_cast<int>(
        std::upper_bound(x.begin(), x.end(), xq) - x.begin()) - 1;
    if (k >= n-1) k = n-2;
    if (k < 0)    k = 0;

    double hk = x[k+1] - x[k];
    double t  = (xq - x[k]) / hk;
    double t2 = t*t, t3 = t2*t;
    double h00 =  2.0*t3 - 3.0*t2 + 1.0;
    double h10 =      t3 - 2.0*t2 + t;
    double h01 = -2.0*t3 + 3.0*t2;
    double h11 =      t3 -     t2;
    return h00*y[k] + h10*hk*d[k] + h01*y[k+1] + h11*hk*d[k+1];
}

// ─── CelesTrak CSV parsing ────────────────────────────────────────────────────

static std::string_view trim(std::string_view sv) {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\r' || sv.front() == '\t'))
        sv.remove_prefix(1);
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\r' || sv.back() == '\t'))
        sv.remove_suffix(1);
    return sv;
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    for (char c : line) {
        if (c == ',') { out.push_back(field); field.clear(); }
        else          { field += c; }
    }
    out.push_back(field);
    return out;
}

struct CelRow {
    int    year, month, day;   // from DATE column
    double f107;               // F10.7_OBS (NaN if missing)
    double kp[8];              // KP1..KP8 in raw tenths (NaN if missing)
    double ap[8];              // AP1..AP8, linear scale, no rescaling needed (NaN if missing)
};

static std::vector<CelRow> parse_celestrak(const std::string& raw) {
    std::istringstream ss(raw);
    std::string line;

    // Skip to the header line (starts with "DATE").
    std::vector<std::string> header;
    while (std::getline(ss, line)) {
        auto sv = trim(line);
        if (sv.starts_with("DATE")) {
            header = split_csv(std::string(sv));
            break;
        }
    }
    if (header.empty())
        throw std::runtime_error(
            "parse_celestrak: no DATE header found");

    // Find column indices.
    auto col_idx = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(header.size()); ++i)
            if (std::string(trim(header[i])) == name) return i;
        return -1;
    };

    const int ci_date = col_idx("DATE");
    const int ci_f107 = col_idx("F10.7_OBS");
    int ci_kp[8];
    int ci_ap[8];
    for (int k = 0; k < 8; ++k) {
        ci_kp[k] = col_idx("KP" + std::to_string(k+1));
        ci_ap[k] = col_idx("AP" + std::to_string(k+1));
    }

    if (ci_date < 0) throw std::runtime_error("parse_celestrak: missing DATE column");
    if (ci_f107 < 0) throw std::runtime_error("parse_celestrak: missing F10.7_OBS column");
    for (int k = 0; k < 8; ++k) {
        if (ci_kp[k] < 0)
            throw std::runtime_error(
                "parse_celestrak: missing KP" + std::to_string(k+1) + " column");
        if (ci_ap[k] < 0)
            throw std::runtime_error(
                "parse_celestrak: missing AP" + std::to_string(k+1) + " column");
    }

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    auto parse_double = [&](const std::string& s) -> double {
        try { return std::stod(s); } catch (...) { return NaN; }
    };

    std::vector<CelRow> rows;
    while (std::getline(ss, line)) {
        auto sv = trim(line);
        if (sv.empty() || !std::isdigit(static_cast<unsigned char>(sv[0])))
            continue;

        auto fields = split_csv(std::string(sv));
        if (static_cast<int>(fields.size()) <= ci_f107) continue;

        CelRow r;
        // Parse DATE: "YYYY-MM-DD"
        std::string date(trim(fields[ci_date]));
        if (std::sscanf(date.c_str(), "%d-%d-%d", &r.year, &r.month, &r.day) != 3)
            continue;

        r.f107 = parse_double(std::string(trim(fields[ci_f107])));
        for (int k = 0; k < 8; ++k) {
            r.kp[k] = (ci_kp[k] < static_cast<int>(fields.size()))
                ? parse_double(std::string(trim(fields[ci_kp[k]]))) : NaN;
            r.ap[k] = (ci_ap[k] < static_cast<int>(fields.size()))
                ? parse_double(std::string(trim(fields[ci_ap[k]]))) : NaN;
        }
        rows.push_back(r);
    }
    return rows;
}

// PCHIP-interpolates an 8-values-per-day 3-hourly series (Kp or Ap) to hourly, per day, using the
// next day's first value as the 24h continuity endpoint. get_v(row, k) extracts the k'th 3-hourly value.
template <typename GetV>
static std::unordered_map<TimePoint, std::array<double, 24>>
build_hourly_daily(const std::vector<CelRow>& rows, GetV get_v) {
    std::unordered_map<TimePoint, std::array<double, 24>> daily;
    daily.reserve(rows.size());

    for (std::size_t ri = 0; ri + 1 < rows.size(); ++ri) {
        const CelRow& r = rows[ri];
        bool ok = true;
        for (int k = 0; k < 8; ++k)
            if (!std::isfinite(get_v(r, k))) { ok = false; break; }
        if (!ok) continue;

        // Next day's first 3-hourly value as the 24h endpoint (ensures continuity).
        const CelRow& next = rows[ri + 1];
        if (!std::isfinite(get_v(next, 0))) continue;

        // Knots: hours 0,3,6,9,12,15,18,21,24 with values [0..7], next[0]
        std::vector<double> xk = {0,3,6,9,12,15,18,21,24};
        std::vector<double> yk(9);
        for (int k = 0; k < 8; ++k) yk[k] = get_v(r, k);
        yk[8] = get_v(next, 0);

        auto dk = pchip_slopes(xk, yk);

        std::string ds = std::to_string(r.year)  + "-" +
                         (r.month < 10 ? "0" : "") + std::to_string(r.month) + "-" +
                         (r.day   < 10 ? "0" : "") + std::to_string(r.day) +
                         " 00:00:00";
        TimePoint day_tp = rope::parse_datetime(ds);

        std::array<double, 24> hourly{};
        for (int h = 0; h < 24; ++h)
            hourly[h] = pchip_eval(xk, yk, dk, static_cast<double>(h));
        daily[day_tp] = hourly;
    }
    return daily;
}

} // anonymous namespace

// ─── Main conversion ─────────────────────────────────────────────────────────

void convert_celestrak_csv_to_swbin(const std::string& raw_csv, const fs::path& dest) {
    // 1. Parse raw CelesTrak CSV.
    auto rows = parse_celestrak(raw_csv);
    if (rows.empty())
        throw std::runtime_error(
            "convert_celestrak_csv_to_swbin: no data rows parsed");

    // 2. Build daily F10.7 series (hours-since-epoch as x).
    //    Reference epoch: midnight of first day with valid F10.7.
    std::vector<double> xf, yf;
    xf.reserve(rows.size());
    yf.reserve(rows.size());
    for (const auto& r : rows) {
        if (!std::isfinite(r.f107)) continue;
        std::string ds = std::to_string(r.year)  + "-" +
                         (r.month < 10 ? "0" : "") + std::to_string(r.month) + "-" +
                         (r.day   < 10 ? "0" : "") + std::to_string(r.day) +
                         " 00:00:00";
        TimePoint tp = rope::parse_datetime(ds);
        xf.push_back(static_cast<double>(tp) / 3600.0);
        yf.push_back(r.f107);
    }
    if (xf.size() < 2)
        throw std::runtime_error(
            "convert_celestrak_csv_to_swbin: too few valid F10.7 rows");

    // PCHIP slopes for F10.7.
    auto df107 = pchip_slopes(xf, yf);

    // 3. PCHIP Kp and Ap per day: 3-hourly → hourly. Kp stays in raw tenths (divided later);
    //    Ap needs no rescaling — CelesTrak already reports it on its native linear scale.
    //    Both models bundled with this framework use celestrak_sw but read different raw
    //    columns (tiegcm-aurora-v1 wants kp, wam-borealis-v1 wants ap), so both are always produced.
    auto kp_daily = build_hourly_daily(rows, [](const CelRow& r, int k) { return r.kp[k]; });
    auto ap_daily = build_hourly_daily(rows, [](const CelRow& r, int k) { return r.ap[k]; });

    // 4. Generate hourly output: inner join of F10.7, Kp, and Ap.
    //    Walk hour-by-hour over the F10.7 range; look up Kp/Ap from their daily maps.
    const double x_min = xf.front();
    const double x_max = xf.back();

    std::vector<TimePoint>  times;
    std::vector<float>      f10v, kpv, apv;

    for (double xh = std::ceil(x_min); xh <= x_max; xh += 1.0) {
        TimePoint tp = static_cast<TimePoint>(std::llround(xh * 3600.0));
        TimePoint day_tp = (tp / 86400) * 86400;
        int h = static_cast<int>((tp % 86400) / 3600);

        auto it_kp = kp_daily.find(day_tp);
        auto it_ap = ap_daily.find(day_tp);
        if (it_kp == kp_daily.end() || it_ap == ap_daily.end()) continue;

        double f107_val = pchip_eval(xf, yf, df107, xh);
        if (!std::isfinite(f107_val)) continue;

        double kp_val = it_kp->second[h] / 10.0;  // divide raw tenths by 10
        double ap_val = it_ap->second[h];
        if (!std::isfinite(kp_val) || !std::isfinite(ap_val)) continue;

        times.push_back(tp);
        f10v.push_back(static_cast<float>(f107_val));
        kpv.push_back(static_cast<float>(kp_val));
        apv.push_back(static_cast<float>(ap_val));
    }

    if (times.empty())
        throw std::runtime_error(
            "convert_celestrak_csv_to_swbin: no output rows after merging F10.7, Kp, and Ap");

    // 5. Write .swbin directly (magic RPSW, matches driver_bin.h v2 format).
    //    Header (16 bytes): magic uint32, version uint32, nrows uint32, ncols uint32
    //    Name table (3 entries): "f10", "kp", "ap" (doy/hour_int are derived from tp
    //    on read and never need to be stored — see DriverRow::get()).
    //    Records: tp int64, then ncols float32 values in name-table order.
    std::error_code ec;
    fs::create_directories(dest.parent_path(), ec);

    std::ofstream f(dest, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error(
        "convert_celestrak_csv_to_swbin: cannot write " + dest.string());

    const std::uint32_t magic   = 0x52505357u;
    const std::uint32_t version = 2u;
    const std::uint32_t nrows   = static_cast<std::uint32_t>(times.size());
    const std::uint32_t ncols   = 3u;
    f.write(reinterpret_cast<const char*>(&magic),   4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&nrows),   4);
    f.write(reinterpret_cast<const char*>(&ncols),   4);

    for (const std::string& name : {std::string("f10"), std::string("kp"), std::string("ap")}) {
        const auto name_len = static_cast<std::uint32_t>(name.size());
        f.write(reinterpret_cast<const char*>(&name_len), 4);
        f.write(name.data(), static_cast<std::streamsize>(name_len));
    }

    for (std::uint32_t i = 0; i < nrows; ++i) {
        auto tp_raw = static_cast<std::int64_t>(times[i]);
        f.write(reinterpret_cast<const char*>(&tp_raw),  8);
        f.write(reinterpret_cast<const char*>(&f10v[i]), 4);
        f.write(reinterpret_cast<const char*>(&kpv[i]),  4);
        f.write(reinterpret_cast<const char*>(&apv[i]),  4);
    }
    if (!f) throw std::runtime_error(
        "convert_celestrak_csv_to_swbin: write failed for " + dest.string());
}

} // namespace rope::io
