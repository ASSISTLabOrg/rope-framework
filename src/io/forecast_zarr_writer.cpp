#include "rope/io/forecast_zarr_writer.h"
#include "rope/core/version.h"

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace rope::io {

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

// Mirrors grid_interpolator.cpp's axis formulas; not shared to avoid a layering cycle.
std::vector<double> lst_axis(const GridSpec& s) {
    std::vector<double> v(s.n_lst);
    for (int i = 0; i < s.n_lst; ++i) v[i] = i * (24.0 / s.n_lst);
    return v;
}
std::vector<double> lat_axis(const GridSpec& s) {
    std::vector<double> v(s.n_lat);
    const double span = s.lat_max_deg - s.lat_min_deg;
    for (int i = 0; i < s.n_lat; ++i) v[i] = s.lat_min_deg + i * (span / (s.n_lat - 1));
    return v;
}
std::vector<double> alt_axis(const GridSpec& s) {
    std::vector<double> v(s.n_alt);
    const double span = s.alt_max_km - s.alt_min_km;
    for (int i = 0; i < s.n_alt; ++i) v[i] = s.alt_min_km + i * (span / (s.n_alt - 1));
    return v;
}

// Strips to [A-Za-z0-9] so the store-directory name has no chars illegal on Windows.
std::string sanitize_for_path(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (std::isalnum(static_cast<unsigned char>(c))) out.push_back(c);
    return out;
}

fs::path make_temp_dir_name(const fs::path& container, const std::string& store_name) {
    auto ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::random_device rd;
    return container / ("." + store_name + ".tmp-" + std::to_string(ticks) + "-" + std::to_string(rd()));
}

void write_text_file(const fs::path& file, const std::string& text) {
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("ForecastZarrWriter: cannot open " + file.string());
    f << text;
    if (!f)
        throw std::runtime_error("ForecastZarrWriter: write failed for " + file.string());
}

void write_json_file(const fs::path& file, const json& j) {
    write_text_file(file, j.dump(2));
}

void write_zlib_chunk(const fs::path& file, const void* data, std::size_t nbytes) {
    uLongf bound = compressBound(static_cast<uLong>(nbytes));
    std::vector<unsigned char> buf(bound);
    uLongf outlen = bound;
    int rc = compress2(buf.data(), &outlen,
                        static_cast<const unsigned char*>(data), static_cast<uLong>(nbytes),
                        /*level=*/5);
    if (rc != Z_OK)
        throw std::runtime_error(
            "ForecastZarrWriter: zlib compress2 failed (rc=" + std::to_string(rc) +
            ") for " + file.string());
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("ForecastZarrWriter: cannot open " + file.string());
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(outlen));
    if (!f)
        throw std::runtime_error("ForecastZarrWriter: write failed for " + file.string());
}

struct ArraySpec {
    std::string name;
    std::vector<long long> shape;
    std::vector<long long> chunks;
    std::string dtype;
    json fill_value;
    std::vector<std::string> dims;
    std::string units;  // empty = omit
};

void make_array_dir(const fs::path& store, const ArraySpec& a) {
    fs::path dir = store / a.name;
    fs::create_directory(dir);
    json zarray = {
        {"zarr_format", 2},
        {"shape", a.shape},
        {"chunks", a.chunks},
        {"dtype", a.dtype},
        {"compressor", {{"id", "zlib"}, {"level", 5}}},
        {"fill_value", a.fill_value},
        {"order", "C"},
        {"filters", nullptr},
    };
    write_json_file(dir / ".zarray", zarray);

    json zattrs = {{"_ARRAY_DIMENSIONS", a.dims}};
    if (!a.units.empty()) zattrs["units"] = a.units;
    write_json_file(dir / ".zattrs", zattrs);
}

} // namespace

struct ForecastZarrWriter::Impl {
    GridSpec shape;
    int H = 0, K = 0;
    int voxels = 0;
    fs::path container_path;
    fs::path tmp_path;
    fs::path final_store_path;
    std::vector<std::int64_t> times_buf;  // flushed in close()
    int next_t = 0;
    bool closed = false;
};

ForecastZarrWriter ForecastZarrWriter::open(
    const GridSpec& shape, int H, int K,
    const std::string& model_kind,
    const std::string& forecast_start, const fs::path& path)
{
    if (H <= 0)
        throw std::runtime_error("ForecastZarrWriter::open: H must be > 0");
    if (K <= 0)
        throw std::runtime_error("ForecastZarrWriter::open: K must be > 0");

    // `path` is a container directory: create if missing, reuse if present.
    if (!fs::exists(path)) {
        fs::create_directory(path);
    } else if (!fs::is_directory(path)) {
        throw std::runtime_error(
            "ForecastZarrWriter::open: " + path.string() + " exists and is not a directory");
    }

    const std::string store_name =
        "forecast_" + sanitize_for_path(forecast_start) + "_H" + std::to_string(H);
    fs::path store_path = path / store_name;
    if (fs::exists(store_path))
        throw std::runtime_error(
            "ForecastZarrWriter::open: " + store_path.string() +
            " already exists (same start+horizon exported into this container before; "
            "remove it or choose a different --zarr directory)");

    fs::path tmp_path = make_temp_dir_name(path, store_name);
    fs::create_directory(tmp_path);

    auto impl = std::make_unique<Impl>();
    impl->shape = shape;
    impl->H = H;
    impl->K = K;
    impl->voxels = shape.voxels();
    impl->container_path = path;
    impl->tmp_path = tmp_path;
    impl->final_store_path = store_path;
    impl->times_buf.assign(static_cast<std::size_t>(H), 0);

    try {
        write_json_file(tmp_path / ".zgroup", json{{"zarr_format", 2}});
        write_json_file(tmp_path / ".zattrs", json{
            {"Conventions",              "CF-1.8"},
            {"model_kind",               model_kind},
            {"rope_version",             rope::version::string()},
            {"forecast_start",           forecast_start},
            {"forecast_horizon_hours",   H},
        });

        // fill_value must stay null: 0/0.0 makes xarray mask real zero values as NaN.
        make_array_dir(tmp_path, {"time", {H}, {H}, "<i8", nullptr,
                                  {"time"}, "seconds since 1970-01-01T00:00:00Z"});
        make_array_dir(tmp_path, {"lst", {shape.n_lst}, {shape.n_lst}, "<f8", nullptr,
                                  {"lst"}, "hours"});
        make_array_dir(tmp_path, {"lat", {shape.n_lat}, {shape.n_lat}, "<f8", nullptr,
                                  {"lat"}, "degrees_north"});
        make_array_dir(tmp_path, {"alt", {shape.n_alt}, {shape.n_alt}, "<f8", nullptr,
                                  {"alt"}, "km"});
        make_array_dir(tmp_path, {"latent", {K}, {K}, "<i4", nullptr,
                                  {"latent"}, ""});
        make_array_dir(tmp_path, {"density",
                                  {H, shape.n_lst, shape.n_lat, shape.n_alt},
                                  {1, shape.n_lst, shape.n_lat, shape.n_alt}, "<f4", nullptr,
                                  {"time", "lst", "lat", "alt"}, "kg m-3"});
        make_array_dir(tmp_path, {"uncertainty",
                                  {H, shape.n_lst, shape.n_lat, shape.n_alt},
                                  {1, shape.n_lst, shape.n_lat, shape.n_alt}, "<f4", nullptr,
                                  {"time", "lst", "lat", "alt"}, "kg m-3"});
        make_array_dir(tmp_path, {"latent_mean", {H, K}, {H, K}, "<f4", nullptr,
                                  {"time", "latent"}, ""});

        auto lst_v = lst_axis(shape);
        auto lat_v = lat_axis(shape);
        auto alt_v = alt_axis(shape);
        std::vector<std::int32_t> latent_idx(static_cast<std::size_t>(K));
        for (int i = 0; i < K; ++i) latent_idx[static_cast<std::size_t>(i)] = i;

        write_zlib_chunk(tmp_path / "lst" / "0",    lst_v.data(),      lst_v.size() * sizeof(double));
        write_zlib_chunk(tmp_path / "lat" / "0",    lat_v.data(),      lat_v.size() * sizeof(double));
        write_zlib_chunk(tmp_path / "alt" / "0",    alt_v.data(),      alt_v.size() * sizeof(double));
        write_zlib_chunk(tmp_path / "latent" / "0", latent_idx.data(), latent_idx.size() * sizeof(std::int32_t));
    } catch (...) {
        std::error_code ec;
        fs::remove_all(tmp_path, ec);
        throw;
    }

    ForecastZarrWriter w;
    w.impl_ = std::move(impl);
    return w;
}

ForecastZarrWriter::ForecastZarrWriter(ForecastZarrWriter&&) noexcept = default;
ForecastZarrWriter& ForecastZarrWriter::operator=(ForecastZarrWriter&&) noexcept = default;

ForecastZarrWriter::~ForecastZarrWriter() {
    if (impl_ && !impl_->closed) {
        std::error_code ec;
        fs::remove_all(impl_->tmp_path, ec);
    }
}

const fs::path& ForecastZarrWriter::store_path() const noexcept {
    return impl_->final_store_path;
}

void ForecastZarrWriter::write_chunk(
    int t_offset, std::span<const std::int64_t> times,
    std::span<const float> density, std::span<const float> uncertainty)
{
    Impl& im = *impl_;
    const std::size_t count = times.size();

    if (t_offset != im.next_t)
        throw std::logic_error(
            "ForecastZarrWriter::write_chunk: expected t_offset=" +
            std::to_string(im.next_t) + ", got " + std::to_string(t_offset));
    if (density.size()     != count * static_cast<std::size_t>(im.voxels) ||
        uncertainty.size() != count * static_cast<std::size_t>(im.voxels))
        throw std::runtime_error(
            "ForecastZarrWriter::write_chunk: array sizes inconsistent with times.size() * voxels");
    if (static_cast<std::size_t>(t_offset) + count > static_cast<std::size_t>(im.H))
        throw std::runtime_error("ForecastZarrWriter::write_chunk: chunk exceeds H");

    for (std::size_t i = 0; i < count; ++i) {
        const int t = t_offset + static_cast<int>(i);
        im.times_buf[static_cast<std::size_t>(t)] = times[i];

        const std::string suffix = std::to_string(t) + ".0.0.0";
        write_zlib_chunk(im.tmp_path / "density" / suffix,
                          density.data() + i * static_cast<std::size_t>(im.voxels),
                          static_cast<std::size_t>(im.voxels) * sizeof(float));
        write_zlib_chunk(im.tmp_path / "uncertainty" / suffix,
                          uncertainty.data() + i * static_cast<std::size_t>(im.voxels),
                          static_cast<std::size_t>(im.voxels) * sizeof(float));
    }
    im.next_t += static_cast<int>(count);
}

void ForecastZarrWriter::write_latent(std::span<const float> mu_lat) {
    Impl& im = *impl_;
    if (mu_lat.size() != static_cast<std::size_t>(im.H) * static_cast<std::size_t>(im.K))
        throw std::runtime_error("ForecastZarrWriter::write_latent: expected H*K floats");
    write_zlib_chunk(im.tmp_path / "latent_mean" / "0.0",
                      mu_lat.data(), mu_lat.size() * sizeof(float));
}

void ForecastZarrWriter::close() {
    Impl& im = *impl_;
    if (im.closed)
        throw std::logic_error("ForecastZarrWriter::close: already closed");
    if (im.next_t != im.H)
        throw std::logic_error(
            "ForecastZarrWriter::close: only " + std::to_string(im.next_t) +
            " of " + std::to_string(im.H) + " timesteps were written");

    write_zlib_chunk(im.tmp_path / "time" / "0",
                      im.times_buf.data(), im.times_buf.size() * sizeof(std::int64_t));

    fs::rename(im.tmp_path, im.final_store_path);
    im.closed = true;
}

} // namespace rope::io
