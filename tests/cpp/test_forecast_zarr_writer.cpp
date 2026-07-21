#include <catch2/catch_test_macros.hpp>
#include "rope/io/forecast_zarr_writer.h"

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using rope::GridSpec;
using rope::io::ForecastZarrWriter;

namespace {

GridSpec small_shape() {
    GridSpec s;
    s.n_lst = 4; s.n_lat = 3; s.n_alt = 5;
    s.lat_min_deg = -60.0; s.lat_max_deg = 60.0;
    s.alt_min_km  = 150.0; s.alt_max_km  = 500.0;
    return s;
}

std::vector<unsigned char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    REQUIRE(f);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

json read_json(const fs::path& p) {
    return json::parse(read_file(p));
}

template <class T>
std::vector<T> inflate_chunk(const fs::path& p, std::size_t n_elems) {
    auto raw = read_file(p);
    std::vector<T> out(n_elems);
    uLongf dest_len = static_cast<uLongf>(n_elems * sizeof(T));
    int rc = uncompress(reinterpret_cast<Bytef*>(out.data()), &dest_len,
                        raw.data(), static_cast<uLong>(raw.size()));
    REQUIRE(rc == Z_OK);
    REQUIRE(dest_len == n_elems * sizeof(T));
    return out;
}

fs::path fresh_dir(const std::string& name) {
    auto p = fs::temp_directory_path() / name;
    fs::remove_all(p);
    return p;
}

} // namespace

TEST_CASE("ForecastZarrWriter: round-trip preserves density/uncertainty/latent/time exactly") {
    auto container = fresh_dir("rope_zarr_roundtrip");
    auto shape = small_shape();
    const int H = 3, K = 2;
    const int voxels = shape.voxels();

    auto writer = ForecastZarrWriter::open(shape, H, K, "stacked_ensemble",
                                           "2024-02-09T00:00:00Z", container);

    std::vector<std::int64_t> times = {1000, 2000, 3000};
    std::vector<float> density(static_cast<std::size_t>(H) * voxels);
    std::vector<float> uncertainty(static_cast<std::size_t>(H) * voxels);
    for (std::size_t i = 0; i < density.size(); ++i) {
        density[i]     = static_cast<float>(i) * 1.5f;
        uncertainty[i]  = static_cast<float>(i) * 0.1f;
    }
    writer.write_chunk(0, times, density, uncertainty);

    std::vector<float> latent_mean(static_cast<std::size_t>(H) * K);
    for (std::size_t i = 0; i < latent_mean.size(); ++i) latent_mean[i] = static_cast<float>(i) + 0.25f;
    writer.write_latent(latent_mean);

    auto store = writer.store_path();
    writer.close();

    REQUIRE(fs::is_directory(store));
    for (const auto& entry : fs::directory_iterator(container))
        CHECK(entry.path().filename().string().find(".tmp-") == std::string::npos);  // no leftover temp dir

    auto times_out = inflate_chunk<std::int64_t>(store / "time" / "0", static_cast<std::size_t>(H));
    CHECK(times_out == times);

    for (int t = 0; t < H; ++t) {
        auto d = inflate_chunk<float>(store / "density" / (std::to_string(t) + ".0.0.0"), static_cast<std::size_t>(voxels));
        auto u = inflate_chunk<float>(store / "uncertainty" / (std::to_string(t) + ".0.0.0"), static_cast<std::size_t>(voxels));
        for (int v = 0; v < voxels; ++v) {
            CHECK(d[static_cast<std::size_t>(v)] == density[static_cast<std::size_t>(t) * voxels + v]);
            CHECK(u[static_cast<std::size_t>(v)] == uncertainty[static_cast<std::size_t>(t) * voxels + v]);
        }
    }

    auto lat_mean_out = inflate_chunk<float>(store / "latent_mean" / "0.0", static_cast<std::size_t>(H) * K);
    CHECK(lat_mean_out == latent_mean);

    // lst/lat/alt coordinate arrays match the same linspace GridInterpolator uses.
    auto lst_out = inflate_chunk<double>(store / "lst" / "0", static_cast<std::size_t>(shape.n_lst));
    for (int i = 0; i < shape.n_lst; ++i)
        CHECK(lst_out[static_cast<std::size_t>(i)] == i * (24.0 / shape.n_lst));
    auto lat_out = inflate_chunk<double>(store / "lat" / "0", static_cast<std::size_t>(shape.n_lat));
    CHECK(lat_out.front() == shape.lat_min_deg);
    CHECK(lat_out.back()  == shape.lat_max_deg);

    // _ARRAY_DIMENSIONS / dtype / chunking match the designed schema.
    auto density_zarray = read_json(store / "density" / ".zarray");
    CHECK(density_zarray["dtype"] == "<f4");
    CHECK(density_zarray["shape"] == json::array({H, shape.n_lst, shape.n_lat, shape.n_alt}));
    CHECK(density_zarray["chunks"] == json::array({1, shape.n_lst, shape.n_lat, shape.n_alt}));
    auto density_zattrs = read_json(store / "density" / ".zattrs");
    CHECK(density_zattrs["_ARRAY_DIMENSIONS"] == json::array({"time", "lst", "lat", "alt"}));

    // Root attrs carry model identity.
    auto root_attrs = read_json(store / ".zattrs");
    CHECK(root_attrs["model_kind"] == "stacked_ensemble");
    CHECK(root_attrs["forecast_horizon_hours"] == H);

    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: store directory name contains no ':'") {
    auto container = fresh_dir("rope_zarr_colonfree");
    auto shape = small_shape();
    auto writer = ForecastZarrWriter::open(shape, 1, 1, "k", "2024-02-09T00:00:00Z", container);
    CHECK(writer.store_path().filename().string().find(':') == std::string::npos);
    std::vector<std::int64_t> t = {1};
    std::vector<float> d(static_cast<std::size_t>(shape.voxels()), 0.0f);
    std::vector<float> u(static_cast<std::size_t>(shape.voxels()), 0.0f);
    writer.write_chunk(0, t, d, u);
    writer.write_latent(std::vector<float>{0.0f});
    writer.close();
    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: container directory is created if missing") {
    auto container = fresh_dir("rope_zarr_autocreate_container");
    REQUIRE_FALSE(fs::exists(container));
    auto writer = ForecastZarrWriter::open(small_shape(), 1, 1, "k", "2024-01-01T00:00:00Z", container);
    CHECK(fs::is_directory(container));
    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: existing non-empty container directory is reused") {
    auto container = fresh_dir("rope_zarr_reuse_container");
    fs::create_directory(container);
    { std::ofstream f(container / "unrelated_prior_export.txt"); f << "hi"; }

    auto writer = ForecastZarrWriter::open(small_shape(), 1, 1, "k", "2024-05-05T00:00:00Z", container);
    auto store = writer.store_path();
    CHECK(store.parent_path() == container);
    CHECK(fs::exists(container / "unrelated_prior_export.txt"));  // untouched

    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: container path that is a regular file throws immediately") {
    auto container = fresh_dir("rope_zarr_container_is_file");
    { std::ofstream f(container); f << "not a directory"; }

    REQUIRE_THROWS_AS(
        ForecastZarrWriter::open(small_shape(), 1, 1, "k", "2024-01-01T00:00:00Z", container),
        std::runtime_error);

    fs::remove(container);
}

TEST_CASE("ForecastZarrWriter: computed store subdirectory collision throws") {
    auto container = fresh_dir("rope_zarr_collision");
    auto shape = small_shape();

    auto writer1 = ForecastZarrWriter::open(shape, 2, 1, "k", "2024-06-01T00:00:00Z", container);
    std::vector<std::int64_t> t = {1, 2};
    std::vector<float> d(static_cast<std::size_t>(shape.voxels()) * 2, 0.0f);
    std::vector<float> u(static_cast<std::size_t>(shape.voxels()) * 2, 0.0f);
    writer1.write_chunk(0, t, d, u);
    writer1.write_latent(std::vector<float>(2, 0.0f));
    writer1.close();

    // Same start + same horizon -> same computed store subdirectory name.
    REQUIRE_THROWS_AS(
        ForecastZarrWriter::open(shape, 2, 1, "k", "2024-06-01T00:00:00Z", container),
        std::runtime_error);

    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: write_chunk out-of-order t_offset throws") {
    auto container = fresh_dir("rope_zarr_out_of_order");
    auto shape = small_shape();
    auto writer = ForecastZarrWriter::open(shape, 2, 1, "k", "2024-01-01T00:00:00Z", container);

    std::vector<std::int64_t> t = {1};
    std::vector<float> d(static_cast<std::size_t>(shape.voxels()), 0.0f);
    std::vector<float> u(static_cast<std::size_t>(shape.voxels()), 0.0f);
    REQUIRE_THROWS_AS(writer.write_chunk(1, t, d, u), std::logic_error);  // must start at 0

    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: close() before all timesteps written throws, and leaves no store directory") {
    auto container = fresh_dir("rope_zarr_incomplete");
    auto shape = small_shape();
    auto writer = ForecastZarrWriter::open(shape, 3, 1, "k", "2024-01-01T00:00:00Z", container);

    std::vector<std::int64_t> t = {1};
    std::vector<float> d(static_cast<std::size_t>(shape.voxels()), 0.0f);
    std::vector<float> u(static_cast<std::size_t>(shape.voxels()), 0.0f);
    writer.write_chunk(0, t, d, u);

    REQUIRE_THROWS_AS(writer.close(), std::logic_error);
    fs::remove_all(container);
}

TEST_CASE("ForecastZarrWriter: destructor cleans up the temp directory when close() is never reached") {
    auto container = fresh_dir("rope_zarr_destructor_cleanup");
    auto shape = small_shape();
    fs::path store_path;
    {
        auto writer = ForecastZarrWriter::open(shape, 1, 1, "k", "2024-01-01T00:00:00Z", container);
        store_path = writer.store_path();
    }
    CHECK_FALSE(fs::exists(store_path));

    for (const auto& entry : fs::directory_iterator(container))
        CHECK(entry.path().filename().string().find(".tmp-") == std::string::npos);

    fs::remove_all(container);
}
