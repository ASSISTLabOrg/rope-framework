#include <catch2/catch_test_macros.hpp>
#include "rope/io/forecast_grid_bin.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace fs = std::filesystem;
using rope::ForecastGrid;
using rope::GridSpec;
using rope::io::ForecastGridBin;
using rope::io::ForecastCacheMissingError;
using rope::io::ForecastCacheCorruptError;

static GridSpec small_shape() {
    GridSpec s;
    s.n_lst = 4; s.n_lat = 3; s.n_alt = 5;
    s.lat_min_deg = -60.0; s.lat_max_deg = 60.0;
    s.alt_min_km  = 150.0; s.alt_max_km  = 500.0;
    return s;
}

static ForecastGrid make_grid(int H, float den, float unc, const GridSpec& shape = small_shape()) {
    ForecastGrid g;
    g.shape = shape;
    g.H = H;
    g.density.assign(static_cast<std::size_t>(H) * shape.voxels(), den);
    g.uncertainty.assign(static_cast<std::size_t>(H) * shape.voxels(), unc);
    g.times.resize(static_cast<std::size_t>(H));
    for (int t = 0; t < H; ++t) g.times[t] = 1700000000 + t * 3600;
    return g;
}

TEST_CASE("ForecastGridBin: round-trip preserves all fields exactly") {
    auto path = fs::temp_directory_path() / "rope_fgbin_roundtrip.bin";
    fs::remove(path);

    auto grid = make_grid(3, 1.5f, 0.25f);
    ForecastGridBin::save(grid, path);
    auto loaded = ForecastGridBin::load(path);

    CHECK(loaded.H == grid.H);
    CHECK(loaded.shape.n_lst == grid.shape.n_lst);
    CHECK(loaded.shape.n_lat == grid.shape.n_lat);
    CHECK(loaded.shape.n_alt == grid.shape.n_alt);
    CHECK(loaded.shape.lat_min_deg == grid.shape.lat_min_deg);
    CHECK(loaded.shape.lat_max_deg == grid.shape.lat_max_deg);
    CHECK(loaded.shape.alt_min_km == grid.shape.alt_min_km);
    CHECK(loaded.shape.alt_max_km == grid.shape.alt_max_km);
    CHECK(loaded.times == grid.times);
    CHECK(loaded.density == grid.density);
    CHECK(loaded.uncertainty == grid.uncertainty);

    fs::remove(path);
}

TEST_CASE("ForecastGridBin: load on a nonexistent path throws ForecastCacheMissingError") {
    auto path = fs::temp_directory_path() / "rope_fgbin_does_not_exist.bin";
    fs::remove(path);
    REQUIRE_THROWS_AS(ForecastGridBin::load(path), ForecastCacheMissingError);
}

TEST_CASE("ForecastGridBin: bad magic throws ForecastCacheCorruptError") {
    auto path = fs::temp_directory_path() / "rope_fgbin_badmagic.bin";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        std::uint32_t bad_magic = 0xDEADBEEFu;
        char padding[56] = {};  // rest of the 60-byte header, contents don't matter
        f.write(reinterpret_cast<const char*>(&bad_magic), 4);
        f.write(padding, sizeof(padding));
    }
    REQUIRE_THROWS_AS(ForecastGridBin::load(path), ForecastCacheCorruptError);
    fs::remove(path);
}

TEST_CASE("ForecastGridBin: unsupported version throws ForecastCacheCorruptError") {
    auto path = fs::temp_directory_path() / "rope_fgbin_badversion.bin";
    auto grid = make_grid(1, 1.0f, 1.0f);
    ForecastGridBin::save(grid, path);

    // Overwrite just the version field (bytes 4..8) with a bogus value.
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        std::uint32_t bad_version = 99;
        f.seekp(4);
        f.write(reinterpret_cast<const char*>(&bad_version), 4);
    }
    REQUIRE_THROWS_AS(ForecastGridBin::load(path), ForecastCacheCorruptError);
    fs::remove(path);
}

TEST_CASE("ForecastGridBin: truncated file throws ForecastCacheCorruptError") {
    auto path = fs::temp_directory_path() / "rope_fgbin_truncated.bin";
    auto grid = make_grid(3, 1.0f, 1.0f);
    ForecastGridBin::save(grid, path);

    // Truncate to just the header — body is missing.
    fs::resize_file(path, 60);
    REQUIRE_THROWS_AS(ForecastGridBin::load(path), ForecastCacheCorruptError);
    fs::remove(path);
}

TEST_CASE("ForecastGridBin: save() rejects grid arrays inconsistent with H*voxels") {
    auto path = fs::temp_directory_path() / "rope_fgbin_badsize.bin";
    auto grid = make_grid(3, 1.0f, 1.0f);
    grid.density.resize(grid.density.size() - 1);  // now inconsistent
    REQUIRE_THROWS_AS(ForecastGridBin::save(grid, path), std::runtime_error);
}

TEST_CASE("ForecastGridBinWriter: chunked writes produce a byte-identical file to ForecastGridBin::save()") {
    using rope::io::ForecastGridBinWriter;

    auto ref_path = fs::temp_directory_path() / "rope_fgbinw_ref.bin";
    auto chunked_path = fs::temp_directory_path() / "rope_fgbinw_chunked.bin";
    fs::remove(ref_path);
    fs::remove(chunked_path);

    const int H = 10;
    auto shape = small_shape();
    auto grid = make_grid(H, 2.5f, 0.5f, shape);
    const int voxels = shape.voxels();
    for (int t = 0; t < H; ++t) {
        grid.times[t] = 1700000000 + t * 3600;
        for (int v = 0; v < voxels; ++v) {
            grid.density[static_cast<std::size_t>(t) * voxels + v]     = static_cast<float>(t * 100 + v);
            grid.uncertainty[static_cast<std::size_t>(t) * voxels + v] = static_cast<float>(t);
        }
    }
    ForecastGridBin::save(grid, ref_path);

    auto writer = ForecastGridBinWriter::open(shape, H, chunked_path);
    for (int t_offset : {0, 4, 8}) {
        int count = (t_offset == 8) ? 2 : 4;
        writer.write_chunk(t_offset,
            std::span(grid.times).subspan(t_offset, count),
            std::span(grid.density).subspan(static_cast<std::size_t>(t_offset) * voxels,
                                            static_cast<std::size_t>(count) * voxels),
            std::span(grid.uncertainty).subspan(static_cast<std::size_t>(t_offset) * voxels,
                                                static_cast<std::size_t>(count) * voxels));
    }
    writer.close();

    auto ref_bytes = fs::file_size(ref_path);
    auto chunked_bytes = fs::file_size(chunked_path);
    REQUIRE(ref_bytes == chunked_bytes);
    {
        std::ifstream fa(ref_path, std::ios::binary), fb(chunked_path, std::ios::binary);
        std::vector<char> a(std::istreambuf_iterator<char>(fa), {});
        std::vector<char> b(std::istreambuf_iterator<char>(fb), {});
        CHECK(a == b);
    }

    fs::remove(ref_path);
    fs::remove(chunked_path);
}

TEST_CASE("ForecastGridBinWriter: write_chunk with an out-of-order t_offset throws") {
    using rope::io::ForecastGridBinWriter;
    auto path = fs::temp_directory_path() / "rope_fgbinw_outoforder.bin";
    fs::remove(path);

    auto shape = small_shape();
    auto writer = ForecastGridBinWriter::open(shape, 5, path);
    std::vector<std::int64_t> times{1, 2};
    std::vector<float> density(2 * shape.voxels(), 0.0f);
    std::vector<float> uncertainty(2 * shape.voxels(), 0.0f);

    REQUIRE_THROWS_AS(writer.write_chunk(2, times, density, uncertainty), std::logic_error);
}

TEST_CASE("ForecastGridBinWriter: close() before all timesteps are written throws") {
    using rope::io::ForecastGridBinWriter;
    auto path = fs::temp_directory_path() / "rope_fgbinw_incomplete.bin";
    fs::remove(path);

    auto shape = small_shape();
    auto writer = ForecastGridBinWriter::open(shape, 5, path);
    std::vector<std::int64_t> times{1, 2};
    std::vector<float> density(2 * shape.voxels(), 0.0f);
    std::vector<float> uncertainty(2 * shape.voxels(), 0.0f);
    writer.write_chunk(0, times, density, uncertainty);

    REQUIRE_THROWS_AS(writer.close(), std::logic_error);
}

TEST_CASE("ForecastGridBinWriter: unclosed destructor leaves a prior good cache file untouched") {
    using rope::io::ForecastGridBinWriter;
    auto path = fs::temp_directory_path() / "rope_fgbinw_midfail.bin";
    fs::remove(path);

    auto good_grid = make_grid(2, 7.0f, 0.7f);
    ForecastGridBin::save(good_grid, path);
    auto before = ForecastGridBin::load(path);

    {
        auto shape = small_shape();
        auto writer = ForecastGridBinWriter::open(shape, 5, path);
        std::vector<std::int64_t> times{1, 2};
        std::vector<float> density(2 * shape.voxels(), 0.0f);
        std::vector<float> uncertainty(2 * shape.voxels(), 0.0f);
        writer.write_chunk(0, times, density, uncertainty);
    }

    auto after = ForecastGridBin::load(path);
    CHECK(after.H == before.H);
    CHECK(after.times == before.times);
    CHECK(after.density == before.density);
    CHECK(after.uncertainty == before.uncertainty);

    fs::remove(path);
}

TEST_CASE("ForecastGridBin: a second save() fully discards the first (single-slot, last-write-wins)") {
    auto path = fs::temp_directory_path() / "rope_fgbin_discard.bin";
    fs::remove(path);

    auto gridA = make_grid(2, 1.0f, 0.1f);
    gridA.times = {1000, 2000};
    ForecastGridBin::save(gridA, path);

    auto gridB = make_grid(4, 9.0f, 0.9f);
    gridB.times = {5000, 6000, 7000, 8000};
    ForecastGridBin::save(gridB, path);

    auto loaded = ForecastGridBin::load(path);
    CHECK(loaded.H == gridB.H);
    CHECK(loaded.times == gridB.times);
    CHECK(loaded.density == gridB.density);
    CHECK(loaded.uncertainty == gridB.uncertainty);

    fs::remove(path);
}
