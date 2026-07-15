#include <catch2/catch_test_macros.hpp>
#include "rope/io/model_manifest.h"
#include "rope/core/types.h"

#include <filesystem>
#include <fstream>
#include <string>

using rope::io::ModelManifest;
namespace fs = std::filesystem;

// Write json_str to exported_dir/model_manifest.json, creating the dir first.
static fs::path write_manifest(const std::string& json_str, const char* dir_name) {
    auto dir = fs::temp_directory_path() / dir_name;
    fs::create_directories(dir);
    fs::remove(dir / "model_manifest.json");
    std::ofstream f(dir / "model_manifest.json");
    f << json_str;
    return dir;
}

// A minimal valid manifest with 3 base models and one decoder stage.
static const char* VALID_3BM = R"({
  "schema_version": 1,
  "kind": "stacked_ensemble",
  "runtime_requirements": { "onnxruntime": "1.25" },
  "latent_dim": 10,
  "driver_columns": ["f10", "kp", "t1", "t2", "t3", "t4"],
  "driver_source": "celestrak_sw",
  "validated": false,
  "grid": { "n_lst": 72, "n_lat": 36, "n_alt": 45,
            "lat_min_deg": -87.5, "lat_max_deg": 87.5,
            "alt_min_km": 100.0, "alt_max_km": 980.0 },
  "stacked_ensemble": {
    "seq_len": 3,
    "decode_batch_size": 120,
    "base_models": [
      { "file": "base_model_00.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1 },
      { "file": "base_model_01.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1 },
      { "file": "base_model_02.onnx", "backend": "onnx", "architecture": "transformer", "inter_op_threads": 2 }
    ],
    "meta_model": { "file": "meta_model.onnx", "backend": "onnx" },
    "decoders": [
      { "backends": { "onnx": "coae_decoder.onnx" },
        "stats": "stats_cae.bin", "alt_start": 0, "alt_end": 45 }
    ],
    "ic": { "kind": "ic_lookup_table", "params": { "grid_axes": ["f10", "kp"], "file": "ic_table.icbin" } }
  }
})";

// -------------------------------------------------------------------
// Missing manifest
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: missing file throws") {
    auto dir = fs::temp_directory_path() / "rope_mtest_nomatch";
    fs::create_directories(dir);
    fs::remove(dir / "model_manifest.json");
    REQUIRE_THROWS_AS(ModelManifest::load(dir), std::runtime_error);
}

// -------------------------------------------------------------------
// Malformed JSON
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: malformed JSON throws") {
    write_manifest("{ this is not json }", "rope_mtest_badjson");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_badjson"),
        std::runtime_error
    );
}

// -------------------------------------------------------------------
// Missing required envelope fields
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: missing schema_version throws") {
    write_manifest(R"({
      "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_nover");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nover"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: missing latent_dim throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "driver_columns": ["f10"], "driver_source": "s", "validated": false,
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_noldim");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_noldim"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: missing driver_columns throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_source": "s", "validated": false,
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_nodcols");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nodcols"),
        std::runtime_error
    );
}

// -------------------------------------------------------------------
// Unsupported kind
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: unsupported kind throws") {
    write_manifest(R"({
      "schema_version": 1,
      "kind": "future_kind_that_does_not_exist",
      "runtime_requirements": {},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false
    })", "rope_mtest_badkind");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_badkind"),
        std::runtime_error
    );
}

// -------------------------------------------------------------------
// Unsupported schema_version
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: unsupported schema_version throws") {
    write_manifest(R"({
      "schema_version": 99, "kind": "stacked_ensemble",
      "runtime_requirements": {}, "latent_dim": 10,
      "driver_columns": ["f10"], "driver_source": "s", "validated": false
    })", "rope_mtest_badver");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_badver"),
        std::runtime_error
    );
}

// -------------------------------------------------------------------
// Nested ic block (rope-registry shape)
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: missing stacked_ensemble.ic throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}]}
    })", "rope_mtest_noic");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_noic"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: ic.params missing grid_axes throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"file":"ic_table.icbin"}}}
    })", "rope_mtest_icnoaxes");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_icnoaxes"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: valid nested ic block parses") {
    auto dir = write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10","kp"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10","kp"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_icok");
    auto m = ModelManifest::load(dir);
    REQUIRE(m.ic_grid_axes.size() == 2);
    CHECK(m.ic_grid_axes[0] == "f10");
    CHECK(m.ic_grid_axes[1] == "kp");
}

// -------------------------------------------------------------------
// Backend referenced without matching runtime_requirements entry
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: onnx backend without onnxruntime version throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_nort");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nort"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: libtorch backend without libtorch version throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx","libtorch":"d.pt"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_nolt");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nolt"),
        std::runtime_error
    );
}

// -------------------------------------------------------------------
// M != 15: 3 base models parse correctly
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: 3 base models parse with correct count") {
    auto dir = write_manifest(VALID_3BM, "rope_mtest_3bm");
    auto m = ModelManifest::load(dir);
    REQUIRE(m.kind == "stacked_ensemble");
    REQUIRE(m.latent_dim == 10);
    REQUIRE(m.driver_columns.size() == 6);
    REQUIRE(m.stacked_ensemble.has_value());
    CHECK(m.stacked_ensemble->base_models.size() == 3);
    CHECK(m.stacked_ensemble->base_models[2].inter_op_threads == 2);
    CHECK(m.stacked_ensemble->base_models[2].architecture == "transformer");
    CHECK(m.stacked_ensemble->decoders.size() == 1);
}

// -------------------------------------------------------------------
// grid — per-model physical shape
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: missing grid throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_nogrid");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nogrid"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: grid.lat_min_deg >= lat_max_deg throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":87.5,"lat_max_deg":-87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_badlat");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_badlat"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: grid with a different physical shape parses") {
    // A model trained on a coarser, shorter-range grid than the 72x36x45 default.
    auto dir = write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":36,"n_lat":18,"n_alt":20,"lat_min_deg":-60.0,"lat_max_deg":60.0,"alt_min_km":150.0,"alt_max_km":500.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":20}],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_altgrid");
    auto m = ModelManifest::load(dir);
    CHECK(m.grid.n_lst == 36);
    CHECK(m.grid.n_lat == 18);
    CHECK(m.grid.n_alt == 20);
    CHECK(m.grid.voxels() == 36 * 18 * 20);
}

// -------------------------------------------------------------------
// Decoder altitude range validation
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: altitude gap throws") {
    // [0,20) then [22,45) — gap between 20 and 22
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d0.onnx"},"stats":"s0.bin","alt_start":0,"alt_end":20},
          {"backends":{"onnx":"d1.onnx"},"stats":"s1.bin","alt_start":22,"alt_end":45}
        ],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_gap");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_gap"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: altitude overlap throws") {
    // [0,25) then [20,45) — overlap between 20 and 25
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d0.onnx"},"stats":"s0.bin","alt_start":0,"alt_end":25},
          {"backends":{"onnx":"d1.onnx"},"stats":"s1.bin","alt_start":20,"alt_end":45}
        ],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_overlap");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_overlap"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: first alt_start != 0 throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":1,"alt_end":45}
        ],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_start");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_start"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: last alt_end != grid.n_alt throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":44}
        ],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_end");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_end"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: valid two-stage altitude split parses") {
    // [0,22) + [22,45) — exact tiling with libtorch declared
    auto dir = write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25","libtorch":"2.7"},
      "latent_dim": 10, "driver_columns": ["f10","kp"], "driver_source": "celestrak_sw",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"dec_lo.onnx","libtorch":"dec_lo.pt"},
           "stats":"stats_lo.bin","alt_start":0,"alt_end":22},
          {"backends":{"onnx":"dec_hi.onnx","libtorch":"dec_hi.pt"},
           "stats":"stats_hi.bin","alt_start":22,"alt_end":45}
        ],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10","kp"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_2stage");
    auto m = ModelManifest::load(dir);
    REQUIRE(m.stacked_ensemble.has_value());
    const auto& stages = m.stacked_ensemble->decoders;
    REQUIRE(stages.size() == 2);
    CHECK(stages[0].alt_start == 0);
    CHECK(stages[0].alt_end   == 22);
    CHECK(stages[1].alt_start == 22);
    CHECK(stages[1].alt_end   == m.grid.n_alt);
    CHECK(stages[0].stats == "stats_lo.bin");
    CHECK(stages[1].stats == "stats_hi.bin");
}

// -------------------------------------------------------------------
// Decoder ordering: stages given out of order must be sorted
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: out-of-order decoder stages are sorted on load") {
    auto dir = write_manifest(R"({
      "schema_version": 1, "kind": "stacked_ensemble",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "validated": false,
      "grid": {"n_lst":72,"n_lat":36,"n_alt":45,"lat_min_deg":-87.5,"lat_max_deg":87.5,"alt_min_km":100.0,"alt_max_km":980.0},
      "stacked_ensemble": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"dec_hi.onnx"},"stats":"stats_hi.bin","alt_start":22,"alt_end":45},
          {"backends":{"onnx":"dec_lo.onnx"},"stats":"stats_lo.bin","alt_start":0,"alt_end":22}
        ],
        "ic": {"kind":"ic_lookup_table","params":{"grid_axes":["f10"],"file":"ic_table.icbin"}}}
    })", "rope_mtest_order");
    auto m = ModelManifest::load(dir);
    const auto& stages = m.stacked_ensemble->decoders;
    REQUIRE(stages.size() == 2);
    CHECK(stages[0].alt_start == 0);
    CHECK(stages[1].alt_start == 22);
}
