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
  "kind": "ensemble_fusion_decoder",
  "runtime_requirements": { "onnxruntime": "1.25" },
  "latent_dim": 10,
  "driver_columns": ["f10", "kp", "t1", "t2", "t3", "t4"],
  "driver_source": "celestrak_sw",
  "ic_grid_axes": ["f10", "kp"],
  "ensemble_fusion_decoder": {
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
    ]
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
      "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}]}
    })", "rope_mtest_nover");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nover"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: missing latent_dim throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "driver_columns": ["f10"], "driver_source": "s", "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}]}
    })", "rope_mtest_noldim");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_noldim"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: missing driver_columns throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_source": "s", "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}]}
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
      "ic_grid_axes": ["f10"]
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
      "schema_version": 99, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {}, "latent_dim": 10,
      "driver_columns": ["f10"], "driver_source": "s", "ic_grid_axes": ["f10"]
    })", "rope_mtest_badver");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_badver"),
        std::runtime_error
    );
}

// -------------------------------------------------------------------
// Backend referenced without matching runtime_requirements entry
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: onnx backend without onnxruntime version throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":45}]}
    })", "rope_mtest_nort");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_nort"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: libtorch backend without libtorch version throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[{"backends":{"onnx":"d.onnx","libtorch":"d.pt"},"stats":"s.bin","alt_start":0,"alt_end":45}]}
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
    REQUIRE(m.kind == "ensemble_fusion_decoder");
    REQUIRE(m.latent_dim == 10);
    REQUIRE(m.driver_columns.size() == 6);
    REQUIRE(m.ensemble_fusion_decoder.has_value());
    CHECK(m.ensemble_fusion_decoder->base_models.size() == 3);
    CHECK(m.ensemble_fusion_decoder->base_models[2].inter_op_threads == 2);
    CHECK(m.ensemble_fusion_decoder->base_models[2].architecture == "transformer");
    CHECK(m.ensemble_fusion_decoder->decoders.size() == 1);
}

// -------------------------------------------------------------------
// Decoder altitude range validation
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: altitude gap throws") {
    // [0,20) then [22,45) — gap between 20 and 22
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d0.onnx"},"stats":"s0.bin","alt_start":0,"alt_end":20},
          {"backends":{"onnx":"d1.onnx"},"stats":"s1.bin","alt_start":22,"alt_end":45}
        ]}
    })", "rope_mtest_gap");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_gap"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: altitude overlap throws") {
    // [0,25) then [20,45) — overlap between 20 and 25
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d0.onnx"},"stats":"s0.bin","alt_start":0,"alt_end":25},
          {"backends":{"onnx":"d1.onnx"},"stats":"s1.bin","alt_start":20,"alt_end":45}
        ]}
    })", "rope_mtest_overlap");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_overlap"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: first alt_start != 0 throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":1,"alt_end":45}
        ]}
    })", "rope_mtest_start");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_start"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: last alt_end != GRID_ALT throws") {
    write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"d.onnx"},"stats":"s.bin","alt_start":0,"alt_end":44}
        ]}
    })", "rope_mtest_end");
    REQUIRE_THROWS_AS(
        ModelManifest::load(fs::temp_directory_path() / "rope_mtest_end"),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest: valid two-stage altitude split parses") {
    // [0,22) + [22,45) — exact tiling with libtorch declared
    auto dir = write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25","libtorch":"2.7"},
      "latent_dim": 10, "driver_columns": ["f10","kp"], "driver_source": "celestrak_sw",
      "ic_grid_axes": ["f10","kp"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"dec_lo.onnx","libtorch":"dec_lo.pt"},
           "stats":"stats_lo.bin","alt_start":0,"alt_end":22},
          {"backends":{"onnx":"dec_hi.onnx","libtorch":"dec_hi.pt"},
           "stats":"stats_hi.bin","alt_start":22,"alt_end":45}
        ]}
    })", "rope_mtest_2stage");
    auto m = ModelManifest::load(dir);
    REQUIRE(m.ensemble_fusion_decoder.has_value());
    const auto& stages = m.ensemble_fusion_decoder->decoders;
    REQUIRE(stages.size() == 2);
    CHECK(stages[0].alt_start == 0);
    CHECK(stages[0].alt_end   == 22);
    CHECK(stages[1].alt_start == 22);
    CHECK(stages[1].alt_end   == rope::GRID_ALT);
    CHECK(stages[0].stats == "stats_lo.bin");
    CHECK(stages[1].stats == "stats_hi.bin");
}

// -------------------------------------------------------------------
// Decoder ordering: stages given out of order must be sorted
// -------------------------------------------------------------------
TEST_CASE("ModelManifest: out-of-order decoder stages are sorted on load") {
    auto dir = write_manifest(R"({
      "schema_version": 1, "kind": "ensemble_fusion_decoder",
      "runtime_requirements": {"onnxruntime":"1.25"},
      "latent_dim": 10, "driver_columns": ["f10"], "driver_source": "s",
      "ic_grid_axes": ["f10"],
      "ensemble_fusion_decoder": {"seq_len":3,"decode_batch_size":120,
        "base_models":[{"file":"a.onnx","backend":"onnx","architecture":"lstm","inter_op_threads":1}],
        "meta_model":{"file":"m.onnx","backend":"onnx"},
        "decoders":[
          {"backends":{"onnx":"dec_hi.onnx"},"stats":"stats_hi.bin","alt_start":22,"alt_end":45},
          {"backends":{"onnx":"dec_lo.onnx"},"stats":"stats_lo.bin","alt_start":0,"alt_end":22}
        ]}
    })", "rope_mtest_order");
    auto m = ModelManifest::load(dir);
    const auto& stages = m.ensemble_fusion_decoder->decoders;
    REQUIRE(stages.size() == 2);
    CHECK(stages[0].alt_start == 0);
    CHECK(stages[1].alt_start == 22);
}
