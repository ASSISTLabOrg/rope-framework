#pragma once
#include "rope/core/types.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rope::io {

struct BaseModelSpec {
    std::string file;
    std::string backend;       // "onnx" or "libtorch"
    std::string architecture;  // metadata/logging only
    int inter_op_threads = 1;
};

struct MetaModelSpec {
    std::string file;
    std::string backend;
};

struct DecoderStageSpec {
    std::map<std::string, std::string> backends;  // "onnx"/"libtorch" -> filename
    std::string stats;
    int alt_start = 0;
    int alt_end   = 0;
};

struct StackedEnsembleSpec {
    int seq_len           = 0;
    int decode_batch_size = 0;
    std::vector<BaseModelSpec>    base_models;
    MetaModelSpec                 meta_model;
    std::vector<DecoderStageSpec> decoders;
};

struct RuntimeRequirements {
    std::string onnxruntime;  // major.minor, empty if not required by this manifest
    std::string libtorch;     // major.minor, empty if not required by this manifest
};

struct ModelManifest {
    int         schema_version = 0;
    std::string kind;
    RuntimeRequirements runtime_requirements;

    // Kind-agnostic data-contract fields.
    int                      latent_dim = 0;
    std::vector<std::string> driver_columns;
    std::string              driver_source;

    // Physical shape of this model's output grid. Required — every model
    // declares the grid it was trained/exported on.
    GridSpec grid;

    // manifest.ic.kind. Not validated here — see forecast::make_ic_source().
    std::string ic_kind;

    // manifest.ic.params.grid_axes.
    std::vector<std::string> ic_grid_axes;

    // Present iff kind == "stacked_ensemble".
    std::optional<StackedEnsembleSpec> stacked_ensemble;

    // Throws on: missing file, malformed JSON, missing required fields,
    // unsupported schema_version/kind, backend referenced without a matching
    // runtime_requirements entry, or invalid decoder altitude ranges.
    static ModelManifest load(const std::filesystem::path& exported_dir);
};

} // namespace rope::io
