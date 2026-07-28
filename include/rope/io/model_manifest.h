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

// One entry of manifest.drivers.columns — a human-readable copy of what a
// driver column means, written at export time (see docs/driver-system.md).
struct DriverColumnInfo {
    std::string name;
    std::string description;
};

struct ModelManifest {
    int         schema_version = 0;
    std::string kind;
    RuntimeRequirements runtime_requirements;

    // Kind-agnostic data-contract fields, parsed from the manifest's top-level
    // "drivers" block ({source, columns: [{name, description}]}).
    int                           latent_dim = 0;
    std::vector<std::string>      driver_columns;      // names only, in feature-vector order
    std::string                   driver_source;
    std::vector<DriverColumnInfo> driver_column_info;  // name + description, same order

    // Whether this model has been validated against a rope-registry
    // validation suite. Informational only — not consumed by the C++ loader
    // beyond parsing it; see rope-registry's manifest-envelope.schema.json.
    bool validated = false;

    // Required — every model declares the grid it was trained/exported on.
    GridSpec grid;

    // manifest.ic.kind. Not validated here — see forecast::make_ic_source().
    std::string ic_kind;

    // manifest.ic.params.grid_axes.
    std::vector<std::string> ic_grid_axes;

    // Present iff kind == "stacked_ensemble".
    std::optional<StackedEnsembleSpec> stacked_ensemble;

    // Throws on missing/malformed file, unsupported schema_version/kind, or invalid fields.
    static ModelManifest load(const std::filesystem::path& exported_dir);

    // Human/machine-readable summary — kind, latent_dim, grid, validated,
    // and the ordered driver column list with descriptions. Shared by the
    // C API (rope_get_manifest_info) and the CLI (`rope manifest`), so both
    // report identical information.
    std::string to_summary_json() const;
};

} // namespace rope::io
