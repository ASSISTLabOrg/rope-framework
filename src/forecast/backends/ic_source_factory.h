#pragma once
// ic_source_factory.h — construct an IICSource from the manifest's
// top-level ic.kind string. Mirrors model_interface.h's make_model()/
// ModelBackend dispatch pattern (IC-kind instead of model-backend), and
// pipeline_registry.h's create_pipeline_for_kind()/known_kinds() pattern
// one level up (pipeline-kind instead of ic-kind).

#include "ic_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rope::forecast {

// Throws std::runtime_error for an unrecognized ic_kind.
std::unique_ptr<IICSource> make_ic_source(
    const std::filesystem::path& dir,
    const std::string&           ic_kind);

// Inline so Cat A tests can call it without linking rope_forecast.
inline std::vector<std::string> known_ic_kinds() {
    return {"ic_lookup_table"};
}

} // namespace rope::forecast
