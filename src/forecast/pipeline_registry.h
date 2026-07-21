#pragma once
#include "rope/forecast/pipeline.h"
#include "rope/io/model_manifest.h"
#include <memory>
#include <string>
#include <vector>

namespace rope::forecast {

std::unique_ptr<Pipeline> create_pipeline_for_kind(
    const Config& cfg, const io::ModelManifest& manifest);

inline std::vector<std::string> known_kinds() {
    return {"stacked_ensemble"};
}

} // namespace rope::forecast
