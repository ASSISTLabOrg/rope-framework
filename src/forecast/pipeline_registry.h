#pragma once
#include "rope/forecast/pipeline.h"
#include "rope/io/model_manifest.h"
#include <memory>
#include <string>
#include <vector>

namespace rope::forecast {

// Throws for unrecognized kinds.
std::unique_ptr<Pipeline> create_pipeline_for_kind(
    const Config& cfg, const io::ModelManifest& manifest);

// Inline so Cat A tests can call it without linking rope_forecast.
inline std::vector<std::string> known_kinds() {
    return {"ensemble_fusion_decoder"};
}

} // namespace rope::forecast
