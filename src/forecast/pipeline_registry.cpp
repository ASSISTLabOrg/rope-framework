#include "pipeline_registry.h"
#include "stacked_ensemble/stacked_ensemble_pipeline.h"
#include <stdexcept>

namespace rope::forecast {

std::unique_ptr<Pipeline> create_pipeline_for_kind(
    const Config& cfg, const io::ModelManifest& manifest)
{
    if (manifest.kind == "stacked_ensemble")
        return std::make_unique<StackedEnsemblePipeline>(cfg, manifest);

    throw std::runtime_error(
        "pipeline_registry: unrecognized kind '" + manifest.kind + "'");
}

} // namespace rope::forecast
