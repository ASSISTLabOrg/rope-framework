#include "pipeline_registry.h"
#include "ensemble_fusion_decoder_pipeline.h"
#include <stdexcept>

namespace rope::forecast {

std::unique_ptr<Pipeline> create_pipeline_for_kind(
    const Config& cfg, const io::ModelManifest& manifest)
{
    if (manifest.kind == "ensemble_fusion_decoder")
        return std::make_unique<EnsembleFusionDecoderPipeline>(cfg, manifest);

    throw std::runtime_error(
        "pipeline_registry: unrecognized kind '" + manifest.kind + "'");
}

} // namespace rope::forecast
