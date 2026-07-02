#include "rope/forecast/pipeline.h"
#include "pipeline_registry.h"
#include "rope/io/model_manifest.h"

namespace rope::forecast {

std::unique_ptr<Pipeline> load(const Config& cfg) {
    io::ModelManifest manifest = io::ModelManifest::load(cfg.exported_dir);
    return create_pipeline_for_kind(cfg, manifest);
}

} // namespace rope::forecast
