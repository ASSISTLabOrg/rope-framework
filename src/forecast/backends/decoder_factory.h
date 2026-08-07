#pragma once
// Constructs an IDecoder from manifest.decoder.kind.
//
// Declare-only: no heavy (model/ORT/LibTorch) includes here, only light headers
// (Config, ModelManifest) — rope_tests links this without rope_forecast/ORT/LibTorch,
// same constraint ic_source_factory.h already satisfies. Concrete decoder headers
// (coae_decoder.h) belong in decoder_factory.cpp only.

#include "decoder.h"
#include "rope/forecast/pipeline.h"
#include "rope/io/model_manifest.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rope::forecast {

// Throws std::runtime_error for an unrecognized decoder_kind.
std::unique_ptr<IDecoder> make_decoder(
    const std::filesystem::path& dir,
    const io::ModelManifest&     manifest,
    const Config&                cfg);

inline std::vector<std::string> known_decoder_kinds() {
    return {"coae"};
}

} // namespace rope::forecast
