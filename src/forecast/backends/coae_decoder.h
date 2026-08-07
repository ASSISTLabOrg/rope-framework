#pragma once
// IDecoder backed by one or more COAE (convolutional autoencoder) altitude-stage
// models (manifest decoder.kind == "coae").

#include "decoder.h"
#include "latent_decoder.h"
#include "model_interface.h"
#include "rope/forecast/pipeline.h"
#include "rope/io/model_manifest.h"
#include "rope/io/stats.h"

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace rope::forecast {

class CoaeDecoder : public IDecoder {
public:
    CoaeDecoder(const std::filesystem::path& dir,
                const io::ModelManifest&     manifest,
                const Config&                cfg);

    std::vector<float> decode(std::span<const float> latents, int H, int K) override;

private:
    struct Stage {
        std::unique_ptr<IModel>               model;
        std::unique_ptr<io::CoaeDenormalizer>  denorm;
        std::unique_ptr<LatentDecoder>         decoder;
        int alt_start;
        int alt_end;
    };
    std::vector<Stage> stages_;
    GridSpec            grid_shape_;
};

} // namespace rope::forecast
