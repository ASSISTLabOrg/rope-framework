#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "rope/core/types.h"
#include "rope/io/stats.h"
#include "model_interface.h"

namespace rope::forecast {

class LatentDecoder {
public:
    LatentDecoder(IModel&                     decoder_model,
                  const io::CoaeDenormalizer& coae_denorm,
                  int                         batch_size,
                  int                         n_alt,
                  int                         n_lst,
                  int                         n_lat)
        : model_(decoder_model)
        , denorm_(coae_denorm)
        , batch_size_(batch_size)
        , stage_voxels_(n_lst * n_lat * n_alt)
    {}

    std::vector<float> decode(std::span<const float> latents, int H, int K) {
        assert(static_cast<int>(latents.size()) == H * K);

        std::vector<float> density(static_cast<size_t>(H) * stage_voxels_);

        int offset_in  = 0;
        int offset_out = 0;

        while (offset_in < H) {
            int batch = std::min(batch_size_, H - offset_in);

            std::vector<float>   inp(latents.begin() + offset_in * K,
                                     latents.begin() + (offset_in + batch) * K);
            std::vector<int64_t> in_shape  = {batch, K};
            std::vector<int64_t> out_shape;
            std::vector<float>   raw = model_.infer(inp, in_shape, out_shape);

            if (static_cast<int>(raw.size()) != batch * stage_voxels_)
                throw std::runtime_error(
                    "LatentDecoder: unexpected decoder output size: got " +
                    std::to_string(raw.size()) + " expected " +
                    std::to_string(batch * stage_voxels_));

            denorm_.apply_inplace(raw.data(), batch, stage_voxels_);

            std::copy(raw.begin(), raw.end(),
                      density.begin() +
                      static_cast<size_t>(offset_out) * stage_voxels_);

            offset_in  += batch;
            offset_out += batch;
        }
        return density;
    }

private:
    IModel&                     model_;
    const io::CoaeDenormalizer& denorm_;
    int                         batch_size_;
    int                         stage_voxels_;
};

} // namespace rope::forecast
