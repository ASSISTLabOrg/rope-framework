#pragma once
// Abstract interface for latent-to-grid decoding.

#include <span>
#include <vector>

namespace rope::forecast {

class IDecoder {
public:
    virtual ~IDecoder() = default;

    // Decodes H latent rows (each of dim K) into the full stitched (H, voxels) density grid.
    virtual std::vector<float> decode(std::span<const float> latents, int H, int K) = 0;
};

} // namespace rope::forecast
