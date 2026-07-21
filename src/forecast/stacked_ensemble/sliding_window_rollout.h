#pragma once
#include <algorithm>
#include <cassert>
#include <vector>

#include "rollout_strategy.h"

namespace rope::forecast {

// Sliding-window auto-regressive rollout
class SlidingWindowRollout : public IRolloutStrategy {
public:
    // K = latent_dim, S = seq_len, D = total_dim
    SlidingWindowRollout(int K, int S, int D) : K_(K), S_(S), D_(D) {}

    void run(
        IModel&      model,
        const float* x_chunk,   // (H+1, S, D)
        int          H,
        float*       preds_out  // (H, K)
    ) const override {
        const int window_size = S_ * D_;
        std::vector<float> inp(window_size);
        std::copy(x_chunk, x_chunk + window_size, inp.begin());

        std::vector<float> out_buf(K_);

        const std::vector<int64_t> in_shape  = {1, S_, D_};
        const std::vector<int64_t> out_shape = {1, K_};

        for (int t = 1; t <= H; ++t) {
            if (!model.try_infer_into(inp.data(), in_shape, out_buf.data(), out_shape)) {
                std::vector<int64_t> dummy_shape;
                std::vector<float> p = model.infer(inp, in_shape, dummy_shape);
                assert(static_cast<int>(p.size()) >= K_);
                std::copy(p.begin(), p.begin() + K_, out_buf.begin());
            }

            std::copy(out_buf.begin(), out_buf.end(),
                      preds_out + (t - 1) * K_);

            if (t < H) {
                // Slide the window: inp[0:S-1] = inp[1:S]
                std::copy(inp.begin() + D_, inp.begin() + S_ * D_, inp.begin());
                float* last_row = inp.data() + (S_ - 1) * D_;
                std::copy(out_buf.begin(), out_buf.end(), last_row);

                const float* next_drv = x_chunk
                    + static_cast<size_t>(t) * S_ * D_
                    + static_cast<size_t>(S_ - 1) * D_
                    + K_;
                std::copy(next_drv, next_drv + (D_ - K_), last_row + K_);
            }
        }
    }

private:
    int K_, S_, D_;
};

} // namespace rope::forecast
