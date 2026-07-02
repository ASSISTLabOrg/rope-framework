#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_interface.h"

namespace rope::forecast {

struct FusionResult {
    std::vector<float> mean;  // (T*K)
    std::vector<float> std;   // (T*K)
};

class EnsembleFuser {
public:
    EnsembleFuser(std::unique_ptr<IModel> inner,
                  int K, int S, int D, int M,
                  bool coeff_level = true)
        : inner_(std::move(inner))
        , K_(K), S_(S), D_(D), M_(M)
        , coeff_level_(coeff_level)
    {}

    // x_chunk_slice: (T, S, D) flat — the feature sequence slice for time T.
    // all_preds:     (M, T, K) flat — base-model latent predictions.
    FusionResult fuse(const float* x_chunk_slice, int T,
                      const float* all_preds) const {
        std::vector<float>   inner_in(x_chunk_slice,
                                      x_chunk_slice + static_cast<size_t>(T) * S_ * D_);
        const std::vector<int64_t> inner_in_shape = {T, S_, D_};
        std::vector<int64_t>       inner_out_shape;
        std::vector<float> W = inner_->infer(inner_in, inner_in_shape, inner_out_shape);

        bool coeff_mode = coeff_level_;
        if (inner_out_shape.size() == 3) {
            coeff_mode = true;
            if (inner_out_shape[1] != M_ || inner_out_shape[2] != K_)
                throw std::runtime_error(
                    "EnsembleFuser: inner output (3-D) shape mismatch: "
                    "M=" + std::to_string(M_) + " K=" + std::to_string(K_));
        } else if (inner_out_shape.size() == 2) {
            coeff_mode = false;
            if (inner_out_shape[1] != M_)
                throw std::runtime_error(
                    "EnsembleFuser: inner output (2-D) shape mismatch: "
                    "M=" + std::to_string(M_));
        } else if (inner_out_shape.size() == 1) {
            const int64_t n = static_cast<int64_t>(W.size());
            if      (n == (int64_t)T * M_ * K_) coeff_mode = true;
            else if (n == (int64_t)T * M_)       coeff_mode = false;
            else throw std::runtime_error(
                "EnsembleFuser: cannot infer fusion mode from flat output size "
                + std::to_string(n));
        }

        return _fuse(W, all_preds, T, coeff_mode);
    }

private:
    std::unique_ptr<IModel> inner_;
    int  K_, S_, D_, M_;
    bool coeff_level_;

    FusionResult _fuse(const std::vector<float>& W,
                       const float* all_preds,
                       int T,
                       bool coeff_mode) const {
        FusionResult r;
        r.mean.assign(T * K_, 0.0f);
        r.std.assign (T * K_, 0.0f);

        if (coeff_mode) {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m)
                    for (int k = 0; k < K_; ++k)
                        r.mean[t * K_ + k] +=
                            W[(t * M_ + m) * K_ + k] *
                            all_preds[(m * T + t) * K_ + k];
        } else {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m) {
                    const float w = W[t * M_ + m];
                    for (int k = 0; k < K_; ++k)
                        r.mean[t * K_ + k] +=
                            w * all_preds[(m * T + t) * K_ + k];
                }
        }

        if (coeff_mode) {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m)
                    for (int k = 0; k < K_; ++k) {
                        const float w    = W[(t * M_ + m) * K_ + k];
                        const float diff = all_preds[(m * T + t) * K_ + k]
                                           - r.mean[t * K_ + k];
                        r.std[t * K_ + k] += w * diff * diff;
                    }
        } else {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m) {
                    const float w = W[t * M_ + m];
                    for (int k = 0; k < K_; ++k) {
                        const float diff = all_preds[(m * T + t) * K_ + k]
                                           - r.mean[t * K_ + k];
                        r.std[t * K_ + k] += w * diff * diff;
                    }
                }
        }
        for (float& v : r.std) v = std::sqrt(v);

        return r;
    }
};

} // namespace rope::forecast
