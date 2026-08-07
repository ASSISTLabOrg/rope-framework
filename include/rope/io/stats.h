#pragma once
// Z-score statistics and normalizers for the ROPE inference pipeline.
//
// Binary format (stats_ts.bin, stats_cae.bin):
//   uint32  ndim
//   uint32  shape[ndim]
//   float32 mu[product(shape)]
//   float32 sigma[product(shape)]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rope::io {

// Raw statistics blob (mu + sigma over a flat shape).
struct Stats {
    std::vector<std::uint32_t> shape;
    std::vector<float>         mu;
    std::vector<float>         sigma;

    std::size_t numel() const noexcept { return mu.size(); }

    static Stats load(const std::filesystem::path& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Stats::load: cannot open " + path.string());

        Stats s;
        std::uint32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        s.shape.resize(ndim);
        f.read(reinterpret_cast<char*>(s.shape.data()),
               static_cast<std::streamsize>(ndim * 4));

        std::size_t n = 1;
        for (auto d : s.shape) n *= d;
        s.mu.resize(n);
        s.sigma.resize(n);
        f.read(reinterpret_cast<char*>(s.mu.data()),
               static_cast<std::streamsize>(n * 4));
        f.read(reinterpret_cast<char*>(s.sigma.data()),
               static_cast<std::streamsize>(n * 4));
        if (!f)
            throw std::runtime_error("Stats::load: read failed for " + path.string());
        return s;
    }
};

// Z-score normalizer for the time-series feature vector: [latent_0…latent_{K-1}, driver_0…driver_{D-K-1}].
class FeatureNormalizer {
public:
    FeatureNormalizer(const Stats& stats, int latent_dim)
        : K_(latent_dim)
        , D_(static_cast<int>(stats.numel()))
        , mu_(stats.mu)
        , sigma_(stats.sigma)
    {
        if (D_ <= K_)
            throw std::runtime_error(
                "FeatureNormalizer: stats dim must exceed latent_dim");
    }

    int latent_dim() const noexcept { return K_; }
    int total_dim()  const noexcept { return D_; }
    int driver_dim() const noexcept { return D_ - K_; }

    // Normalize a full feature vector of length D (in-place).
    void norm_full_inplace(float* x) const noexcept {
        for (int i = 0; i < D_; ++i)
            x[i] = (x[i] - mu_[i]) / sigma_[i];
    }

    // Normalize only the driver portion (length D-K).
    void norm_driver_inplace(float* drv) const noexcept {
        for (int i = 0; i < D_ - K_; ++i)
            drv[i] = (drv[i] - mu_[K_ + i]) / sigma_[K_ + i];
    }

    // De-normalize a latent vector of length K (in-place).
    void denorm_latents_inplace(float* lat) const noexcept {
        for (int i = 0; i < K_; ++i)
            lat[i] = lat[i] * sigma_[i] + mu_[i];
    }

    // De-normalize a contiguous (N, K) block.
    void denorm_latents_block(float* block, int N) const noexcept {
        for (int n = 0; n < N; ++n)
            denorm_latents_inplace(block + n * K_);
    }

private:
    int                K_, D_;
    std::vector<float> mu_, sigma_;
};

// Converts decoder output (log10 space) to density: 10^(x*sigma+mu). Stats may be scalar or spatial.
class CoaeDenormalizer {
public:
    explicit CoaeDenormalizer(const Stats& stats)
        : mu_(stats.mu), sigma_(stats.sigma), n_(mu_.size()) {}

    // Apply denormalization in-place to a flat (T, 1, 72, 36, 45) block.
    void apply_inplace(float* block, int T, int voxels_per_t) const noexcept {
        for (int t = 0; t < T; ++t) {
            float* ptr = block + static_cast<std::size_t>(t) * voxels_per_t;
            for (int v = 0; v < voxels_per_t; ++v) {
                float mu_v    = (n_ == 1) ? mu_[0]    : mu_[v];
                float sigma_v = (n_ == 1) ? sigma_[0] : sigma_[v];
                ptr[v] = std::pow(10.0f, ptr[v] * sigma_v + mu_v);
            }
        }
    }

private:
    std::vector<float> mu_, sigma_;
    std::size_t        n_;
};

} // namespace rope::io
