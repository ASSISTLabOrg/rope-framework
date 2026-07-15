#pragma once
#include <vector>

namespace rope::forecast {

struct UnscentedResult {
    std::vector<float> sigma_lat;  // (H_lat * N_SIG * K), N_SIG = 2K+1
    std::vector<float> Wm;         // (N_SIG) — mean weights
    std::vector<float> Wc;         // (N_SIG) — covariance weights
    int N_SIG;
};

// Compute sigma points for the Unscented Transform (α=1, β=2, κ=0 → λ=0, c=K).
// mu_lat:       (H_lat, K) — mean latent trajectory
// base_latents: (M, T, K)  — denormed base-model predictions (T = H_lat − 1)
// init_lat:     (K)        — IC latent at t=0
UnscentedResult ut_sigma_points(
    const std::vector<float>& mu_lat,
    const std::vector<float>& base_latents,
    const std::vector<float>& init_lat,
    int M, int K, int H_lat
);

} // namespace rope::forecast
