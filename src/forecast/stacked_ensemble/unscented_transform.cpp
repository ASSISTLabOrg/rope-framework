#include "unscented_transform.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rope::forecast {

// Cholesky-Banachiewicz in-place; returns false if not SPD.
static bool cholesky_inplace(float* a, int k) {
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = a[i * k + j];
            for (int r = 0; r < j; ++r)
                s -= a[i * k + r] * a[j * k + r];
            if (i == j) {
                if (s <= 0.0f) return false;
                a[i * k + j] = std::sqrt(s);
            } else {
                a[i * k + j] = s / a[j * k + j];
            }
        }
        for (int j = i + 1; j < k; ++j)
            a[i * k + j] = 0.0f;
    }
    return true;
}

UnscentedResult ut_sigma_points(
    const std::vector<float>& mu_lat,
    const std::vector<float>& base_latents,
    const std::vector<float>& init_lat,
    int M, int K, int H_lat)
{
    const float c_ut  = static_cast<float>(K);
    const int   N_SIG = 2 * K + 1;
    const int   T     = H_lat - 1;

    UnscentedResult res;
    res.N_SIG = N_SIG;
    res.Wm.assign(N_SIG, 1.0f / (2.0f * c_ut));
    res.Wc.assign(N_SIG, 1.0f / (2.0f * c_ut));
    res.Wm[0] = 0.0f;
    res.Wc[0] = 2.0f;  // 1 - α² + β

    const size_t sig_stride = static_cast<size_t>(N_SIG) * K;
    res.sigma_lat.assign(static_cast<size_t>(H_lat) * sig_stride, 0.0f);

    std::vector<float> Pt(K * K);
    std::vector<float> cPt(K * K);
    constexpr float EPS_JIT  = 1e-6f;
    constexpr float EPS_JIT2 = 1e-3f;

    for (int t = 0; t < H_lat; ++t) {
        const float* mu_t  = mu_lat.data() + t * K;
        float*       sig_t = res.sigma_lat.data() + t * static_cast<ptrdiff_t>(sig_stride);

        std::copy(mu_t, mu_t + K, sig_t);

        std::fill(Pt.begin(), Pt.end(), 0.0f);
        for (int m = 0; m < M; ++m) {
            const float* x_mt = (t == 0)
                ? init_lat.data()
                : base_latents.data() + m * T * K + (t - 1) * K;
            for (int i = 0; i < K; ++i) {
                float di = x_mt[i] - mu_t[i];
                for (int j = 0; j <= i; ++j)
                    Pt[i * K + j] += di * (x_mt[j] - mu_t[j]);
            }
        }
        const float inv_m = (M > 1) ? 1.0f / static_cast<float>(M - 1) : 0.0f;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j <= i; ++j) {
                float v = Pt[i * K + j] * inv_m;
                Pt[i * K + j] = v;
                Pt[j * K + i] = v;
            }

        for (int idx = 0; idx < K * K; ++idx) cPt[idx] = c_ut * Pt[idx];
        for (int i   = 0; i   < K;     ++i)   cPt[i * K + i] += c_ut * EPS_JIT;

        if (!cholesky_inplace(cPt.data(), K)) {
            // cholesky_inplace overwrote cPt in place before failing; rebuild from Pt.
            for (int idx = 0; idx < K * K; ++idx) cPt[idx] = c_ut * Pt[idx];
            for (int i   = 0; i   < K;     ++i)   cPt[i * K + i] += c_ut * (EPS_JIT + EPS_JIT2);
            if (!cholesky_inplace(cPt.data(), K))
                throw std::runtime_error(
                    "UT: Cholesky failed at t=" + std::to_string(t));
        }

        for (int i = 0; i < K; ++i) {
            float* sp = sig_t + (1 + i)     * K;
            float* sm = sig_t + (1 + K + i) * K;
            for (int j = 0; j < K; ++j) {
                float sji = cPt[j * K + i];
                sp[j] = mu_t[j] + sji;
                sm[j] = mu_t[j] - sji;
            }
        }
    }

    return res;
}

} // namespace rope::forecast
