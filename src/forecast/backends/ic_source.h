#pragma once
// Abstract interface for IC (Initial Condition) loading.

#include <vector>

namespace rope::forecast {

class IICSource {
public:
    virtual ~IICSource() = default;

    // Returns latent_dim() coefficients for (f10, kp).
    virtual std::vector<float> get_latent_coeffs(float f10, float kp) const = 0;

    virtual int latent_dim() const noexcept = 0;
};

} // namespace rope::forecast
