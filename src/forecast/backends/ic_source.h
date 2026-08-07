#pragma once
// Abstract interface for IC (Initial Condition) loading.

#include <span>
#include <string>
#include <vector>

namespace rope::forecast {

class IICSource {
public:
    virtual ~IICSource() = default;
    virtual std::vector<float> get_latent_coeffs(std::span<const float> axis_values) const = 0;

    // The driver-column names this source is indexed by, in axis order.
    virtual std::vector<std::string> axis_names() const = 0;

    virtual int latent_dim() const noexcept = 0;
};

} // namespace rope::forecast
