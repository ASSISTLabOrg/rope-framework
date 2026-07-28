#pragma once
// Abstract interface for IC (Initial Condition) loading.

#include <span>
#include <string>
#include <vector>

namespace rope::forecast {

class IICSource {
public:
    virtual ~IICSource() = default;

    // Returns latent_dim() coefficients for the given axis values, in the
    // order axis_names() declares. axis_values.size() must equal
    // axis_names().size() (2 today; a std::span keeps this interface stable
    // if a higher-dimensional IC kind is added later).
    virtual std::vector<float> get_latent_coeffs(std::span<const float> axis_values) const = 0;

    // The driver-column names this source is indexed by, in axis order.
    virtual std::vector<std::string> axis_names() const = 0;

    virtual int latent_dim() const noexcept = 0;
};

} // namespace rope::forecast
