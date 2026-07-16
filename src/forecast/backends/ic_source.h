#pragma once
// ic_source.h — abstract interface for IC (Initial Condition) loading.
//
// An IC source maps (F10, Kp) driver values to a K-dimensional latent
// initial-condition vector, used to seed a forecast's t=0 latent state
// (mu_lat[0]). Concrete sources: LookupTableICSource
// (lookup_table_ic_source.h), constructed via make_ic_source()
// (ic_source_factory.h).

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
