#pragma once
// lookup_table_ic_source.h — IICSource backed by io::ICTable, for the
// "ic_lookup_table" manifest ic.kind. Pure composition: owns an io::ICTable
// and delegates both interface methods unchanged. ICTable/IcBin themselves
// are not modified by this wrapper.

#include "ic_source.h"
#include "rope/io/ic_table.h"

#include <utility>

namespace rope::forecast {

class LookupTableICSource : public IICSource {
public:
    explicit LookupTableICSource(io::ICTable table) : table_(std::move(table)) {}

    std::vector<float> get_latent_coeffs(float f10, float kp) const override {
        return table_.get_latent_coeffs(f10, kp);
    }

    int latent_dim() const noexcept override { return table_.latent_dim(); }

private:
    io::ICTable table_;
};

} // namespace rope::forecast
