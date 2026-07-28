#pragma once
// IICSource backed by io::ICTable (manifest ic.kind == "ic_lookup_table").

#include "ic_source.h"
#include "rope/io/ic_table.h"

#include <span>
#include <utility>

namespace rope::forecast {

class LookupTableICSource : public IICSource {
public:
    explicit LookupTableICSource(io::ICTable table) : table_(std::move(table)) {}

    std::vector<float> get_latent_coeffs(std::span<const float> axis_values) const override {
        return table_.get_latent_coeffs(axis_values);
    }

    std::vector<std::string> axis_names() const override { return table_.axis_names(); }

    int latent_dim() const noexcept override { return table_.latent_dim(); }

private:
    io::ICTable table_;
};

} // namespace rope::forecast
