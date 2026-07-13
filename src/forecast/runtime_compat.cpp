#include "runtime_compat.h"

#include <onnxruntime_cxx_api.h>
#ifdef ROPE_USE_LIBTORCH
#  include <torch/version.h>
#endif

#include <stdexcept>
#include <string>

namespace rope::forecast {

// Extract "major.minor" from a version string like "1.25.0".
static std::string major_minor(const std::string& v) {
    auto first = v.find('.');
    if (first == std::string::npos) return v;
    auto second = v.find('.', first + 1);
    if (second == std::string::npos) return v;
    return v.substr(0, second);
}

void check_runtime_compat(const io::RuntimeRequirements& reqs) {
    if (!reqs.onnxruntime.empty()) {
        const std::string linked   = major_minor(Ort::GetVersionString());
        const std::string required = reqs.onnxruntime;
        if (linked != required)
            throw std::runtime_error(
                "ONNX Runtime version mismatch: manifest requires " + required +
                ", linked runtime is " + linked);
    }

#ifdef ROPE_USE_LIBTORCH
    if (!reqs.libtorch.empty()) {
        const std::string linked   = std::to_string(TORCH_VERSION_MAJOR) + "." +
                                     std::to_string(TORCH_VERSION_MINOR);
        const std::string required = reqs.libtorch;
        if (linked != required)
            throw std::runtime_error(
                "LibTorch version mismatch: manifest requires " + required +
                ", linked runtime is " + linked);
    }
#endif
}

} // namespace rope::forecast
