#pragma once
// model_interface.h — abstract model interface for the forecast pipeline.
//
// Backend-agnostic.  All pipeline code interacts only with IModel.
// Concrete backends: OnnxModel (onnx_model.h), LibTorchModel (libtorch_model.h).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rope::forecast {

// ---------------------------------------------------------------------------
// Pure virtual model interface.
// ---------------------------------------------------------------------------
class IModel {
public:
    virtual ~IModel() = default;

    // Single forward pass.
    //   input        — flat float32 row-major data
    //   input_shape  — shape of the input tensor
    //   output_shape — filled with the output shape on return
    // Returns flat float32 output row-major.
    virtual std::vector<float> infer(
        const std::vector<float>&   input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>&       output_shape
    ) = 0;

    // Zero-copy IoBinding path: write directly into caller-supplied buffers.
    // Returns true if the backend supports this path; false means the caller
    // must fall back to infer().  Default: false (unsupported).
    virtual bool try_infer_into(
        float*                      input_buf,
        const std::vector<int64_t>& input_shape,
        float*                      output_buf,
        const std::vector<int64_t>& output_shape
    ) { (void)input_buf; (void)input_shape; (void)output_buf; (void)output_shape; return false; }

    virtual const std::string& name() const = 0;
};

// ---------------------------------------------------------------------------
// Backend tag used by make_model().
// ---------------------------------------------------------------------------
enum class ModelBackend {
    ONNX,
#ifdef ROPE_USE_LIBTORCH
    LibTorch,
#endif
};

// ---------------------------------------------------------------------------
// Factory declaration — implemented in model_factory.cpp.
// ---------------------------------------------------------------------------
std::unique_ptr<IModel> make_model(
    const std::string& path,
    ModelBackend       backend            = ModelBackend::ONNX,
    int                intra_op_threads   = 1,
    int                inter_op_threads   = 1,
    bool               use_dnnl           = false,
    const std::string& device             = "cpu"
);

} // namespace rope::forecast
