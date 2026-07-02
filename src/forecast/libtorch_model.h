#pragma once
// libtorch_model.h — TorchScript backend for the forecast pipeline.
//
// Include only in model_factory.cpp (single inclusion point for heavy headers).
// Device strings follow PyTorch conventions: "cpu", "cuda", "cuda:0", etc.

#include <stdexcept>
#include <string>
#include <vector>

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/utils.h>

#include "model_interface.h"

namespace rope::forecast {

class LibTorchModel : public IModel {
public:
    explicit LibTorchModel(const std::string& path,
                           const std::string& device_str = "cpu",
                           int                num_threads = 0)
        : name_(path)
        , device_(device_str)
    {
        if (device_.is_cuda() && !torch::cuda::is_available()) {
            throw std::runtime_error(
                "LibTorchModel: device=\"" + device_str + "\" requested but "
                "torch::cuda::is_available() returned false.\n"
                "  For NVIDIA GPUs: ensure a CUDA-enabled LibTorch build and "
                "a compatible CUDA driver.\n"
                "  For AMD GPUs: ensure a ROCm-enabled LibTorch build and "
                "ROCm drivers are installed.");
        }

        if (num_threads > 0 && device_.is_cpu())
            at::set_num_threads(num_threads);

        module_ = torch::jit::load(path, device_);
        module_.eval();
    }

    std::vector<float> infer(
        const std::vector<float>&   input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>&       output_shape
    ) override {
        torch::NoGradGuard no_grad;

        auto in_cpu = torch::from_blob(
            const_cast<float*>(input.data()),
            input_shape,
            torch::kFloat32);
        auto in_tensor = in_cpu.to(device_);

        auto out = module_.forward({in_tensor})
                          .toTensor()
                          .to(torch::kCPU)
                          .contiguous();

        const auto& sizes = out.sizes();
        output_shape.assign(sizes.begin(), sizes.end());
        const float* ptr = out.data_ptr<float>();
        return std::vector<float>(ptr, ptr + out.numel());
    }

    const std::string& name() const override { return name_; }

private:
    std::string        name_;
    torch::Device      device_;
    torch::jit::Module module_;
};

} // namespace rope::forecast
