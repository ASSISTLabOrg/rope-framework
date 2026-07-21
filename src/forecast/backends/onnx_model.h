#pragma once
// onnx_model.h — ONNX Runtime backend for the forecast pipeline.

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#ifdef ROPE_USE_DNNL
#  include <dnnl_provider_options.h>
#endif

#include "model_interface.h"

namespace rope::forecast {

namespace detail {
inline Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "rope_forecast");
    return env;
}
inline Ort::MemoryInfo& cpu_mem_info() {
    static Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    return mi;
}
} // namespace detail

class OnnxModel : public IModel {
public:
    explicit OnnxModel(const std::string& path,
                       int intra_op_threads = 1,
                       int inter_op_threads = 1,
                       bool use_dnnl       = false)
        : name_(path)
    {
        Ort::SessionOptions opts;
        opts.SetLogSeverityLevel(3);  // suppress WARNING-level noise (e.g. dynamic shape hints)
        opts.SetIntraOpNumThreads(intra_op_threads);
        opts.SetInterOpNumThreads(inter_op_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef ROPE_USE_DNNL
        if (use_dnnl) {
            OrtDnnlProviderOptions dnnl_opts{};
            dnnl_opts.use_arena = 1;
            try {
                opts.AppendExecutionProvider_Dnnl(dnnl_opts);
            } catch (const Ort::Exception& e) {
                std::cout << "  [oneDNN EP unavailable, falling back to CPU: "
                          << e.what() << "]\n";
            }
        }
#else
        (void)use_dnnl;
#endif

        session_ = std::make_unique<Ort::Session>(
            detail::ort_env(), std::filesystem::path(path).c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        size_t n_in  = session_->GetInputCount();
        size_t n_out = session_->GetOutputCount();

        if (n_in != 1)
            throw std::runtime_error(
                "OnnxModel: expected 1 input, got " + std::to_string(n_in) +
                " for model: " + path);

        input_names_storage_.resize(n_in);
        output_names_storage_.resize(n_out);
        input_names_.resize(n_in);
        output_names_.resize(n_out);

        for (size_t i = 0; i < n_in; ++i) {
            input_names_storage_[i] =
                session_->GetInputNameAllocated(i, alloc).get();
            input_names_[i] = input_names_storage_[i].c_str();
        }
        for (size_t i = 0; i < n_out; ++i) {
            output_names_storage_[i] =
                session_->GetOutputNameAllocated(i, alloc).get();
            output_names_[i] = output_names_storage_[i].c_str();
        }
    }

    // Generic inference — allocates output each call.
    std::vector<float> infer(
        const std::vector<float>&   input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>&       output_shape
    ) override {
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            detail::cpu_mem_info(),
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size());

        auto out_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &in_tensor, 1,
            output_names_.data(), output_names_.size());

        auto& out = out_tensors[0];
        auto  ti  = out.GetTensorTypeAndShapeInfo();
        output_shape = ti.GetShape();
        size_t n = ti.GetElementCount();
        const float* ptr = out.GetTensorData<float>();
        return std::vector<float>(ptr, ptr + n);
    }

    // Zero-copy IoBinding path — overrides IModel::try_infer_into().
    bool try_infer_into(
        float*                      input_buf,
        const std::vector<int64_t>& input_shape,
        float*                      output_buf,
        const std::vector<int64_t>& output_shape
    ) override {
        if (!binding_)
            binding_ = std::make_unique<Ort::IoBinding>(*session_);

        Ort::Value in_val = Ort::Value::CreateTensor<float>(
            detail::cpu_mem_info(),
            input_buf, numel(input_shape),
            input_shape.data(), input_shape.size());
        Ort::Value out_val = Ort::Value::CreateTensor<float>(
            detail::cpu_mem_info(),
            output_buf, numel(output_shape),
            output_shape.data(), output_shape.size());

        binding_->BindInput(input_names_[0], in_val);
        binding_->BindOutput(output_names_[0], out_val);
        session_->Run(Ort::RunOptions{nullptr}, *binding_);
        return true;
    }

    const std::string& name() const override { return name_; }

private:
    static size_t numel(const std::vector<int64_t>& shape) {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    std::string name_;
    std::unique_ptr<Ort::Session>   session_;
    std::unique_ptr<Ort::IoBinding> binding_;

    std::vector<std::string>  input_names_storage_;
    std::vector<std::string>  output_names_storage_;
    std::vector<const char*>  input_names_;
    std::vector<const char*>  output_names_;
};

} // namespace rope::forecast
