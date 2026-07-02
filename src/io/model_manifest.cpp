#include "rope/io/model_manifest.h"
#include "rope/core/types.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

namespace rope::io {

static std::string missing_field(const char* field, const std::string& path) {
    return std::string("ModelManifest::load: missing required field '") +
           field + "' in " + path;
}

ModelManifest ModelManifest::load(const std::filesystem::path& exported_dir) {
    auto path = exported_dir / "model_manifest.json";

    if (!std::filesystem::exists(path))
        throw std::runtime_error(
            "ModelManifest::load: no model_manifest.json in " +
            exported_dir.string());

    std::ifstream f(path);
    if (!f)
        throw std::runtime_error(
            "ModelManifest::load: cannot open " + path.string());

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "ModelManifest::load: malformed JSON in " +
            path.string() + ": " + e.what());
    }

    ModelManifest m;
    const std::string ps = path.string();

    // schema_version
    if (!j.contains("schema_version") || !j["schema_version"].is_number_integer())
        throw std::runtime_error(missing_field("schema_version", ps));
    m.schema_version = j["schema_version"].get<int>();
    if (m.schema_version != 1)
        throw std::runtime_error(
            "ModelManifest::load: unsupported schema_version " +
            std::to_string(m.schema_version) + " in " + ps);

    // kind
    if (!j.contains("kind") || !j["kind"].is_string())
        throw std::runtime_error(missing_field("kind", ps));
    m.kind = j["kind"].get<std::string>();
    if (m.kind != "ensemble_fusion_decoder")
        throw std::runtime_error(
            "ModelManifest::load: unsupported kind '" + m.kind + "' in " + ps);

    // runtime_requirements (object with optional keys)
    if (j.contains("runtime_requirements") && j["runtime_requirements"].is_object()) {
        const auto& rr = j["runtime_requirements"];
        if (rr.contains("onnxruntime") && rr["onnxruntime"].is_string())
            m.runtime_requirements.onnxruntime = rr["onnxruntime"].get<std::string>();
        if (rr.contains("libtorch") && rr["libtorch"].is_string())
            m.runtime_requirements.libtorch = rr["libtorch"].get<std::string>();
    }

    // latent_dim
    if (!j.contains("latent_dim") || !j["latent_dim"].is_number_integer())
        throw std::runtime_error(missing_field("latent_dim", ps));
    m.latent_dim = j["latent_dim"].get<int>();
    if (m.latent_dim <= 0)
        throw std::runtime_error(
            "ModelManifest::load: 'latent_dim' must be positive in " + ps);

    // driver_columns
    if (!j.contains("driver_columns") || !j["driver_columns"].is_array())
        throw std::runtime_error(missing_field("driver_columns", ps));
    m.driver_columns = j["driver_columns"].get<std::vector<std::string>>();
    if (m.driver_columns.empty())
        throw std::runtime_error(
            "ModelManifest::load: 'driver_columns' must not be empty in " + ps);

    // driver_source
    if (!j.contains("driver_source") || !j["driver_source"].is_string())
        throw std::runtime_error(missing_field("driver_source", ps));
    m.driver_source = j["driver_source"].get<std::string>();
    if (m.driver_source.empty())
        throw std::runtime_error(
            "ModelManifest::load: 'driver_source' must not be empty in " + ps);

    // ic_grid_axes
    if (!j.contains("ic_grid_axes") || !j["ic_grid_axes"].is_array())
        throw std::runtime_error(missing_field("ic_grid_axes", ps));
    m.ic_grid_axes = j["ic_grid_axes"].get<std::vector<std::string>>();
    if (m.ic_grid_axes.empty())
        throw std::runtime_error(
            "ModelManifest::load: 'ic_grid_axes' must not be empty in " + ps);

    // Kind-specific block
    if (!j.contains("ensemble_fusion_decoder") || !j["ensemble_fusion_decoder"].is_object())
        throw std::runtime_error(missing_field("ensemble_fusion_decoder", ps));

    const auto& jk = j["ensemble_fusion_decoder"];
    EnsembleFusionDecoderSpec spec;

    if (!jk.contains("seq_len") || !jk["seq_len"].is_number_integer())
        throw std::runtime_error(missing_field("ensemble_fusion_decoder.seq_len", ps));
    spec.seq_len = jk["seq_len"].get<int>();
    if (spec.seq_len <= 0)
        throw std::runtime_error(
            "ModelManifest::load: 'seq_len' must be positive in " + ps);

    if (!jk.contains("decode_batch_size") || !jk["decode_batch_size"].is_number_integer())
        throw std::runtime_error(missing_field("ensemble_fusion_decoder.decode_batch_size", ps));
    spec.decode_batch_size = jk["decode_batch_size"].get<int>();
    if (spec.decode_batch_size <= 0)
        throw std::runtime_error(
            "ModelManifest::load: 'decode_batch_size' must be positive in " + ps);

    // base_models
    if (!jk.contains("base_models") || !jk["base_models"].is_array() ||
            jk["base_models"].empty())
        throw std::runtime_error(missing_field("ensemble_fusion_decoder.base_models", ps));
    for (const auto& jbm : jk["base_models"]) {
        BaseModelSpec bm;
        if (!jbm.contains("file") || !jbm["file"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: base_model entry missing 'file' in " + ps);
        bm.file = jbm["file"].get<std::string>();
        if (!jbm.contains("backend") || !jbm["backend"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: base_model entry missing 'backend' in " + ps);
        bm.backend = jbm["backend"].get<std::string>();
        bm.architecture    = jbm.value("architecture", std::string{});
        bm.inter_op_threads = jbm.value("inter_op_threads", 1);
        spec.base_models.push_back(std::move(bm));
    }

    // meta_model
    if (!jk.contains("meta_model") || !jk["meta_model"].is_object())
        throw std::runtime_error(missing_field("ensemble_fusion_decoder.meta_model", ps));
    {
        const auto& jmm = jk["meta_model"];
        if (!jmm.contains("file") || !jmm["file"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: meta_model missing 'file' in " + ps);
        spec.meta_model.file = jmm["file"].get<std::string>();
        if (!jmm.contains("backend") || !jmm["backend"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: meta_model missing 'backend' in " + ps);
        spec.meta_model.backend = jmm["backend"].get<std::string>();
    }

    // decoders
    if (!jk.contains("decoders") || !jk["decoders"].is_array() || jk["decoders"].empty())
        throw std::runtime_error(missing_field("ensemble_fusion_decoder.decoders", ps));
    for (const auto& jd : jk["decoders"]) {
        DecoderStageSpec ds;
        if (!jd.contains("backends") || !jd["backends"].is_object())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'backends' in " + ps);
        for (const auto& [key, val] : jd["backends"].items()) {
            if (!val.is_string())
                throw std::runtime_error(
                    "ModelManifest::load: decoder backends values must be strings in " + ps);
            ds.backends[key] = val.get<std::string>();
        }
        if (ds.backends.empty())
            throw std::runtime_error(
                "ModelManifest::load: decoder 'backends' must not be empty in " + ps);
        if (!jd.contains("stats") || !jd["stats"].is_string())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'stats' in " + ps);
        ds.stats = jd["stats"].get<std::string>();
        if (!jd.contains("alt_start") || !jd["alt_start"].is_number_integer())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'alt_start' in " + ps);
        ds.alt_start = jd["alt_start"].get<int>();
        if (!jd.contains("alt_end") || !jd["alt_end"].is_number_integer())
            throw std::runtime_error(
                "ModelManifest::load: decoder stage missing 'alt_end' in " + ps);
        ds.alt_end = jd["alt_end"].get<int>();
        spec.decoders.push_back(std::move(ds));
    }

    // Validate decoder altitude ranges: sort by alt_start, verify exact tiling of [0, GRID_ALT).
    std::sort(spec.decoders.begin(), spec.decoders.end(),
        [](const DecoderStageSpec& a, const DecoderStageSpec& b) {
            return a.alt_start < b.alt_start;
        });
    if (spec.decoders.front().alt_start != 0)
        throw std::runtime_error(
            "ModelManifest::load: first decoder must have alt_start=0 in " + ps);
    for (std::size_t i = 1; i < spec.decoders.size(); ++i) {
        if (spec.decoders[i].alt_start != spec.decoders[i - 1].alt_end)
            throw std::runtime_error(
                "ModelManifest::load: decoder altitude ranges have a gap or overlap in " + ps);
    }
    if (spec.decoders.back().alt_end != rope::GRID_ALT)
        throw std::runtime_error(
            "ModelManifest::load: last decoder must have alt_end=" +
            std::to_string(rope::GRID_ALT) + " in " + ps);

    m.ensemble_fusion_decoder = std::move(spec);

    // Cross-check: every backend referenced must have a version in runtime_requirements.
    std::set<std::string> used_backends;
    for (const auto& bm : m.ensemble_fusion_decoder->base_models)
        used_backends.insert(bm.backend);
    used_backends.insert(m.ensemble_fusion_decoder->meta_model.backend);
    for (const auto& ds : m.ensemble_fusion_decoder->decoders)
        for (const auto& [b, _] : ds.backends)
            used_backends.insert(b);

    for (const auto& b : used_backends) {
        if (b == "onnx" && m.runtime_requirements.onnxruntime.empty())
            throw std::runtime_error(
                "ModelManifest::load: backend 'onnx' is used but "
                "runtime_requirements.onnxruntime is not set in " + ps);
        if (b == "libtorch" && m.runtime_requirements.libtorch.empty())
            throw std::runtime_error(
                "ModelManifest::load: backend 'libtorch' is used but "
                "runtime_requirements.libtorch is not set in " + ps);
    }

    return m;
}

} // namespace rope::io
