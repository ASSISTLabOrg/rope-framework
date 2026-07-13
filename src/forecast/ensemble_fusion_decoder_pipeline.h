#pragma once

#include "rope/forecast/pipeline.h"
#include "rope/io/model_manifest.h"
#include "rope/io/driver_db.h"
#include "rope/io/ic_table.h"
#include "rope/io/stats.h"

#include "model_interface.h"
#include "ensemble_fuser.h"
#include "latent_decoder.h"
#include "rollout_strategy.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rope::forecast {

class EnsembleFusionDecoderPipeline : public Pipeline {
public:
    EnsembleFusionDecoderPipeline(const Config& cfg, const io::ModelManifest& manifest);
    ForecastGrid run(const std::string& start_iso, int horizon) override;

private:
    // --- Scalars / config ---
    int  K_, S_, M_, DECODE_BATCH_;
    bool compute_uncertainty_{true};

    std::vector<std::string>               driver_cols_;
    std::string                            driver_source_;
    std::function<void(std::string_view)>  log_;

    // --- Data sources ---
    std::unique_ptr<io::SpaceWeatherDB>    sw_db_;
    std::unique_ptr<io::ICTable>           ic_table_;
    std::unique_ptr<io::FeatureNormalizer> ts_norm_;

    // --- Models ---
    std::vector<std::unique_ptr<IModel>>   base_models_;
    std::unique_ptr<EnsembleFuser>         meta_model_;

    struct DecoderStage {
        std::unique_ptr<IModel>              model;
        std::unique_ptr<io::CAEDenormalizer> denorm;
        std::unique_ptr<LatentDecoder>       decoder;
        int alt_start;
        int alt_end;
    };
    std::vector<DecoderStage>              decoder_stages_;
    std::unique_ptr<IRolloutStrategy>      rollout_;

    // --- Constructor helpers ---
    void load_ic_table(const std::filesystem::path& dir);
    void load_sw_db(const Config& cfg);
    void load_base_models(const Config& cfg, const io::ModelManifest& manifest,
                           const std::filesystem::path& dir);
    void load_meta_model(const Config& cfg, const io::ModelManifest& manifest,
                          const std::filesystem::path& dir, int D);
    void load_decoder_stages(const Config& cfg, const io::ModelManifest& manifest,
                              const std::filesystem::path& dir);

    // --- Sequence building (formerly SequenceBuilder) ---
    std::vector<float> build_X_init_norm(
        const std::vector<io::DriverRow>& hist_rows) const;
    std::vector<float> build_x_chunk(
        const std::vector<float>& X_init_norm,
        const std::vector<io::DriverRow>& fcast_rows, int n_windows) const;

    // --- run() phases ---
    std::vector<float> run_base_rollout(
        const std::vector<float>& x_chunk, int H);

    struct MeanLatents {
        std::vector<float> mu_lat;    // (H_lat, K) — IC prepended, denormed
        std::vector<float> init_lat;  // (K) — IC latent at t=0, denormed
    };
    MeanLatents compute_mean_latents(
        const std::vector<float>& x_chunk,
        const std::vector<float>& base_latents_norm,
        const std::vector<float>& X_init, int H) const;

    struct GridArrays {
        std::vector<float> density;
        std::vector<float> uncertainty;
    };
    GridArrays decode_direct(
        const std::vector<float>& mu_lat, int H_lat);
    GridArrays decode_with_uncertainty(
        const std::vector<float>& mu_lat,
        std::vector<float>        base_latents_norm,  // by value — denormed in-place
        const std::vector<float>& init_lat, int H_lat);
};

} // namespace rope::forecast
