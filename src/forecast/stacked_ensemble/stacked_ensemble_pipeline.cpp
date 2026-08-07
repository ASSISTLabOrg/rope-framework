#include "stacked_ensemble_pipeline.h"

#include "backends/decoder_factory.h"
#include "backends/ic_source_factory.h"
#include "backends/runtime_compat.h"
#include "sliding_window_rollout.h"
#include "unscented_transform.h"

#include "rope/io/driver_bin.h"
#include "rope/io/driver_cache.h"
#include "rope/core/platform.h"
#include "rope/core/text.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef ROPE_USE_OPENMP
#  include <omp.h>
#endif

namespace rope::forecast {

namespace fs = std::filesystem;

static void fill_driver(const io::DriverRow&            row,
                        const std::vector<std::string>& cols,
                        float*                          out) {
    for (int i = 0; i < static_cast<int>(cols.size()); ++i)
        out[i] = row.get(cols[i]);
}

static std::string join(const std::vector<std::string>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += v[i];
    }
    return s + "]";
}

static ModelBackend parse_backend(const std::string& s) {
    if (s == "onnx") return ModelBackend::ONNX;
#ifdef ROPE_USE_LIBTORCH
    if (s == "libtorch") return ModelBackend::LibTorch;
#endif
    throw std::runtime_error(
        "StackedEnsemblePipeline: backend '" + s + "' is not compiled in");
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

StackedEnsemblePipeline::StackedEnsemblePipeline(
    const Config& cfg, const io::ModelManifest& manifest)
{
    log_ = cfg.log ? cfg.log : [](std::string_view){};

    check_runtime_compat(manifest.runtime_requirements);

    const auto&     spec = *manifest.stacked_ensemble;
    const fs::path& dir  = cfg.exported_dir;

    K_            = manifest.latent_dim;
    S_            = spec.seq_len;
    M_            = static_cast<int>(spec.base_models.size());
    compute_uncertainty_ = cfg.compute_uncertainty;
    driver_cols_         = manifest.driver_columns;
    driver_source_       = manifest.driver_source;
    grid_shape_          = manifest.grid;
    manifest_kind_          = manifest.kind;

    io::Stats stats_ts = io::Stats::load(dir / "stats_ts.bin");
    ts_norm_ = std::make_unique<io::FeatureNormalizer>(stats_ts, K_);

    const int D  = ts_norm_->total_dim();
    const int dd = ts_norm_->driver_dim();

    if (dd != static_cast<int>(manifest.driver_columns.size()))
        throw std::runtime_error(
            "StackedEnsemblePipeline: manifest has " +
            std::to_string(manifest.driver_columns.size()) +
            " driver_columns but stats_ts.bin expects driver_dim=" +
            std::to_string(dd));

    load_ic_source(manifest, dir);
    log_("Loading decoder\xe2\x80\xa6");
    decoder_ = make_decoder(dir, manifest, cfg);
    load_sw_db(cfg);
    load_base_models(cfg, manifest, dir);
    load_meta_model(cfg, manifest, dir, D);

    rollout_ = std::make_unique<SlidingWindowRollout>(K_, S_, D);

    log_("Pipeline loaded.  total_dim=" + std::to_string(D) +
         "  driver_dim=" + std::to_string(dd) +
         "  source=" + (driver_source_.empty() ? "(explicit path)" : driver_source_) +
         "  uncertainty=" + (compute_uncertainty_ ? "on" : "off"));
}

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

// True if both name lists match pairwise, ignoring ASCII case.
static bool axis_names_match(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), rope::core::iequals_ascii);
}

void StackedEnsemblePipeline::load_ic_source(
    const io::ModelManifest& manifest, const fs::path& dir)
{
    log_("Loading IC source\xe2\x80\xa6");
    ic_source_ = make_ic_source(dir, manifest.ic_kind);
    if (ic_source_->latent_dim() != K_)
        throw std::runtime_error(
            "StackedEnsemblePipeline: manifest latent_dim=" +
            std::to_string(K_) + " does not match IC source latent_dim=" +
            std::to_string(ic_source_->latent_dim()));
    if (!axis_names_match(ic_source_->axis_names(), manifest.ic_grid_axes))
        throw std::runtime_error(
            "StackedEnsemblePipeline: IC source axes " + join(ic_source_->axis_names()) +
            " do not match manifest ic.params.grid_axes " + join(manifest.ic_grid_axes));
}

void StackedEnsemblePipeline::load_sw_db(const Config& cfg) {
    log_("Loading space-weather database\xe2\x80\xa6");
    fs::path path = cfg.driver_path;
    if (path.empty()) {
        if (driver_source_.empty())
            throw std::runtime_error(
                "StackedEnsemblePipeline: no driver_path set and manifest "
                "has no 'driver_source'; cannot locate driver data.");
        fs::path cache_dir = cfg.cache_dir.empty()
            ? platform::default_cache_dir()
            : cfg.cache_dir;
        io::DriverCacheManager mgr{cache_dir, cfg.cache_max_age_hours};
        path = mgr.get_path(driver_source_);
    }
    sw_db_ = std::make_unique<io::SpaceWeatherDB>(
        io::SpaceWeatherDB::from_file(path));
}

void StackedEnsemblePipeline::load_base_models(
    const Config& cfg, const io::ModelManifest& manifest, const fs::path& dir)
{
    const auto& spec = *manifest.stacked_ensemble;
    log_("Loading " + std::to_string(M_) + " base models\xe2\x80\xa6");
    base_models_.reserve(M_);
    for (const auto& bm : spec.base_models) {
        base_models_.push_back(
            make_model((dir / bm.file).string(),
                       parse_backend(bm.backend),
                       cfg.intra_threads_base,
                       bm.inter_op_threads,
                       false, "cpu"));
    }
}

void StackedEnsemblePipeline::load_meta_model(
    const Config& cfg, const io::ModelManifest& manifest,
    const fs::path& dir, int D)
{
    const auto& spec = *manifest.stacked_ensemble;
    int meta_threads = cfg.intra_threads_meta;
    if (meta_threads <= 0)
        meta_threads = static_cast<int>(std::thread::hardware_concurrency());

    log_("Loading meta model\xe2\x80\xa6");
    meta_model_ = std::make_unique<EnsembleFuser>(
        make_model((dir / spec.meta_model.file).string(),
                   parse_backend(spec.meta_model.backend),
                   meta_threads, /*inter_op_threads=*/1,
                   false, "cpu"),
        K_, S_, D, M_);
}

// ---------------------------------------------------------------------------
// Sequence building (formerly SequenceBuilder)
// ---------------------------------------------------------------------------

std::vector<float> StackedEnsemblePipeline::build_X_init_norm(
    const std::vector<io::DriverRow>& hist_rows) const
{
    const int K  = K_;
    const int D  = ts_norm_->total_dim();
    const int Dd = ts_norm_->driver_dim();

    const auto&        ic_axes = ic_source_->axis_names();
    std::vector<float> axis_vals(ic_axes.size());

    std::vector<float> X(static_cast<size_t>(S_) * D, 0.0f);
    for (int s = 0; s < S_; ++s) {
        const io::DriverRow& row  = hist_rows[s];
        float*               dest = X.data() + s * D;

        for (std::size_t a = 0; a < ic_axes.size(); ++a)
            axis_vals[a] = row.get(ic_axes[a]);
        std::vector<float> coeffs = ic_source_->get_latent_coeffs(axis_vals);
        for (int k = 0; k < K; ++k)
            dest[k] = coeffs[k];

        fill_driver(row, driver_cols_, dest + K);
        ts_norm_->norm_full_inplace(dest);
    }
    (void)Dd;
    return X;
}

std::vector<float> StackedEnsemblePipeline::build_x_chunk(
    const std::vector<float>&        X_init_norm,
    const std::vector<io::DriverRow>& fcast_rows,
    int                              n_windows) const
{
    const int K  = K_;
    const int D  = ts_norm_->total_dim();
    const int Dd = ts_norm_->driver_dim();

    std::vector<float> chunk(static_cast<size_t>(n_windows) * S_ * D, 0.0f);
    std::copy(X_init_norm.begin(), X_init_norm.end(), chunk.begin());

    std::vector<float> drv(Dd);
    for (int t = 1; t < n_windows; ++t) {
        const float* prev = chunk.data() + (t - 1) * S_ * D;
        float*       curr = chunk.data() +  t      * S_ * D;

        std::copy(prev + D, prev + S_ * D, curr);

        float* last_row = curr + (S_ - 1) * D;
        std::fill(last_row, last_row + K, 0.0f);
        fill_driver(fcast_rows[t], driver_cols_, drv.data());
        ts_norm_->norm_driver_inplace(drv.data());
        std::copy(drv.begin(), drv.end(), last_row + K);
    }
    (void)K;
    return chunk;
}

// ---------------------------------------------------------------------------
// run() phases
// ---------------------------------------------------------------------------

std::vector<float> StackedEnsemblePipeline::run_base_rollout(
    const std::vector<float>& x_chunk, int H)
{
    const int K = K_;
    const int M = M_;
    const int T = H;

    std::vector<float> base_latents(static_cast<size_t>(M) * T * K, 0.0f);

#ifdef ROPE_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < M; ++m) {
        float* out = base_latents.data() + m * T * K;
        rollout_->run(*base_models_[m], x_chunk.data(), H, out);
    }
    return base_latents;
}

StackedEnsemblePipeline::MeanLatents
StackedEnsemblePipeline::compute_mean_latents(
    const std::vector<float>& x_chunk,
    const std::vector<float>& base_latents_norm,
    const std::vector<float>& X_init, int H) const
{
    const int K = K_;
    const int S = S_;
    const int D = ts_norm_->total_dim();
    const int T = H;

    FusionResult meta_out = meta_model_->fuse(
        x_chunk.data(), T, base_latents_norm.data());
    auto meta_mean_norm = std::move(meta_out.mean);
    ts_norm_->denorm_latents_block(meta_mean_norm.data(), T);

    std::vector<float> init_lat(K);
    std::copy(X_init.begin() + (S - 1) * D,
              X_init.begin() + (S - 1) * D + K,
              init_lat.begin());
    ts_norm_->denorm_latents_inplace(init_lat.data());

    const int H_lat = H + 1;
    std::vector<float> mu_lat(static_cast<size_t>(H_lat) * K);
    std::copy(init_lat.begin(), init_lat.end(), mu_lat.begin());
    std::copy(meta_mean_norm.begin(), meta_mean_norm.end(), mu_lat.begin() + K);

    return {std::move(mu_lat), std::move(init_lat)};
}

// Rollout+fusion stay whole-horizon (cheap, latent-space); only decode is chunked.
void StackedEnsemblePipeline::run_streaming(
    const std::string& start_iso, int horizon, int chunk_hours,
    const GridChunkSink& sink, const LatentSink& latent_sink)
{
    const int H = horizon;
    const int S = S_;
    if (chunk_hours <= 0) chunk_hours = horizon + 1;

    std::vector<io::DriverRow> all_rows =
        io::DriverWindowBuilder::build(*sw_db_, start_iso, H + 1, S);

    std::vector<io::DriverRow> hist_rows(all_rows.begin(), all_rows.begin() + S);
    std::vector<io::DriverRow> fcast_rows(
        all_rows.begin() + S - 1,
        all_rows.begin() + S - 1 + H + 1);

    auto X_init  = build_X_init_norm(hist_rows);
    auto x_chunk = build_x_chunk(X_init, fcast_rows, H + 1);
    auto base_latents = run_base_rollout(x_chunk, H);
    auto [mu_lat, init_lat] = compute_mean_latents(x_chunk, base_latents, X_init, H);

    if (latent_sink)
        latent_sink(std::span(mu_lat));

    const int H_lat  = H + 1;
    const int voxels = grid_shape_.voxels();

    std::vector<std::int64_t> times(H_lat);
    for (int t = 0; t < H_lat; ++t) times[t] = fcast_rows[t].tp;

    if (compute_uncertainty_) {
        const int M = M_;
        ts_norm_->denorm_latents_block(base_latents.data(), M * H);

        auto ut = ut_sigma_points(mu_lat, base_latents, init_lat, M, K_, H_lat);
        const int N_SIG = ut.N_SIG;

        for (int t_lat = 0; t_lat < H_lat; t_lat += chunk_hours) {
            const int count_lat = std::min(chunk_hours, H_lat - t_lat);
            auto sigma_slice = std::span(ut.sigma_lat)
                .subspan(static_cast<std::size_t>(t_lat) * N_SIG * K_,
                         static_cast<std::size_t>(count_lat) * N_SIG * K_);

            std::vector<float> dens_sigmas =
                decoder_->decode(sigma_slice, count_lat * N_SIG, K_);

            const std::size_t local_voxels = static_cast<std::size_t>(count_lat) * voxels;
            std::vector<float> density_mean(local_voxels, 0.0f);
            for (int lt = 0; lt < count_lat; ++lt)
                for (int sig = 0; sig < N_SIG; ++sig) {
                    const float* src = dens_sigmas.data() +
                                       (static_cast<std::size_t>(lt) * N_SIG + sig) * voxels;
                    float* dst = density_mean.data() + static_cast<std::size_t>(lt) * voxels;
                    const float w = ut.Wm[sig];
                    for (int v = 0; v < voxels; ++v)
                        dst[v] += w * src[v];
                }

            std::vector<float> uncertainty(local_voxels, 0.0f);
            for (int lt = 0; lt < count_lat; ++lt)
                for (int sig = 0; sig < N_SIG; ++sig) {
                    const float* src = dens_sigmas.data() +
                                       (static_cast<std::size_t>(lt) * N_SIG + sig) * voxels;
                    const float* mu  = density_mean.data() +
                                       static_cast<std::size_t>(lt) * voxels;
                    float* dst = uncertainty.data() + static_cast<std::size_t>(lt) * voxels;
                    const float w = ut.Wc[sig];
                    for (int v = 0; v < voxels; ++v) {
                        float d = src[v] - mu[v];
                        dst[v] += w * d * d;
                    }
                }
            for (float& u : uncertainty)
                u = std::sqrt(std::max(u, 0.0f));

            sink(t_lat,
                 std::span(times).subspan(static_cast<std::size_t>(t_lat),
                                          static_cast<std::size_t>(count_lat)),
                 std::span(density_mean),
                 std::span(uncertainty));
        }
    } else {
        for (int t_lat = 0; t_lat < H_lat; t_lat += chunk_hours) {
            const int count_lat = std::min(chunk_hours, H_lat - t_lat);
            auto mu_slice = std::span(mu_lat)
                .subspan(static_cast<std::size_t>(t_lat) * K_, static_cast<std::size_t>(count_lat) * K_);

            std::vector<float> density = decoder_->decode(mu_slice, count_lat, K_);
            std::vector<float> uncertainty(static_cast<std::size_t>(count_lat) * voxels, 0.0f);

            sink(t_lat,
                 std::span(times).subspan(static_cast<std::size_t>(t_lat),
                                          static_cast<std::size_t>(count_lat)),
                 std::span(density),
                 std::span(uncertainty));
        }
    }
}

} // namespace rope::forecast
