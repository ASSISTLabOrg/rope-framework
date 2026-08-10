#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/interpolate/grid_interpolator.h"
#include "rope/core/types.h"
#include "rope/core/datetime.h"
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace Catch::Matchers;
using rope::ForecastGrid;
using rope::GridSpec;
using rope::interpolate::ExtrapolationOptions;
using rope::interpolate::GridInterpolator;
using rope::interpolate::SpatialOutOfRangeError;
using rope::interpolate::TimeOutOfRangeError;

// Matches the shape used throughout these tests: 72 LST x 36 lat x 45 alt,
// covering the same physical range the hardcoded constants used to assume.
static GridSpec test_shape() {
    GridSpec s;
    s.n_lst = 72; s.n_lat = 36; s.n_alt = 45;
    s.lat_min_deg = -87.5; s.lat_max_deg = 87.5;
    s.alt_min_km  = 100.0; s.alt_max_km  = 980.0;
    return s;
}

// Epoch-hour helper
static rope::TimePoint epoch_h(int h) {
    return rope::parse_datetime("2024-01-01T00:00:00") + static_cast<rope::TimePoint>(h) * 3600;
}

// Minimal 2-step grid, all density=1.0, uncertainty=0.5 at every voxel.
// log10(1.0)=0 → interpolation returns 10^0=1.0 everywhere.
static ForecastGrid make_uniform_grid(float den = 1.0f, float unc = 0.5f) {
    ForecastGrid g;
    g.shape = test_shape();
    g.H = 2;
    g.density.assign(static_cast<std::size_t>(2) * g.shape.voxels(), den);
    g.uncertainty.assign(static_cast<std::size_t>(2) * g.shape.voxels(), unc);
    g.times = {epoch_h(0), epoch_h(1)};
    return g;
}

TEST_CASE("GridInterpolator: construction and metadata") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    CHECK(gi.time_min() == epoch_h(0));
    CHECK(gi.time_max() == epoch_h(1));
}

TEST_CASE("GridInterpolator: empty grid throws on construction") {
    ForecastGrid empty;
    REQUIRE_THROWS_AS(GridInterpolator{empty}, std::runtime_error);
}

TEST_CASE("GridInterpolator: query_interp returns positive density and uncertainty") {
    auto grid = make_uniform_grid(1.0f, 0.5f);
    GridInterpolator gi(grid);

    auto r = gi.query_interp(epoch_h(0), 12.0, 0.0, 400.0);
    CHECK(r.density     > 0.0);
    CHECK(r.uncertainty > 0.0);
}

TEST_CASE("GridInterpolator: query_hold returns positive density and uncertainty") {
    auto grid = make_uniform_grid(1.0f, 0.5f);
    GridInterpolator gi(grid);

    auto r = gi.query_hold(epoch_h(0), 12.0, 0.0, 400.0);
    CHECK(r.density     > 0.0);
    CHECK(r.uncertainty > 0.0);
}

TEST_CASE("GridInterpolator: uniform grid interpolates to constant") {
    auto grid = make_uniform_grid(1.0f, 0.5f);
    GridInterpolator gi(grid);

    // With all voxels = 1.0, spatial_interp in log10 space returns 0, then 10^0 = 1.
    auto r = gi.query_interp(epoch_h(0), 6.0, 30.0, 300.0);
    CHECK_THAT(r.density, WithinRel(1.0, 1e-5));
}

TEST_CASE("GridInterpolator: time interpolation between two steps") {
    // density=1.0 at t0 and density=100.0 at t1.
    // Spatial interpolation is in log10 space → r0.density=1.0, r1.density=100.0.
    // Temporal interpolation is LINEAR: lerp(1.0, 100.0, 0.5) = 50.5.
    ForecastGrid g;
    g.shape = test_shape();
    g.H = 2;
    g.density.assign(static_cast<std::size_t>(2) * g.shape.voxels(), 1.0f);
    g.uncertainty.assign(static_cast<std::size_t>(2) * g.shape.voxels(), 1.0f);
    std::fill(g.density.begin() + g.shape.voxels(), g.density.end(), 100.0f);
    g.times = {epoch_h(0), epoch_h(2)};  // 2-hour span

    GridInterpolator gi(g);
    auto r = gi.query_interp(epoch_h(1), 12.0, 0.0, 400.0);  // midpoint
    CHECK_THAT(r.density, WithinRel(50.5, 1e-4));
}

TEST_CASE("GridInterpolator: query at exact time step equals hold") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);

    auto r_interp = gi.query_interp(epoch_h(0), 12.0, 0.0, 400.0);
    auto r_hold   = gi.query_hold(epoch_h(0), 12.0, 0.0, 400.0);
    CHECK_THAT(r_interp.density,     WithinRel(r_hold.density,     1e-9));
    CHECK_THAT(r_interp.uncertainty, WithinRel(r_hold.uncertainty, 1e-9));
}

TEST_CASE("GridInterpolator: LST is periodic - query at 25h equals query at 1h") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);

    auto r1  = gi.query_interp(epoch_h(0), 1.0,  0.0, 400.0);
    auto r25 = gi.query_interp(epoch_h(0), 25.0, 0.0, 400.0);
    CHECK_THAT(r1.density,  WithinRel(r25.density,  1e-9));
    CHECK_THAT(r1.uncertainty, WithinRel(r25.uncertainty, 1e-9));
}

TEST_CASE("GridInterpolator: latitude beyond +/-90 throws") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0,  90.5, 400.0), SpatialOutOfRangeError);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, -90.5, 400.0), SpatialOutOfRangeError);
}

TEST_CASE("GridInterpolator: latitude between lat_max_deg and 90 no longer throws") {
    // test_shape() covers [-87.5, 87.5]; 87.5..90 and -90..-87.5 are the polar-cap bands.
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    for (double lat : {87.6, 89.0, 90.0, -87.6, -89.0, -90.0}) {
        auto r = gi.query_interp(epoch_h(0), 12.0, lat, 400.0);
        CHECK(r.density     > 0.0);
        CHECK(r.uncertainty > 0.0);
    }
}

TEST_CASE("GridInterpolator: value at the pole is the LST geometric mean of the boundary row") {
    auto shape = test_shape();
    ForecastGrid g;
    g.shape = shape;
    g.H = 1;
    g.density.assign(shape.voxels(), 1.0f);
    g.uncertainty.assign(shape.voxels(), 1.0f);
    g.times = {epoch_h(0)};

    // Boundary rows (ai=0 south, ai=n_lat-1 north) alternate 10 / 1000 across LST at every
    // altitude; the geometric mean of alternating 10/1000 over an even count is sqrt(10*1000)=100.
    for (int a : {0, shape.n_lat - 1}) {
        for (int l = 0; l < shape.n_lst; ++l) {
            for (int z = 0; z < shape.n_alt; ++z) {
                std::size_t idx = static_cast<std::size_t>(l) * (shape.n_lat * shape.n_alt)
                                 + static_cast<std::size_t>(a) * shape.n_alt + z;
                g.density[idx] = (l % 2 == 0) ? 10.0f : 1000.0f;
            }
        }
    }

    GridInterpolator gi(g);
    for (double pole : {90.0, -90.0}) {
        auto r_noon = gi.query_interp(epoch_h(0), 12.0, pole, 400.0);
        auto r_dawn = gi.query_interp(epoch_h(0),  6.0, pole, 400.0);
        CHECK_THAT(r_noon.density, WithinRel(100.0, 1e-3));
        // Independent of LST: querying the pole at two different LSTs must agree exactly.
        CHECK_THAT(r_noon.density, WithinRel(r_dawn.density, 1e-9));
    }
}

TEST_CASE("GridInterpolator: altitude below alt_min_km always throws") {
    // The floor is hard regardless of the extrapolation toggle.
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, 0.0, 50.0), SpatialOutOfRangeError);

    GridInterpolator gi_off(grid, ExtrapolationOptions{false, 8});
    REQUIRE_THROWS_AS(gi_off.query_interp(epoch_h(0), 12.0, 0.0, 50.0), SpatialOutOfRangeError);
}

TEST_CASE("GridInterpolator: altitude past alt_max_km extrapolates by default") {
    // test_shape() covers alt [100, 980]; 990 is past the boundary but under the 2000 km ceiling.
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    auto r = gi.query_interp(epoch_h(0), 12.0, 0.0, 990.0);
    CHECK(r.density     > 0.0);
    CHECK(r.uncertainty > 0.0);
}

TEST_CASE("GridInterpolator: altitude extrapolation can be disabled") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid, ExtrapolationOptions{false, 8});
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, 0.0, 990.0), SpatialOutOfRangeError);
}

TEST_CASE("GridInterpolator: altitude extrapolation ceiling at 2000 km is not configurable") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    CHECK_NOTHROW(gi.query_interp(epoch_h(0), 12.0, 0.0, 2000.0));
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, 0.0, 2000.5), SpatialOutOfRangeError);
}

TEST_CASE("GridInterpolator: n_etp_pts validation") {
    auto grid = make_uniform_grid();  // test_shape(): n_alt = 45
    REQUIRE_THROWS_AS(GridInterpolator(grid, ExtrapolationOptions{true, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS(GridInterpolator(grid, ExtrapolationOptions{true, 46}), std::invalid_argument);

    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.set_extrapolation_options(ExtrapolationOptions{true, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS(gi.set_extrapolation_options(ExtrapolationOptions{true, 46}), std::invalid_argument);
}

TEST_CASE("GridInterpolator: extrapolated density is continuous with the in-grid value at the boundary") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    auto r_boundary = gi.query_interp(epoch_h(0), 12.0, 0.0, 980.0);
    auto r_just_past = gi.query_interp(epoch_h(0), 12.0, 0.0, 980.5);
    CHECK_THAT(r_just_past.density, WithinRel(r_boundary.density, 1e-2));
}

TEST_CASE("GridInterpolator: extrapolation composes with the polar cap") {
    // lat in the polar-cap band AND alt past alt_max_km, simultaneously -- the case the
    // real-column slope_/intercept_ arrays don't cover; must use the pole's own fit.
    auto shape = test_shape();
    ForecastGrid g;
    g.shape = shape;
    g.H = 1;
    g.density.assign(shape.voxels(), 1.0f);
    g.uncertainty.assign(shape.voxels(), 1.0f);
    g.times = {epoch_h(0)};

    // Make the top few altitude rows at the north boundary decrease by 10x per step, alternating
    // 10/1000 across LST like the geometric-mean test above, so the pole fit is non-trivial.
    for (int a : {0, shape.n_lat - 1}) {
        for (int l = 0; l < shape.n_lst; ++l) {
            for (int z = shape.n_alt - 3; z < shape.n_alt; ++z) {
                std::size_t idx = static_cast<std::size_t>(l) * (shape.n_lat * shape.n_alt)
                                 + static_cast<std::size_t>(a) * shape.n_alt + z;
                float base = (l % 2 == 0) ? 10.0f : 1000.0f;
                g.density[idx] = base / std::pow(10.0f, static_cast<float>(shape.n_alt - 1 - z));
            }
        }
    }

    GridInterpolator gi(g);
    auto r_noon = gi.query_interp(epoch_h(0), 12.0, 90.0, 1200.0);
    auto r_dawn = gi.query_interp(epoch_h(0),  6.0, 90.0, 1200.0);
    CHECK(r_noon.density > 0.0);
    // Still LST-invariant at the pole even while extrapolating past alt_max_km.
    CHECK_THAT(r_noon.density, WithinRel(r_dawn.density, 1e-6));
}

TEST_CASE("GridInterpolator: out-of-range time throws") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0) - 1, 12.0, 0.0, 400.0), TimeOutOfRangeError);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(1) + 1, 12.0, 0.0, 400.0), TimeOutOfRangeError);
}

TEST_CASE("GridInterpolator: density is non-negative for positive grid values") {
    auto grid = make_uniform_grid(0.001f, 0.001f);
    GridInterpolator gi(grid);
    for (double lst : {0.0, 6.0, 12.0, 18.0}) {
        for (double lat : {-80.0, 0.0, 80.0}) {
            auto r = gi.query_interp(epoch_h(0), lst, lat, 400.0);
            CHECK(r.density     >= 0.0);
            CHECK(r.uncertainty >= 0.0);
        }
    }
}
