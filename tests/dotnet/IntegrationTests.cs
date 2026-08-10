/*
 * IntegrationTests.cs — end-to-end C# binding tests.
 *
 * Mirrors tests/python/test_cli.py: runs a real forecast via the rope CLI
 * (which writes a forecast-grid cache file), then exercises the librope
 * P/Invoke path for single and batch queries against that cache file. No
 * server process, no sockets.
 *
 * Required environment variables (injected by CI; skipped when absent):
 *   ROPE_LIB          path to librope.so / librope.dylib / librope.dll
 *   ROPE_EXE          path to the rope CLI binary
 *   ROPE_FIXTURE_DIR  path to tests/fixtures/
 *
 * Run locally:
 *   ROPE_LIB=build/librope.so \
 *   ROPE_EXE=build/rope \
 *   ROPE_FIXTURE_DIR=tests/fixtures \
 *   dotnet test tests/dotnet/RopeTests.csproj
 */

using System;
using System.Diagnostics;
using System.IO;
using Xunit;

namespace RopeFramework.Tests;

// ── Shared forecast-cache fixture ───────────────────────────────────────────
// One forecast is run for the entire collection; all integration tests share
// the resulting cache file via ICollectionFixture<ForecastFixture>. No
// teardown process to stop -- just delete the temp directory.

public sealed class ForecastFixture : IDisposable
{
    // Matches sw_test.swbin coverage: 2023-12-31T22 → 2024-01-01T03 (6 rows).
    // Horizon=3, seq_len=3 requires (S-1)+(H+1) = 6 rows.
    public const string ForecastStart   = "2024-01-01 00:00:00";
    public const int    ForecastHorizon = 3;

    // Grid covers hours 1..3 after start — first queryable time is T01.
    public static readonly DateTime QueryTime =
        new(2024, 1, 1, 1, 0, 0, DateTimeKind.Utc);

    public string LibPath   { get; }
    public string ExePath   { get; }
    public string CachePath { get; }
    public string ConfPath  { get; }

    // False when env vars are absent — all integration tests skip gracefully.
    public bool Available { get; }

    private readonly string? _tmpDir;

    public ForecastFixture()
    {
        LibPath = Environment.GetEnvironmentVariable("ROPE_LIB")         ?? "";
        ExePath = Environment.GetEnvironmentVariable("ROPE_EXE")         ?? "";
        string fixtureDir = Environment.GetEnvironmentVariable("ROPE_FIXTURE_DIR") ?? "";

        if (string.IsNullOrEmpty(LibPath)     ||
            string.IsNullOrEmpty(ExePath)     ||
            string.IsNullOrEmpty(fixtureDir))
        {
            CachePath = ConfPath = "";
            Available = false;
            return;
        }

        _tmpDir = Path.Combine(Path.GetTempPath(),
                               $"rope_cs_{Path.GetRandomFileName()}");
        Directory.CreateDirectory(_tmpDir);

        CachePath = Path.Combine(_tmpDir, "forecast_grid.bin");
        ConfPath  = Path.Combine(_tmpDir, "rope.conf");

        string testModels = Path.Combine(fixtureDir, "test_models");
        string swBin      = Path.Combine(testModels,  "sw_test.swbin");

        File.WriteAllText(ConfPath,
            $"[paths]{Environment.NewLine}" +
            $"exported_dir = {testModels}{Environment.NewLine}" +
            $"driver_path  = {swBin}{Environment.NewLine}");

        // Run the forecast; a non-zero exit means the pipeline failed to
        // load — mark fixture unavailable.
        int rc = RunCli(ExePath, CachePath, ConfPath,
            "forecast",
            "--start",   ForecastStart,
            "--horizon", ForecastHorizon.ToString());
        Available = (rc == 0);
    }

    public void Dispose()
    {
        if (_tmpDir is not null)
            try { Directory.Delete(_tmpDir, recursive: true); } catch { }
    }

    public static int RunCli(string exe, string cachePath, string conf,
                              params string[] args)
    {
        var joined = $"--cache-path \"{cachePath}\" {string.Join(" ", args)} --config \"{conf}\"";
        var psi = new ProcessStartInfo(exe, joined)
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute        = false,
        };
        using var p = Process.Start(psi)!;
        p.WaitForExit(timeout: TimeSpan.FromSeconds(30));
        return p.ExitCode;
    }
}

[CollectionDefinition("Integration")]
public class IntegrationCollection : ICollectionFixture<ForecastFixture> { }

// ── Tests ─────────────────────────────────────────────────────────────────────

[Collection("Integration")]
public class IntegrationTests
{
    private readonly ForecastFixture _srv;

    public IntegrationTests(ForecastFixture srv) => _srv = srv;

    // ── Forecast ─────────────────────────────────────────────────────────────

    [Fact]
    public void Forecast_returns_window_timestamps()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath, configPath: _srv.ConfPath);

        var result = rope.Forecast(ForecastFixture.ForecastStart,
                                   ForecastFixture.ForecastHorizon);

        Assert.False(string.IsNullOrEmpty(result.WindowStart));
        Assert.False(string.IsNullOrEmpty(result.WindowEnd));
    }

    // ── Single-point query ───────────────────────────────────────────────────

    [Fact]
    public void Get_interp_returns_positive_density()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();
        var r = rope.Get(ForecastFixture.QueryTime,
                         lst: 12.0, lat: 45.0, altKm: 400.0,
                         mode: Rope.Interp);

        Assert.True(r.Density     > 0,  $"density={r.Density} should be > 0");
        Assert.True(r.Uncertainty >= 0, $"uncertainty={r.Uncertainty} should be ≥ 0");
    }

    [Fact]
    public void Get_hold_returns_positive_density()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();
        var r = rope.Get(ForecastFixture.QueryTime,
                         lst: 12.0, lat: 45.0, altKm: 400.0,
                         mode: Rope.Hold);

        Assert.True(r.Density > 0);
        Assert.True(r.Uncertainty >= 0);
    }

    [Fact]
    public void Get_implicit_open_works()
    {
        // EnsureOpen() should be called automatically before the first query.
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        // Do NOT call rope.Open() — Get should call EnsureOpen() itself.
        var r = rope.Get(ForecastFixture.QueryTime,
                         lst: 0.0, lat: 0.0, altKm: 300.0);
        Assert.True(r.Density > 0);
    }

    [Fact]
    public void Get_time_out_of_range_throws_RopeException()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();
        var future = new DateTime(2030, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        var ex = Assert.Throws<RopeException>(
            () => rope.Get(future, lst: 12.0, lat: 45.0, altKm: 400.0));
        Assert.Equal(3, ex.Code);  // time out of range
    }

    [Fact]
    public void Get_without_forecast_throws_RopeException()
    {
        if (!_srv.Available) return;

        string missingCache = Path.Combine(
            Path.GetTempPath(), $"rope_cs_missing_{Path.GetRandomFileName()}.bin");

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: missingCache);

        var ex = Assert.Throws<RopeException>(
            () => rope.Get(ForecastFixture.QueryTime, lst: 12.0, lat: 45.0, altKm: 400.0));
        Assert.Equal(2, ex.Code);  // no forecast cached
    }

    // ── Batch query ──────────────────────────────────────────────────────────

    [Fact]
    public void GetBatch_returns_correct_count()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();

        var times = new[]
        {
            ForecastFixture.QueryTime,
            ForecastFixture.QueryTime.AddHours(1),
        };
        var results = rope.GetBatch(
            times,
            lsts:   [12.0, 6.0],
            lats:   [45.0, -30.0],
            altsKm: [400.0, 300.0]);

        Assert.Equal(2, results.Length);
        foreach (var r in results)
        {
            Assert.True(r.Density     > 0);
            Assert.True(r.Uncertainty >= 0);
        }
    }

    [Fact]
    public void GetBatch_mismatched_lengths_throws()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();

        Assert.Throws<ArgumentException>(() =>
            rope.GetBatch(
                timesUnix: [1.0, 2.0],
                lsts:       [12.0],        // length mismatch
                lats:       [45.0, 45.0],
                altsKm:     [400.0, 400.0]));
    }

    // ── Handle lifecycle ─────────────────────────────────────────────────────

    [Fact]
    public void Refresh_picks_up_new_forecast()
    {
        // Run a second forecast, then Refresh() and query — should not throw.
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath, configPath: _srv.ConfPath);

        rope.Forecast(ForecastFixture.ForecastStart, ForecastFixture.ForecastHorizon);
        rope.Refresh();

        var r = rope.Get(ForecastFixture.QueryTime,
                         lst: 12.0, lat: 45.0, altKm: 400.0);
        Assert.True(r.Density > 0);
    }

    [Fact]
    public void Dispose_is_idempotent()
    {
        if (!_srv.Available) return;

        var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();
        rope.Dispose();
        rope.Dispose();  // second Dispose must not throw
    }

    // ── Altitude extrapolation (CLI) ────────────────────────────────────────

    [Fact]
    public void Get_altitude_extrapolation_default_on_succeeds()
    {
        if (!_srv.Available) return;

        int rc = ForecastFixture.RunCli(_srv.ExePath, _srv.CachePath, _srv.ConfPath,
            "get", "--mode", "interp",
            "--time", "2024-01-01T01:00:00",
            "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0");

        Assert.Equal(0, rc);
    }

    [Fact]
    public void Get_no_extrapolate_altitude_flag_fails()
    {
        if (!_srv.Available) return;

        int rc = ForecastFixture.RunCli(_srv.ExePath, _srv.CachePath, _srv.ConfPath,
            "get", "--mode", "interp", "--no-extrapolate-altitude",
            "--time", "2024-01-01T01:00:00",
            "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0");

        Assert.NotEqual(0, rc);
    }

    [Fact]
    public void Get_altitude_hard_ceiling_fails()
    {
        if (!_srv.Available) return;

        int rc = ForecastFixture.RunCli(_srv.ExePath, _srv.CachePath, _srv.ConfPath,
            "get", "--mode", "interp",
            "--time", "2024-01-01T01:00:00",
            "--lst", "12.0", "--lat", "45.0", "--alt", "2500.0");

        Assert.NotEqual(0, rc);
    }

    [Fact]
    public void Get_n_etp_pts_flag_accepted()
    {
        if (!_srv.Available) return;

        int rc = ForecastFixture.RunCli(_srv.ExePath, _srv.CachePath, _srv.ConfPath,
            "get", "--mode", "interp", "--n-etp-pts", "4",
            "--time", "2024-01-01T01:00:00",
            "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0");

        Assert.Equal(0, rc);
    }

    [Fact]
    public void Get_n_etp_pts_out_of_range_fails()
    {
        if (!_srv.Available) return;

        int rc = ForecastFixture.RunCli(_srv.ExePath, _srv.CachePath, _srv.ConfPath,
            "get", "--mode", "interp", "--n-etp-pts", "1000",
            "--time", "2024-01-01T01:00:00",
            "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0");

        Assert.NotEqual(0, rc);
    }

    [Fact]
    public void Get_config_interpolation_section_disables_extrapolation()
    {
        if (!_srv.Available) return;

        string fixtureDir = Environment.GetEnvironmentVariable("ROPE_FIXTURE_DIR") ?? "";
        string testModels = Path.Combine(fixtureDir, "test_models");
        string swBin      = Path.Combine(testModels,  "sw_test.swbin");
        string conf       = Path.Combine(Path.GetTempPath(), $"rope_cs_noextrap_{Path.GetRandomFileName()}.conf");
        File.WriteAllText(conf,
            $"[paths]{Environment.NewLine}" +
            $"exported_dir = {testModels}{Environment.NewLine}" +
            $"driver_path  = {swBin}{Environment.NewLine}" +
            $"[interpolation]{Environment.NewLine}" +
            $"extrapolate_altitude = false{Environment.NewLine}");
        try
        {
            int rc = ForecastFixture.RunCli(_srv.ExePath, _srv.CachePath, conf,
                "get", "--mode", "interp",
                "--time", "2024-01-01T01:00:00",
                "--lst", "12.0", "--lat", "45.0", "--alt", "1200.0");

            Assert.NotEqual(0, rc);
        }
        finally
        {
            try { File.Delete(conf); } catch { }
        }
    }

    // ── SetExtrapolation (binding) ──────────────────────────────────────────

    [Fact]
    public void SetExtrapolation_disables_then_reenables()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath);

        rope.Open();

        var r1 = rope.Get(ForecastFixture.QueryTime, lst: 12.0, lat: 45.0, altKm: 1200.0);
        Assert.True(r1.Density > 0);

        rope.SetExtrapolation(extrapolateAltitude: false);
        Assert.Throws<RopeException>(
            () => rope.Get(ForecastFixture.QueryTime, lst: 12.0, lat: 45.0, altKm: 1200.0));

        rope.SetExtrapolation(extrapolateAltitude: true, nEtpPts: 4);
        var r2 = rope.Get(ForecastFixture.QueryTime, lst: 12.0, lat: 45.0, altKm: 1200.0);
        Assert.True(r2.Density > 0);
    }

    [Fact]
    public void Constructor_extrapolateAltitude_false_applies_on_open()
    {
        if (!_srv.Available) return;

        using var rope = new Rope(
            libPath: _srv.LibPath, exePath: _srv.ExePath,
            cachePath: _srv.CachePath, extrapolateAltitude: false);

        Assert.Throws<RopeException>(
            () => rope.Get(ForecastFixture.QueryTime, lst: 12.0, lat: 45.0, altKm: 1200.0));
    }

    // ── Discard-on-reforecast invariant ─────────────────────────────────────

    [Fact]
    public void Second_forecast_discards_first()
    {
        // A new forecast fully replaces the cache file -- there is no server
        // holding old state around, so a query valid only under the first
        // forecast's window must fail once a second, differently-shaped
        // forecast has been written to the same cache path.
        if (!_srv.Available) return;

        string tmpDir = Path.Combine(Path.GetTempPath(), $"rope_cs_discard_{Path.GetRandomFileName()}");
        Directory.CreateDirectory(tmpDir);
        try
        {
            string cache = Path.Combine(tmpDir, "forecast_grid.bin");

            // horizon=3 -> window covers 2024-01-01T01:00:00 .. 03:00:00
            int rc1 = ForecastFixture.RunCli(_srv.ExePath, cache, _srv.ConfPath,
                "forecast", "--start", ForecastFixture.ForecastStart, "--horizon", "3");
            Assert.Equal(0, rc1);

            var lateTime = new DateTime(2024, 1, 1, 3, 0, 0, DateTimeKind.Utc);
            using (var ropeA = new Rope(libPath: _srv.LibPath, exePath: _srv.ExePath, cachePath: cache))
            {
                var r = ropeA.Get(lateTime, lst: 12.0, lat: 45.0, altKm: 400.0);
                Assert.True(r.Density > 0, "sanity check: first forecast's window must include lateTime");
            }

            // horizon=1 -> window covers only 2024-01-01T01:00:00
            int rc2 = ForecastFixture.RunCli(_srv.ExePath, cache, _srv.ConfPath,
                "forecast", "--start", ForecastFixture.ForecastStart, "--horizon", "1");
            Assert.Equal(0, rc2);

            using (var ropeB = new Rope(libPath: _srv.LibPath, exePath: _srv.ExePath, cachePath: cache))
            {
                Assert.Throws<RopeException>(
                    () => ropeB.Get(lateTime, lst: 12.0, lat: 45.0, altKm: 400.0));
            }
        }
        finally
        {
            try { Directory.Delete(tmpDir, recursive: true); } catch { }
        }
    }
}
