using RopeFramework;

if (args.Length < 1)
{
    Console.Error.WriteLine("usage: PackSmokeConsumer <rope.conf path>");
    return 1;
}

string configPath = args[0];

// No libPath/exePath passed — this exercises the exact auto-detection path
// (ResolveLibPath/ResolveExePath) every real NuGet consumer relies on.
using var rope = new Rope(configPath: configPath);

var forecast = rope.Forecast("2024-01-01 00:00:00", 3);
Console.WriteLine($"forecast window: {forecast.WindowStart}..{forecast.WindowEnd}");

var result = rope.Get(
    new DateTime(2024, 1, 1, 1, 0, 0, DateTimeKind.Utc),
    lst: 12.0, lat: 45.0, altKm: 400.0);
Console.WriteLine($"density={result.Density} uncertainty={result.Uncertainty}");

if (!(result.Density > 0))
    throw new Exception($"density should be > 0, got {result.Density}");
if (!(result.Uncertainty >= 0))
    throw new Exception($"uncertainty should be >= 0, got {result.Uncertainty}");

Console.WriteLine("Pack smoke test passed.");
return 0;
