using RopeFramework;
using Xunit;

namespace RopeFramework.Tests;

public class RopeExceptionTests
{
    // ── Code → name mapping ──────────────────────────────────────────────────

    [Theory]
    [InlineData(0, "ok")]
    [InlineData(2, "no forecast cached")]
    [InlineData(3, "time out of range")]
    [InlineData(4, "spatial point out of range")]
    [InlineData(5, "bad argument")]
    [InlineData(6, "internal error")]
    [InlineData(7, "forecast cache corrupt")]
    public void Known_code_formats_as_name(int code, string name)
    {
        var ex = new RopeException(code, "detail");
        Assert.Contains($"[{name}]", ex.Message);
        Assert.Contains("detail", ex.Message);
        Assert.Equal(code, ex.Code);
    }

    [Fact]
    public void Unknown_code_formats_as_integer()
    {
        var ex = new RopeException(99, "msg");
        Assert.Contains("[99]", ex.Message);
        Assert.Equal(99, ex.Code);
    }

    [Fact]
    public void Negative_code_formats_as_integer()
    {
        var ex = new RopeException(-1, "msg");
        Assert.Contains("[-1]", ex.Message);
    }

    [Fact]
    public void Is_an_Exception()
    {
        Assert.IsAssignableFrom<Exception>(new RopeException(0, "x"));
    }
}

public class ToUnixTests
{
    // ToUnix is internal — visible via InternalsVisibleTo("RopeTests").

    [Fact]
    public void Unix_epoch_gives_zero()
    {
        var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        Assert.Equal(0.0, Rope.ToUnix(epoch));
    }

    [Fact]
    public void Known_timestamp_is_correct()
    {
        // 2024-01-01 00:00:00 UTC = 1704067200
        var dt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        Assert.Equal(1704067200.0, Rope.ToUnix(dt));
    }

    [Fact]
    public void Unspecified_kind_is_treated_as_utc()
    {
        var utc        = new DateTime(2024, 6, 1, 12, 0, 0, DateTimeKind.Utc);
        var unspecified = new DateTime(2024, 6, 1, 12, 0, 0, DateTimeKind.Unspecified);
        Assert.Equal(Rope.ToUnix(utc), Rope.ToUnix(unspecified));
    }

    [Fact]
    public void One_hour_step_is_3600_seconds()
    {
        var t0 = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var t1 = new DateTime(2024, 1, 1, 1, 0, 0, DateTimeKind.Utc);
        Assert.Equal(3600.0, Rope.ToUnix(t1) - Rope.ToUnix(t0));
    }
}
