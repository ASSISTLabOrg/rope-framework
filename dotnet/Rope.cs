/*
 * Rope.cs — C# binding for the ROPE framework.
 *
 * Wraps librope via P/Invoke for fast in-process interpolation queries
 * against a memory-mapped forecast-grid cache file, and the rope CLI
 * subprocess to run forecasts (which write that cache file).
 *
 * Targets both .NET Framework 4.8 and .NET 8 from a single source file —
 * avoids System.Runtime.InteropServices.NativeLibrary and System.Text.Json,
 * neither of which exist on .NET Framework, in favor of a small P/Invoke
 * loader and minimal JSON field extraction.
 *
 * Typical usage
 * -------------
 *     using RopeFramework;
 *
 *     var r = new Rope();
 *     r.Forecast("2024-02-09 00:00:00", horizon: 24);
 *
 *     using (r)  // opens handle, closes on dispose
 *     {
 *         var result = r.Get(
 *             time: new DateTime(2024, 2, 9, 6, 0, 0, DateTimeKind.Utc),
 *             lst: 7.5, lat: 45.0, altKm: 400.0);
 *         Console.WriteLine("density=" + result.Density + "  uncertainty=" + result.Uncertainty);
 *     }
 */

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("RopeTests")]

namespace RopeFramework
{
    public class RopeException : Exception
    {
        public int Code { get; }

        private static readonly System.Collections.Generic.Dictionary<int, string> s_names =
            new System.Collections.Generic.Dictionary<int, string>
        {
            { 0, "ok" },
            { 2, "no forecast cached" },
            { 3, "time out of range" },
            { 4, "spatial point out of range" },
            { 5, "bad argument" },
            { 6, "internal error" },
            { 7, "forecast cache corrupt" },
        };

        public RopeException(int code, string message)
            : base("[" + (s_names.TryGetValue(code, out var name) ? name : code.ToString()) + "] " + message)
        {
            Code = code;
        }
    }

    public sealed class QueryResult
    {
        public double Density     { get; }
        public double Uncertainty { get; }

        internal QueryResult(double density, double uncertainty)
        {
            Density     = density;
            Uncertainty = uncertainty;
        }
    }

    public sealed class ForecastResult
    {
        public string WindowStart { get; }
        public string WindowEnd   { get; }

        internal ForecastResult(string windowStart, string windowEnd)
        {
            WindowStart = windowStart;
            WindowEnd   = windowEnd;
        }
    }

    public sealed unsafe class Rope : IDisposable
    {
        public const int Hold   = 0;
        public const int Interp = 1;

        private const int ErrBufLen = 256;

        // -------------------------------------------------------------------------
        // Native delegate types
        // -------------------------------------------------------------------------

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr RopeOpenFn(byte* sockPath, byte* errBuf, int errLen);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int RopeQueryFn(
            IntPtr interp, int mode,
            double timeUnix, double lst, double lat, double altKm,
            double* density, double* uncertainty,
            byte* errBuf, int errLen);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int RopeQueryBatchFn(
            IntPtr interp, int mode, int n,
            double* timesUnix, double* lst, double* lat, double* altKm,
            double* density, double* uncertainty,
            byte* errBuf, int errLen);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void RopeCloseFn(IntPtr interp);

        // -------------------------------------------------------------------------
        // Fields
        // -------------------------------------------------------------------------

        private readonly IntPtr            _lib;
        private readonly RopeOpenFn        _ropeOpen;
        private readonly RopeQueryFn       _ropeQuery;
        private readonly RopeQueryBatchFn  _ropeQueryBatch;
        private readonly RopeCloseFn       _ropeClose;

        private readonly string _cachePath;
        private readonly string _exePath;
        private readonly string _configPath;
        private IntPtr _handle;
        private bool   _disposed;

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------

        /// <param name="libPath">
        ///   Path to librope.so / librope.dylib / librope.dll.
        ///   Defaults to a file alongside Rope.dll, then lib/librope.* relative
        ///   to the package root.
        /// </param>
        /// <param name="exePath">
        ///   Path to the rope CLI executable.
        ///   Defaults to a file alongside Rope.dll, then bin/rope relative to
        ///   the package root.
        /// </param>
        /// <param name="cachePath">
        ///   Forecast-grid cache file path. Null → platform default.
        /// </param>
        /// <param name="configPath">
        ///   Path to rope.conf. Defaults to config/rope.conf in the package root.
        /// </param>
        public Rope(
            string libPath    = null,
            string exePath    = null,
            string cachePath  = null,
            string configPath = null)
        {
            string asmDir = Path.GetDirectoryName(typeof(Rope).Assembly.Location) ?? ".";
            string root   = Path.GetFullPath(Path.Combine(asmDir, ".."));

            if (libPath == null) libPath = ResolveLibPath(root, asmDir);
            if (exePath == null) exePath = ResolveExePath(root, asmDir);

            _lib            = NativeLib.Load(libPath);
            _ropeOpen       = Load<RopeOpenFn>("rope_open");
            _ropeQuery      = Load<RopeQueryFn>("rope_query");
            _ropeQueryBatch = Load<RopeQueryBatchFn>("rope_query_batch");
            _ropeClose      = Load<RopeCloseFn>("rope_close");

            _cachePath  = cachePath;
            _exePath    = exePath;
            _configPath = configPath;
        }

        private T Load<T>(string symbol) where T : class =>
            Marshal.GetDelegateForFunctionPointer<T>(NativeLib.GetExport(_lib, symbol));

        // -------------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------------

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Close();
            NativeLib.Free(_lib);
        }

        // -------------------------------------------------------------------------
        // Handle lifecycle
        // -------------------------------------------------------------------------

        /// <summary>Memory-map the cached forecast grid and open an interpolation handle.</summary>
        public void Open()
        {
            if (_handle != IntPtr.Zero) return;
            byte* err = stackalloc byte[ErrBufLen];

            IntPtr handle;
            if (_cachePath != null)
            {
                byte[] cache = Encoding.UTF8.GetBytes(_cachePath + '\0');
                fixed (byte* cachePtr = cache)
                    handle = _ropeOpen(cachePtr, err, ErrBufLen);
            }
            else
            {
                handle = _ropeOpen(null, err, ErrBufLen);
            }

            if (handle == IntPtr.Zero)
                throw new RopeException(2, ReadErr(err));
            _handle = handle;
        }

        /// <summary>Release the interpolation handle (unmaps the cache file).</summary>
        public void Close()
        {
            if (_handle == IntPtr.Zero) return;
            _ropeClose(_handle);
            _handle = IntPtr.Zero;
        }

        /// <summary>Re-map the current cache file (picks up a forecast written since Open()).</summary>
        public void Refresh()
        {
            Close();
            Open();
        }

        /// <summary>Alias for Close(). There is no background process to stop --
        /// kept as a method for source compatibility with existing code.</summary>
        public void Shutdown()
        {
            Close();
        }

        // -------------------------------------------------------------------------
        // Forecast (via CLI subprocess)
        // -------------------------------------------------------------------------

        /// <summary>Run a forecast and atomically write the resulting grid to the cache file.</summary>
        /// <param name="start">Forecast start time (UTC).</param>
        /// <param name="horizon">Forecast duration in hours.</param>
        public ForecastResult Forecast(DateTime start, int horizon) =>
            Forecast(start.ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss"), horizon);

        /// <summary>Run a forecast and atomically write the resulting grid to the cache file.</summary>
        /// <param name="start">ISO 8601 forecast start time string (UTC).</param>
        /// <param name="horizon">Forecast duration in hours.</param>
        public ForecastResult Forecast(string start, int horizon)
        {
            if (_exePath == null)
                throw new InvalidOperationException(
                    "rope executable not found; pass exePath explicitly or check your package layout");

            // --cache-path is a global option and must precede the subcommand name.
            var sb = new StringBuilder();
            if (_cachePath != null)
            {
                sb.Append("--cache-path \"");
                sb.Append(_cachePath);
                sb.Append("\" ");
            }
            sb.Append("forecast --start \"");
            sb.Append(start);
            sb.Append("\" --horizon ");
            sb.Append(horizon);
            if (_configPath != null)
            {
                sb.Append(" --config \"");
                sb.Append(_configPath);
                sb.Append('"');
            }

            var proc = Process.Start(new ProcessStartInfo(_exePath, sb.ToString())
            {
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                UseShellExecute        = false,
            });

            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();

            if (proc.ExitCode != 0)
                throw new RopeException(6, (stderr.Length > 0 ? stderr : stdout).Trim());

            string lastLine = null;
            foreach (var line in stdout.Split('\n'))
                if (line.Trim().Length > 0)
                    lastLine = line.Trim();

            string json = lastLine ?? "{}";
            return new ForecastResult(
                ExtractJsonString(json, "window_start"),
                ExtractJsonString(json, "window_end"));
        }

        // -------------------------------------------------------------------------
        // Interpolation queries
        // -------------------------------------------------------------------------

        /// <summary>Query density and uncertainty at a single point.</summary>
        /// <param name="timeUnix">Query time as Unix timestamp (seconds since 1970-01-01T00:00:00 UTC).</param>
        /// <param name="lst">Local Solar Time, hours [0, 24).</param>
        /// <param name="lat">Geodetic latitude, degrees [-87.5, 87.5].</param>
        /// <param name="altKm">Geometric altitude, km [100, 980].</param>
        /// <param name="mode">Rope.Hold or Rope.Interp (default).</param>
        public QueryResult Get(double timeUnix, double lst, double lat, double altKm, int mode = Interp)
        {
            EnsureOpen();
            double density, uncertainty;
            byte* err = stackalloc byte[ErrBufLen];

            int rc = _ropeQuery(_handle, mode, timeUnix, lst, lat, altKm,
                                &density, &uncertainty, err, ErrBufLen);
            if (rc != 0) throw new RopeException(rc, ReadErr(err));

            return new QueryResult(density, uncertainty);
        }

        /// <summary>Query density and uncertainty at a single point.</summary>
        /// <param name="time">Query time (UTC).</param>
        /// <param name="lst">Local Solar Time, hours [0, 24).</param>
        /// <param name="lat">Geodetic latitude, degrees [-87.5, 87.5].</param>
        /// <param name="altKm">Geometric altitude, km [100, 980].</param>
        /// <param name="mode">Rope.Hold or Rope.Interp (default).</param>
        public QueryResult Get(DateTime time, double lst, double lat, double altKm, int mode = Interp) =>
            Get(ToUnix(time), lst, lat, altKm, mode);

        /// <summary>Query density and uncertainty at N points in one call.</summary>
        /// <param name="timesUnix">Array of N Unix timestamps.</param>
        /// <param name="lsts">Array of N Local Solar Time values.</param>
        /// <param name="lats">Array of N latitude values.</param>
        /// <param name="altsKm">Array of N altitude values.</param>
        /// <param name="mode">Rope.Hold or Rope.Interp (default), applied to all points.</param>
        public QueryResult[] GetBatch(
            double[] timesUnix, double[] lsts, double[] lats, double[] altsKm,
            int mode = Interp)
        {
            int n = timesUnix.Length;
            if (lsts.Length != n || lats.Length != n || altsKm.Length != n)
                throw new ArgumentException("all input arrays must have the same length");

            EnsureOpen();

            var density     = new double[n];
            var uncertainty = new double[n];
            byte* err = stackalloc byte[ErrBufLen];

            fixed (double* tPtr   = timesUnix,
                           lstPtr  = lsts,
                           latPtr  = lats,
                           altPtr  = altsKm,
                           denPtr  = density,
                           uncPtr  = uncertainty)
            {
                int rc = _ropeQueryBatch(_handle, mode, n,
                                         tPtr, lstPtr, latPtr, altPtr,
                                         denPtr, uncPtr, err, ErrBufLen);
                if (rc != 0) throw new RopeException(rc, ReadErr(err));
            }

            var results = new QueryResult[n];
            for (int i = 0; i < n; i++)
                results[i] = new QueryResult(density[i], uncertainty[i]);
            return results;
        }

        /// <summary>Query density and uncertainty at N points in one call.</summary>
        /// <param name="times">Array of N DateTime values (UTC).</param>
        /// <param name="lsts">Array of N Local Solar Time values.</param>
        /// <param name="lats">Array of N latitude values.</param>
        /// <param name="altsKm">Array of N altitude values.</param>
        /// <param name="mode">Rope.Hold or Rope.Interp (default), applied to all points.</param>
        public QueryResult[] GetBatch(
            DateTime[] times, double[] lsts, double[] lats, double[] altsKm,
            int mode = Interp)
        {
            var timesUnix = new double[times.Length];
            for (int i = 0; i < times.Length; i++)
                timesUnix[i] = ToUnix(times[i]);
            return GetBatch(timesUnix, lsts, lats, altsKm, mode);
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private void EnsureOpen()
        {
            if (_handle == IntPtr.Zero) Open();
        }

        private static string ReadErr(byte* buf)
        {
            int len = 0;
            while (len < ErrBufLen && buf[len] != 0) len++;
            return Encoding.UTF8.GetString(buf, len);
        }

        internal static double ToUnix(DateTime dt)
        {
            if (dt.Kind == DateTimeKind.Unspecified)
                dt = DateTime.SpecifyKind(dt, DateTimeKind.Utc);
            var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
            return (dt.ToUniversalTime() - epoch).TotalSeconds;
        }

        private static string ResolveLibPath(string root, string asmDir)
        {
            string[] candidates;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                candidates = new string[]
                {
                    Path.Combine(asmDir, "librope.dll"),
                    Path.Combine(root,   "bin", "librope.dll"),
                };
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                candidates = new string[]
                {
                    Path.Combine(asmDir, "librope.dylib"),
                    Path.Combine(root,   "lib", "librope.dylib"),
                };
            else
                candidates = new string[]
                {
                    Path.Combine(asmDir, "librope.so"),
                    Path.Combine(root,   "lib",   "librope.so"),
                    Path.Combine(root,   "build", "librope.so"),
                };

            foreach (var c in candidates)
                if (File.Exists(c)) return c;

            throw new FileNotFoundException(
                "librope not found; pass libPath explicitly or check your package layout");
        }

        private static string ResolveExePath(string root, string asmDir)
        {
            string[] candidates;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                candidates = new string[]
                {
                    Path.Combine(asmDir, "rope.exe"),
                    Path.Combine(root,   "bin", "rope.exe"),
                };
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                candidates = new string[]
                {
                    Path.Combine(asmDir, "rope"),
                    Path.Combine(asmDir, "rope-osx.bin"),
                    Path.Combine(root,   "bin",   "rope"),
                    Path.Combine(root,   "build", "rope"),
                };
            else
                candidates = new string[]
                {
                    Path.Combine(asmDir, "rope"),
                    Path.Combine(asmDir, "rope.bin"),
                    Path.Combine(root,   "bin",   "rope"),
                    Path.Combine(root,   "build", "rope"),
                };

            foreach (var c in candidates)
                if (File.Exists(c)) return c;

            return null;
        }

        private static string ExtractJsonString(string json, string key)
        {
            string search = "\"" + key + "\"";
            int ki = json.IndexOf(search, StringComparison.Ordinal);
            if (ki < 0) return "";
            int colon = json.IndexOf(':', ki + search.Length);
            if (colon < 0) return "";
            int q1 = json.IndexOf('"', colon + 1);
            if (q1 < 0) return "";
            int q2 = json.IndexOf('"', q1 + 1);
            if (q2 < 0) return "";
            return json.Substring(q1 + 1, q2 - q1 - 1);
        }
    }

    internal static class NativeLib
    {
        // Windows
        [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibraryW(string lpFileName);

        [DllImport("kernel32", SetLastError = true)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);

        [DllImport("kernel32")]
        private static extern bool FreeLibrary(IntPtr hModule);

        // Unix
        [DllImport("libdl", EntryPoint = "dlopen")]
        private static extern IntPtr dlopen_libdl(string path, int flags);
        [DllImport("libdl", EntryPoint = "dlsym")]
        private static extern IntPtr dlsym_libdl(IntPtr handle, string symbol);
        [DllImport("libdl", EntryPoint = "dlclose")]
        private static extern int dlclose_libdl(IntPtr handle);

        [DllImport("libdl.so.2", EntryPoint = "dlopen")]
        private static extern IntPtr dlopen_libdl2(string path, int flags);
        [DllImport("libdl.so.2", EntryPoint = "dlsym")]
        private static extern IntPtr dlsym_libdl2(IntPtr handle, string symbol);
        [DllImport("libdl.so.2", EntryPoint = "dlclose")]
        private static extern int dlclose_libdl2(IntPtr handle);

        [DllImport("libc", EntryPoint = "dlopen")]
        private static extern IntPtr dlopen_libc(string path, int flags);
        [DllImport("libc", EntryPoint = "dlsym")]
        private static extern IntPtr dlsym_libc(IntPtr handle, string symbol);
        [DllImport("libc", EntryPoint = "dlclose")]
        private static extern int dlclose_libc(IntPtr handle);

        private const int RTLD_NOW = 2;

        private static readonly (Func<string, int, IntPtr> Open, Func<IntPtr, string, IntPtr> Sym, Func<IntPtr, int> Close)[] s_dlCandidates =
        {
            (dlopen_libdl,  dlsym_libdl,  dlclose_libdl),
            (dlopen_libdl2, dlsym_libdl2, dlclose_libdl2),
            (dlopen_libc,   dlsym_libc,   dlclose_libc),
        };

        private static int s_dlIndex = -1;

        public static IntPtr Load(string path)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                IntPtr h = LoadLibraryW(path);
                if (h == IntPtr.Zero)
                    throw new DllNotFoundException(
                        "Failed to load '" + path + "': error " + Marshal.GetLastWin32Error());
                return h;
            }

            if (s_dlIndex >= 0)
            {
                IntPtr h = s_dlCandidates[s_dlIndex].Open(path, RTLD_NOW);
                if (h == IntPtr.Zero)
                    throw new DllNotFoundException("Failed to load '" + path + "'");
                return h;
            }

            for (int i = 0; i < s_dlCandidates.Length; i++)
            {
                IntPtr h;
                try
                {
                    h = s_dlCandidates[i].Open(path, RTLD_NOW);
                }
                catch (DllNotFoundException)
                {
                    continue;
                }
                s_dlIndex = i;
                if (h == IntPtr.Zero)
                    throw new DllNotFoundException("Failed to load '" + path + "'");
                return h;
            }
            throw new DllNotFoundException("dlopen unavailable via libdl, libdl.so.2, or libc");
        }

        public static IntPtr GetExport(IntPtr lib, string symbol)
        {
            IntPtr ptr = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? GetProcAddress(lib, symbol)
                : s_dlCandidates[s_dlIndex].Sym(lib, symbol);
            if (ptr == IntPtr.Zero)
                throw new EntryPointNotFoundException("Symbol '" + symbol + "' not found");
            return ptr;
        }

        public static void Free(IntPtr lib)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                FreeLibrary(lib);
            else
                s_dlCandidates[s_dlIndex].Close(lib);
        }
    }
}
