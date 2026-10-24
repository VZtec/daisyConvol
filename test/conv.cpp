// conv.cpp - desktop verification harness for the partitioned
// convolution engine. Builds with the portable reference FFT backend
// (fft_backend_ref.cpp) or, on a PC, with the target's CMSIS-DSP backend
// (fft_backend_cmsis.cpp, see CMakeLists.txt), and compares the engine
// output against a direct / large-FFT reference convolution. Pass criterion:
// relative RMS error below -100 dB over the full output.
//
// Build: see test/CMakeLists.txt (test/all.sh runs everything).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "daisyconvol/conv_engine.h"

namespace daisyconvol
{
// White-box access for the tests that need to fast-forward the counters.
struct ConvEngineTestAccess
{
    // Shifts every block counter by `blocks` and every main-loop stage frame
    // counter by `frames`, as if the engine had run that long on silence.
    // The engine must be idle (main loop caught up, no reset or switch
    // pending) and have seen >= 2 main-loop periods; blocks must be a
    // multiple of 4 main-loop periods and of every time-sliced period, and
    // frames a multiple of 4.
    static void FastForward(ConvEngine& e, uint32_t blocks, uint32_t frames)
    {
        e.blk_ += blocks;
        e.h0_switch_at_ += blocks;
        for(size_t i = 0; i < e.num_sliced_; i++)
            e.sliced_[i].switch_at += blocks;
        if(!e.bg_)
            return;
        e.q2_ += frames;
        e.s2_next_ += frames;
        e.bg_j_ += frames;
        e.bg_sw_frame_ += frames;
        e.frames_produced_.store(e.frames_produced_.load() + frames);
        e.results_ready_.store(e.results_ready_.load() + frames);
    }
    static uint32_t R2(const ConvEngine& e) { return (uint32_t)e.r2_; }
    static uint32_t Rmax(const ConvEngine& e) { return (uint32_t)e.rmax_; }
};
} // namespace daisyconvol

namespace
{

using CE       = daisyconvol::ConvEngine;
using Geometry = daisyconvol::Geometry;
using Access   = daisyconvol::ConvEngineTestAccess;
using Vec      = std::vector<float>;
using DVec     = std::vector<double>;

constexpr double kPi = 3.14159265358979323846;

// Deterministic PRNG (xorshift), uniform in [-1, 1).
struct Rng
{
    uint64_t s = 0x243F6A8885A308D3ull;
    double   Next()
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double)(int64_t)s / 9223372036854775808.0;
    }
    size_t Below(size_t n) { return (size_t)((Next() + 1.0) * 0.5 * n) % n; }
};

Vec Noise(size_t n, Rng& rng, double gain = 1.0)
{
    Vec v(n);
    for(auto& x : v)
        x = (float)(rng.Next() * gain);
    return v;
}

// ---------- reference convolution ----------------------------------------

DVec ConvDirect(const Vec& x, const Vec& h, size_t out_len)
{
    DVec y(out_len, 0.0);
    for(size_t n = 0; n < out_len; n++)
    {
        double       acc  = 0.0;
        const size_t kmax = n < h.size() - 1 ? n : h.size() - 1;
        for(size_t k = 0; k <= kmax; k++)
            if(n - k < x.size())
                acc += (double)h[k] * (double)x[n - k];
        y[n] = acc;
    }
    return y;
}

void RefCFft(DVec& re, DVec& im, bool inverse)
{
    const size_t n = re.size();
    for(size_t i = 1, j = 0; i < n; i++)
    {
        size_t bit = n >> 1;
        for(; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if(i < j)
        {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for(size_t len = 2; len <= n; len <<= 1)
    {
        const double ang = (inverse ? 2.0 : -2.0) * kPi / (double)len;
        const double wr = std::cos(ang), wi = std::sin(ang);
        for(size_t i = 0; i < n; i += len)
        {
            double cr = 1.0, ci = 0.0;
            for(size_t k = 0; k < len / 2; k++)
            {
                const size_t a = i + k, b = i + k + len / 2;
                const double tr = re[b] * cr - im[b] * ci;
                const double ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                const double ncr = cr * wr - ci * wi;
                ci               = cr * wi + ci * wr;
                cr               = ncr;
            }
        }
    }
}

DVec ConvFft(const Vec& x, const Vec& h, size_t out_len)
{
    size_t n = 1;
    while(n < x.size() + h.size())
        n <<= 1;
    DVec xr(n, 0.0), xi(n, 0.0), hr(n, 0.0), hi(n, 0.0);
    for(size_t i = 0; i < x.size(); i++)
        xr[i] = x[i];
    for(size_t i = 0; i < h.size(); i++)
        hr[i] = h[i];
    RefCFft(xr, xi, false);
    RefCFft(hr, hi, false);
    for(size_t i = 0; i < n; i++)
    {
        const double r  = xr[i] * hr[i] - xi[i] * hi[i];
        const double im = xr[i] * hi[i] + xi[i] * hr[i];
        xr[i]           = r;
        xi[i]           = im;
    }
    RefCFft(xr, xi, true);
    DVec y(out_len);
    for(size_t i = 0; i < out_len; i++)
        y[i] = i < n ? xr[i] / (double)n : 0.0;
    return y;
}

DVec Reference(const Vec& x, const Vec& h, size_t out_len)
{
    return (h.size() * out_len <= 4000000) ? ConvDirect(x, h, out_len)
                                           : ConvFft(x, h, out_len);
}

// ---------- harness -------------------------------------------------------

int g_failures = 0;

void Check(bool ok, const char* what)
{
    if(!ok)
    {
        printf("  FAIL: %s\n", what);
        g_failures++;
    }
}

CE::Config MakeConfig(size_t          block,
                      size_t          max_ir,
                      bool            allow_bg,
                      bool            external   = false,
                      const Geometry* custom     = nullptr,
                      size_t          outputs    = 1,
                      bool            external_bg = false)
{
    CE::Config cfg;
    cfg.block_size          = block;
    cfg.max_ir_len          = max_ir;
    cfg.allow_background    = allow_bg;
    cfg.external_ir         = external;
    cfg.geometry            = custom;
    cfg.num_outputs         = outputs;
    cfg.external_background = external_bg;
    return cfg;
}

struct EngineRig
{
    Vec        fast, medium, slow;
    CE         eng;
    CE::Config cfg;
    Geometry   geo;

    // Exact arena sizes: Init must not need more than the sizing functions.
    bool Init(const CE::Config& c)
    {
        cfg = c;
        geo = CE::ConfigGeometry(c);
        fast.assign(CE::FastFloats(c) + 1, 0.0f); // +1: never 0-sized
        medium.assign(CE::MediumFloats(c) + 1, 0.0f);
        slow.assign(CE::SlowFloats(c) + 1, 0.0f);
        daisyconvol::Arena fa{fast.data(), fast.size() - 1, 0};
        daisyconvol::Arena me{medium.data(), medium.size() - 1, 0};
        daisyconvol::Arena sl{slow.data(), slow.size() - 1, 0};
        return eng.Init(c, fa, me, sl);
    }
    bool Init(size_t          block,
              size_t          max_ir,
              bool            allow_bg,
              bool            external = false,
              const Geometry* custom   = nullptr)
    {
        return Init(MakeConfig(block, max_ir, allow_bg, external, custom));
    }
    size_t Outputs() const { return cfg.num_outputs; }
};

double ErrDb(const float* y, const double* ref, size_t n)
{
    double se = 0.0, sr = 0.0;
    for(size_t i = 0; i < n; i++)
    {
        const double e = (double)y[i] - ref[i];
        se += e * e;
        sr += ref[i] * ref[i];
    }
    if(sr == 0.0)
        return se == 0.0 ? -999.0 : 0.0;
    return 10.0 * std::log10(se / sr);
}

double ErrDb(const Vec& y, const DVec& ref, size_t begin, size_t end)
{
    return ErrDb(y.data() + begin, ref.data() + begin, end - begin);
}

// Output buffers of a multi-output run.
struct Outs
{
    std::vector<Vec> y;
    Outs(size_t outputs, size_t n) : y(outputs, Vec(n, 0.0f)) {}
    void Ptrs(size_t off, float** p)
    {
        for(size_t c = 0; c < y.size(); c++)
            p[c] = y[c].data() + off;
    }
};

// Runs the engine over x, calling DoBackgroundWork() after every block
// except inside [gap_begin_blk, gap_end_blk) to simulate a stalled main loop.
Outs RunEngine(CE&        eng,
               const Vec& x,
               size_t     gap_begin_blk = 0,
               size_t     gap_end_blk   = 0)
{
    const size_t B       = eng.block_size();
    const size_t nblocks = x.size() / B;
    Outs         o(eng.num_outputs(), nblocks * B);
    for(size_t blk = 0; blk < nblocks; blk++)
    {
        float* p[CE::kMaxOutputs];
        o.Ptrs(blk * B, p);
        eng.ProcessBlock(x.data() + blk * B, p, B);
        const bool stalled = blk >= gap_begin_blk && blk < gap_end_blk;
        if(!stalled)
            eng.DoBackgroundWork();
    }
    return o;
}

void PrintGeometry(const Geometry& g)
{
    printf("   ");
    for(size_t s = 0; s < g.num_stages; s++)
        printf(" %zu x %zu%s",
               g.count[s],
               g.part[s],
               g.IsBackground(s) ? " (main loop)" : "");
    printf("  -> %zu samples\n", g.Length());
}

// IR lengths probing every stage boundary of g, up to max_ir.
std::vector<size_t> BoundaryLengths(const Geometry& g, size_t max_ir)
{
    std::vector<size_t> v = {1, g.block - 3};
    for(size_t s = 1; s < g.num_stages; s++)
    {
        const size_t o = g.Offset(s);
        v.push_back(o);
        v.push_back(o + 1);
        v.push_back(o + g.part[s] / 2 + 7); // partial first partition
    }
    v.push_back(max_ir);
    std::vector<size_t> out;
    for(size_t len : v)
        if(len >= 1 && len <= max_ir && (out.empty() || out.back() < len))
            out.push_back(len);
    return out;
}

// One IR per output: output c gets lens[c] samples.
bool TestIrLengths(EngineRig& rig, const std::vector<size_t>& lens, Rng& rng)
{
    const size_t     B = rig.eng.block_size();
    const size_t     C = rig.Outputs();
    std::vector<Vec> irs;
    size_t           longest = 0;
    for(size_t c = 0; c < C; c++)
    {
        irs.push_back(Noise(lens[c], rng, 0.5));
        longest = std::max(longest, lens[c]);
    }
    // input: enough to exercise the full tail plus slack
    const size_t xlen
        = ((longest + 3 * rig.geo.part[rig.geo.num_stages - 1] + 4096 + B - 1)
           / B)
          * B;
    const Vec x = Noise(xlen, rng);

    const float* ptrs[CE::kMaxOutputs];
    for(size_t c = 0; c < C; c++)
        ptrs[c] = irs[c].data();
    if(!rig.eng.SetIR(ptrs, lens.data()))
    {
        Check(false, "SetIR");
        return false;
    }
    const Outs       o      = RunEngine(rig.eng, x);
    double           worst  = -999.0;
    for(size_t c = 0; c < C; c++)
    {
        const DVec ref = Reference(x, irs[c], o.y[c].size());
        worst          = std::max(worst, ErrDb(o.y[c], ref, 0, ref.size()));
    }
    const daisyconvol::Perf pf = rig.eng.perf();
    const bool ok = worst < -100.0 && pf.s2_underruns == 0;
    if(!ok)
        printf("  ir_len=%7zu  err=%8.2f dB  underruns=%u\n",
               lens[0],
               worst,
               pf.s2_underruns);
    Check(worst < -100.0, "error < -100 dB");
    Check(pf.s2_underruns == 0, "no main-loop stage underruns");
    return ok;
}

// Every stage boundary for one configuration (all outputs get the same
// length; with several outputs each one gets its own IR and output c also
// gets a shorter one, so outputs reaching different stages are mixed).
void TestConfig(const CE::Config& cfg, Rng& rng, const char* label)
{
    EngineRig rig;
    if(!rig.Init(cfg))
    {
        printf("  B=%zu max_ir=%zu %s: FAIL engine Init\n",
               cfg.block_size,
               cfg.max_ir_len,
               label);
        g_failures++;
        return;
    }
    const size_t C = cfg.num_outputs;
    size_t       n = 0, pass = 0;
    for(size_t len : BoundaryLengths(rig.geo, cfg.max_ir_len))
    {
        std::vector<size_t> lens(C, len);
        for(size_t c = 1; c < C; c++)
            lens[c] = std::max<size_t>(1, len / (2 * c + 1));
        n++;
        pass += TestIrLengths(rig, lens, rng);
    }
    printf("  B=%4zu max_ir=%6zu %-10s: %zu/%zu IR lengths OK  fast=%zu "
           "medium=%zu slow=%zu KB\n",
           cfg.block_size,
           cfg.max_ir_len,
           label,
           pass,
           n,
           CE::FastFloats(cfg) * 4 / 1024,
           CE::MediumFloats(cfg) * 4 / 1024,
           CE::SlowFloats(cfg) * 4 / 1024);
    PrintGeometry(rig.geo);

    Vec too_long(cfg.max_ir_len + 1, 0.1f);
    if(C == 1)
    {
        Check(!rig.eng.SetIR(too_long.data(), too_long.size()),
              "reject IR longer than max_ir");
        Check(!rig.eng.SetIR(nullptr, 100), "reject null IR");
        Vec bad(100, 0.1f);
        bad[37] = std::numeric_limits<float>::quiet_NaN();
        Check(!rig.eng.SetIR(bad.data(), bad.size()), "reject NaN in IR");
        bad[37] = std::numeric_limits<float>::infinity();
        Check(!rig.eng.SetIR(bad.data(), bad.size()), "reject inf in IR");
    }
    else
    {
        Vec one(10, 0.1f);
        Check(!rig.eng.SetIR(one.data(), one.size()),
              "single-IR SetIR rejected with several outputs");
    }
}

void TestConfig(size_t          block,
                size_t          max_ir,
                bool            allow_bg,
                Rng&            rng,
                const Geometry* custom = nullptr)
{
    TestConfig(MakeConfig(block, max_ir, allow_bg, false, custom),
               rng,
               custom ? "custom" : (allow_bg ? "bg" : "callback"));
}

// FFT sizes each default geometry needs must be in the set daisyconvol.mk
// enables for that block size (and in the reduced set for
// DAISYCONVOL_MAX_IR <= 4096).
void TestGeometries()
{
    struct MkSet
    {
        size_t b;
        size_t ffts[4];
        size_t cab[4]; // DAISYCONVOL_MAX_IR <= 4096
    };
    const MkSet mk[] = {{16, {32, 256, 4096}, {32, 256}},
                        {32, {64, 512, 4096}, {64, 512}},
                        {64, {128, 1024, 4096}, {128, 1024}},
                        {128, {256, 2048, 4096}, {256, 2048}},
                        {256, {512, 4096}, {512}},
                        {512, {1024, 4096}, {1024}},
                        {1024, {2048, 4096}, {2048}},
                        {2048, {4096}, {4096}}};
    const size_t lens[]
        = {1, 100, 1000, 4095, 4096, 4097, 20000, 48000, 96000, 144000, 480000};
    for(const MkSet& m : mk)
    {
        for(size_t len : lens)
        {
            for(bool bg : {true, false})
            {
                const Geometry g = CE::MakeGeometry(m.b, len, bg);
                Check(g.Valid(), "default geometry valid");
                Check(g.Length() >= len, "default geometry covers max_ir");
                for(size_t s = 0; s < g.num_stages; s++)
                {
                    bool in_mk = false, in_cab = false;
                    for(size_t f : m.ffts)
                        in_mk |= f == 2 * g.part[s];
                    for(size_t f : m.cab)
                        in_cab |= f == 2 * g.part[s];
                    if(!in_mk || (len <= 4096 && !in_cab))
                        printf("  B=%zu len=%zu: FFT %zu not in "
                               "daisyconvol.mk\n",
                               m.b,
                               len,
                               2 * g.part[s]);
                    Check(in_mk, "FFT size enabled by daisyconvol.mk");
                    Check(len > 4096 || in_cab,
                          "FFT size enabled with DAISYCONVOL_MAX_IR <= 4096");
                    Check(bg || !g.IsBackground(s),
                          "no main-loop stage without allow_background");
                }
            }
        }
    }

    // The historical geometries are unchanged.
    const Geometry g32 = CE::MakeGeometry(32, 144000);
    Check(g32.num_stages == 3 && g32.count[0] == 16 && g32.part[1] == 256
              && g32.count[1] == 14 && g32.part[2] == 2048 && g32.background,
          "B=32 default: 16x32, 14x256, 2048 in main loop");
    const Geometry g16 = CE::MakeGeometry(16, 72000);
    Check(g16.num_stages == 3 && g16.count[0] == 16 && g16.part[1] == 128
              && g16.count[1] == 30 && g16.part[2] == 2048 && g16.background,
          "B=16 default: 16x16, 30x128, 2048 in main loop");
    const Geometry cab = CE::MakeGeometry(32, 4096);
    Check(cab.num_stages == 2 && !cab.background && cab.Length() == 4096,
          "B=32, 4096: stages 0+1 only");

    // A smaller CPU budget per sample (e.g. 96 kHz) moves the 256-sample
    // stage out of the callback at B = 32.
    const Geometry g96 = CE::MakeGeometry(32, 96000, true, 5000);
    Check(g96.Valid() && g96.num_stages == 2 && g96.part[1] == 2048
              && g96.background,
          "cycles_per_sample = 5000: stage 1 goes to the main loop");

    // Invalid inputs.
    Check(!CE::MakeGeometry(8, 1000).Valid(), "B=8 unsupported");
    Check(!CE::MakeGeometry(48, 1000).Valid(), "B=48 unsupported");
    Geometry bad = CE::MakeGeometry(32, 4096);
    bad.count[0] = 15;
    Check(!bad.Valid(), "broken stage offsets rejected");
    Geometry one = CE::MakeGeometry(32, 400);
    one.background = true;
    Check(one.num_stages == 1 && !one.Valid(),
          "main-loop flag on a single-stage geometry rejected");
    Check(CE::SlowFloats(one) == 0, "no slow arena for an invalid geometry");
    EngineRig rig;
    Check(!rig.Init(32, 4096, true, false, &bad), "Init rejects bad geometry");

    // Config-based sizing matches the geometry-based one for one output,
    // and the plain (b, max_ir) helpers match a default Config.
    for(size_t b : {16u, 32u, 128u, 1024u})
    {
        for(size_t len : {4096u, 96000u})
        {
            CE::Config c   = MakeConfig(b, len, true);
            Geometry   g   = CE::ConfigGeometry(c);
            const bool eq  = CE::FastFloats(c) == CE::FastFloats(g)
                            && CE::MediumFloats(c) == CE::MediumFloats(g)
                            && CE::SlowFloats(c) == CE::SlowFloats(g);
            const bool leg = CE::FastFloats(b, len) == CE::FastFloats(c)
                             && CE::MediumFloats(b, len) == CE::MediumFloats(c)
                             && CE::SlowFloats(b, len) == CE::SlowFloats(c);
            Check(eq, "Config sizes == Geometry sizes (1 output)");
            Check(leg, "FastFloats(b, max_ir) == FastFloats(Config)");
            c.external_ir = true;
            Check(CE::SpectraFloats(c) == CE::SpectraFloats(b, len),
                  "SpectraFloats(Config) == SpectraFloats(b, max_ir)");
            Check(CE::PrepareScratchFloats(c)
                      == CE::PrepareScratchFloats(b, len),
                  "PrepareScratchFloats(Config) == (b, max_ir)");
        }
    }
    CE::Config bad_cfg = MakeConfig(32, 4096, true);
    bad_cfg.num_outputs = 0;
    Check(CE::FastFloats(bad_cfg) == 0 && !rig.Init(bad_cfg),
          "num_outputs = 0 rejected");
    bad_cfg.num_outputs = CE::kMaxOutputs + 1;
    Check(CE::FastFloats(bad_cfg) == 0 && !rig.Init(bad_cfg),
          "num_outputs > kMaxOutputs rejected");

    // Relocation buffer for the B = 32 reverb set (FFT 64, 512, 4096): per
    // N-point FFT, 2N floats of twiddles and N/2 uint16 bin positions.
    Check(CE::FftTableBytes(MakeConfig(32, 144000, true))
              == 9 * (64 + 512 + 4096),
          "FftTableBytes(Config) for B = 32");
    Check(CE::FftTableBytes(MakeConfig(32, 4096, true)) == 576 + 4608,
          "FftTableBytes(Config) for a B = 32 cabinet");

    // External IR mode: a main-loop stage only when asked for.
    Check(rig.Init(32, 100000, true, true) && !rig.eng.geometry().background,
          "external IR: default geometry stays in the callback");
    Check(rig.Init(MakeConfig(32, 100000, true, true, nullptr, 1, true))
              && rig.eng.geometry().background,
          "external IR + external_background: main-loop stage");
    const Geometry with_bg = CE::MakeGeometry(32, 100000, true);
    Check(rig.Init(32, 100000, true, true, &with_bg),
          "external IR accepts a custom main-loop stage");
    printf("  geometry checks done\n");
}

void TestUnderrunRecovery(size_t block, Rng& rng)
{
    EngineRig    rig;
    const size_t max_ir = 96000;
    if(!rig.Init(block, max_ir, true) || !rig.geo.background)
    {
        Check(false, "underrun test Init");
        return;
    }
    const size_t s2 = rig.geo.num_stages - 1;
    const size_t P2 = rig.geo.part[s2];
    const size_t R2 = P2 / block;

    const Vec    ir      = Noise(rig.geo.Offset(s2) + 8 * P2 + 123, rng, 0.5);
    const size_t nblocks = 60 * R2;
    const Vec    x       = Noise(nblocks * block, rng);
    Check(rig.eng.SetIR(ir.data(), ir.size()), "SetIR");

    // Stall the main loop for ~5 periods of the main-loop stage.
    const size_t gap_b = 20 * R2 + 7, gap_e = gap_b + 5 * R2;
    const Outs   o     = RunEngine(rig.eng, x, gap_b, gap_e);
    const Vec&   y     = o.y[0];
    const DVec   ref   = ConvFft(x, ir, y.size());

    const daisyconvol::Perf pf     = rig.eng.perf();
    const double            pre    = ErrDb(y, ref, 0, gap_b * block);
    const size_t settle = (gap_e + ir.size() / block + 3 * R2) * block;
    const double post
        = settle < y.size() ? ErrDb(y, ref, settle, y.size()) : -999.0;
    bool finite = true;
    for(float v : y)
        finite &= std::isfinite(v);
    printf("  B=%zu stall: underruns=%u dropped=%u  pre %7.2f dB  post "
           "%7.2f dB\n",
           block,
           pf.s2_underruns,
           pf.s2_dropped_inputs,
           pre,
           post);
    Check(pf.s2_underruns > 0, "stall detected as underruns");
    Check(pre < -100.0, "pre-stall error < -100 dB");
    Check(post < -100.0, "post-stall recovery < -100 dB");
    Check(finite, "output finite throughout stall");
}

// ---------- external IR switching ---------------------------------------

// Latch model of LatchIR: switches latch every rl blocks while none is in
// progress and complete lead blocks later.
struct SwapTiming
{
    size_t rl = 1, lead = 1;
    explicit SwapTiming(const Geometry& g)
    {
        size_t rmax = 1;
        for(size_t s = 1; s < g.num_stages; s++)
            if(!g.IsBackground(s) && g.Ratio(s) > rmax)
                rmax = g.Ratio(s);
        const size_t r2 = g.background ? g.Ratio(g.num_stages - 1) : 0;
        rl              = g.background ? r2 : rmax;
        lead            = g.background ? 2 * r2 : rmax;
    }
};

// A prepared IR set (one IR per output) for external mode.
struct Prepared
{
    std::vector<Vec> irs; // per output
    Vec              storage;
    CE::IrSpectra    spec;
};

bool Prepare(EngineRig& rig, Prepared& p, const std::vector<size_t>& lens,
             Rng& rng)
{
    const size_t C = rig.Outputs();
    p.irs.clear();
    const float* ptrs[CE::kMaxOutputs];
    for(size_t c = 0; c < C; c++)
    {
        p.irs.push_back(Noise(lens[c], rng, 0.5));
        ptrs[c] = p.irs[c].data();
    }
    p.storage.assign(rig.eng.spectra_floats(), 0.0f);
    Vec scratch(rig.eng.prepare_scratch_floats());
    const bool ok = rig.eng.PrepareIR(ptrs,
                                      lens.data(),
                                      p.storage.data(),
                                      p.storage.size(),
                                      scratch.data(),
                                      scratch.size(),
                                      &p.spec);
    // What the engine actually uses: truncated to the geometry.
    for(auto& ir : p.irs)
        if(ir.size() > rig.geo.Length())
            ir.resize(rig.geo.Length());
    return ok;
}

// External IR mode: IRs prepared up front, switched while "audio" runs.
// Expected output: a hard cut between the exact convolutions of the old and
// the new IR. A SwapIR() before block k is latched at the first block
// q >= k that is a multiple of rl with no switch in progress; the output is
// the new IR's from block q + lead on (see SwapTiming). SwapPending() and
// ActiveIR() must follow the same model.
void TestExternalSwap(const CE::Config& cfg, Rng& rng)
{
    EngineRig rig;
    if(!rig.Init(cfg))
    {
        Check(false, "external Init");
        return;
    }
    const Geometry&  g = rig.geo;
    const SwapTiming tm(g);
    const size_t     B = cfg.block_size;
    const size_t     C = cfg.num_outputs;
    const size_t     L = g.Length();
    Check(!rig.eng.SetIR(nullptr, 1), "SetIR rejected in external mode");

    // IR sets: full coverage, stage-0 only, partial, one sample, too long.
    // With several outputs each output gets a different length.
    const size_t base[]
        = {L, g.part[0] * g.count[0] - 5, L / 3 + 17, 1, L + 999};
    const size_t          nir = sizeof(base) / sizeof(base[0]);
    std::vector<Prepared> ir(nir);
    for(size_t i = 0; i < nir; i++)
    {
        std::vector<size_t> lens(C);
        for(size_t c = 0; c < C; c++)
            lens[c] = c == 0 ? base[i] : std::max<size_t>(1, base[i] / (c + 2));
        Check(Prepare(rig, ir[i], lens, rng), "PrepareIR");
    }
    Check(ir[4].spec.len == L, "PrepareIR truncates to the geometry length");
    {
        // storage / scratch too small, NaN in an IR
        Prepared   tmp;
        Vec        small(rig.eng.spectra_floats() - 1);
        Vec        scratch(rig.eng.prepare_scratch_floats());
        Vec        nan_ir(50, 0.1f);
        nan_ir[3]         = std::numeric_limits<float>::quiet_NaN();
        const float* p[CE::kMaxOutputs];
        size_t       lens[CE::kMaxOutputs];
        for(size_t c = 0; c < C; c++)
        {
            p[c]    = ir[0].irs[c].data();
            lens[c] = ir[0].irs[c].size();
        }
        Check(!rig.eng.PrepareIR(p, lens, small.data(), small.size(),
                                 scratch.data(), scratch.size(), &tmp.spec),
              "PrepareIR rejects small storage");
        Vec storage(rig.eng.spectra_floats());
        Check(!rig.eng.PrepareIR(p, lens, storage.data(), storage.size(),
                                 scratch.data(), scratch.size() - 1,
                                 &tmp.spec),
              "PrepareIR rejects small scratch");
        p[C - 1]    = nan_ir.data();
        lens[C - 1] = nan_ir.size();
        Check(!rig.eng.PrepareIR(p, lens, storage.data(), storage.size(),
                                 scratch.data(), scratch.size(), &tmp.spec),
              "PrepareIR rejects NaN");
    }

    // Schedule in units of rl so every geometry gets the same shape: an
    // unaligned swap, one on a boundary, two inside one latch window (last
    // wins), back-to-back latches, silence.
    const size_t rl   = tm.rl;
    const size_t nper = 40;
    struct Ev
    {
        size_t blk;
        int    ir;
    };
    const Ev evs[] = {{0, 0},
                      {2 * rl + rl / 2, 1},
                      {6 * rl, 2},
                      {10 * rl + 1, 3},
                      {10 * rl + 2, 0}, // overrides 3 before it latches
                      {16 * rl, 1},
                      {17 * rl, 2}, // latched right after 1 (if possible)
                      {22 * rl + 3, -1},
                      {26 * rl, 4},
                      {33 * rl + 1, 2}};
    const size_t nev     = sizeof(evs) / sizeof(evs[0]);
    const size_t nblocks = nper * rl + 3 * tm.lead + 64;

    const Vec x = Noise(nblocks * B, rng);
    std::vector<std::vector<DVec>> ref(nir, std::vector<DVec>(C));
    for(size_t i = 0; i < nir; i++)
        for(size_t c = 0; c < C; c++)
            ref[i][c] = Reference(x, ir[i].irs[c], x.size());

    // Model: per block, which IR is live, and SwapPending() after it.
    std::vector<int>  live(nblocks, -1);
    std::vector<char> pending_after(nblocks, 0);
    std::vector<int>  active_after(nblocks, -1);
    {
        int    pending = -2; // -2 = none
        size_t e       = 0;
        int    cur     = -1;
        int    due_ir  = -2;
        size_t due_at  = 0;
        for(size_t k = 0; k < nblocks; k++)
        {
            while(e < nev && evs[e].blk == k)
                pending = evs[e++].ir;
            if(due_ir != -2 && k == due_at)
            {
                cur    = due_ir;
                due_ir = -2;
            }
            if(k % rl == 0 && pending != -2 && due_ir == -2)
            {
                due_ir  = pending;
                due_at  = k + tm.lead;
                pending = -2;
            }
            live[k]          = cur;
            pending_after[k] = pending != -2 || due_ir != -2;
            active_after[k]  = cur;
        }
    }

    Outs   o(C, x.size());
    size_t e             = 0;
    bool   pend_ok       = true;
    size_t pend_bad_blk  = 0;
    for(size_t k = 0; k < nblocks; k++)
    {
        while(e < nev && evs[e].blk == k)
        {
            rig.eng.SwapIR(evs[e].ir < 0 ? nullptr : &ir[evs[e].ir].spec);
            e++;
        }
        float* p[CE::kMaxOutputs];
        o.Ptrs(k * B, p);
        rig.eng.ProcessBlock(x.data() + k * B, p, B);
        rig.eng.DoBackgroundWork();
        const CE::IrSpectra* want
            = active_after[k] < 0 ? nullptr : &ir[active_after[k]].spec;
        if(pend_ok
           && (rig.eng.SwapPending() != (pending_after[k] != 0)
               || (!pending_after[k] && rig.eng.ActiveIR() != want)))
        {
            pend_ok      = false;
            pend_bad_blk = k;
        }
    }
    if(!pend_ok)
        printf("  SwapPending/ActiveIR mismatch at block %zu\n", pend_bad_blk);

    double se = 0.0, sr = 0.0, worst = -999.0;
    for(size_t k = 0; k < nblocks; k++)
    {
        double bse = 0.0, bsr = 0.0;
        for(size_t c = 0; c < C; c++)
        {
            for(size_t i = k * B; i < (k + 1) * B; i++)
            {
                const double r  = live[k] < 0 ? 0.0 : ref[live[k]][c][i];
                const double er = (double)o.y[c][i] - r;
                bse += er * er;
                bsr += r * r;
            }
        }
        se += bse;
        sr += bsr;
        // Per-block error relative to the running signal level, so a wrong
        // IR in any single block shows up.
        const double lvl = sr > 0 ? sr / (double)((k + 1) * B * C) : 1.0;
        const double db  = 10.0 * std::log10(bse / (B * C * lvl) + 1e-30);
        if(db > worst)
            worst = db;
    }
    const double err = 10.0 * std::log10(se / sr);

    // Reset() with a swap not latched yet: applies at once.
    rig.eng.SwapIR(&ir[2].spec);
    rig.eng.Reset();
    const size_t nb2 = std::max<size_t>(8 * tm.lead, 64) + L / B
                       + 3 * g.part[g.num_stages - 1] / B;
    const Vec x2 = Noise(nb2 * B, rng);
    Outs      o2 = RunEngine(rig.eng, x2);
    double    err2 = -999.0;
    for(size_t c = 0; c < C; c++)
        err2 = std::max(
            err2,
            ErrDb(o2.y[c], Reference(x2, ir[2].irs[c], x2.size()), 0,
                  x2.size()));
    Check(!rig.eng.SwapPending() && rig.eng.ActiveIR() == &ir[2].spec,
          "Reset applies a pending swap");

    printf("  B=%4zu L=%6zu stages=%zu%s outs=%zu rl=%3zu lead=%3zu: swap "
           "err %7.2f dB, worst block %7.2f dB, swap+reset %7.2f dB\n",
           B,
           L,
           g.num_stages,
           g.background ? " (bg)" : "     ",
           C,
           tm.rl,
           tm.lead,
           err,
           worst,
           err2);
    Check(err < -100.0, "external swap error < -100 dB");
    Check(worst < -100.0, "external swap worst block < -100 dB");
    Check(err2 < -100.0, "SwapIR then Reset error < -100 dB");
    Check(pend_ok, "SwapPending() / ActiveIR() follow the latch model");
    Check(rig.eng.perf().s2_underruns == 0, "no underruns while swapping");
}

// Once SwapPending() is false, the storage of every IR but ActiveIR() is
// free: poison it with NaN and check the output stays exact.
void TestStorageReuse(const CE::Config& cfg, Rng& rng)
{
    EngineRig rig;
    if(!rig.Init(cfg))
    {
        Check(false, "storage reuse Init");
        return;
    }
    const size_t     B = cfg.block_size;
    const size_t     C = cfg.num_outputs;
    const size_t     L = rig.geo.Length();
    std::vector<size_t> lens(C, L);
    Prepared a, b;
    Check(Prepare(rig, a, lens, rng) && Prepare(rig, b, lens, rng),
          "PrepareIR");

    const size_t nb = 3 * L / B + 16 * SwapTiming(rig.geo).lead + 64;
    const Vec    x  = Noise(nb * B, rng);
    Outs         o(C, x.size());
    rig.eng.SwapIR(&a.spec);
    size_t switched_at = 0;
    bool   poisoned    = false;
    for(size_t k = 0; k < nb; k++)
    {
        if(k == nb / 3)
            rig.eng.SwapIR(&b.spec);
        float* p[CE::kMaxOutputs];
        o.Ptrs(k * B, p);
        rig.eng.ProcessBlock(x.data() + k * B, p, B);
        rig.eng.DoBackgroundWork();
        if(k >= nb / 3 && !poisoned && !rig.eng.SwapPending())
        {
            Check(rig.eng.ActiveIR() == &b.spec, "ActiveIR after the swap");
            std::fill(a.storage.begin(),
                      a.storage.end(),
                      std::numeric_limits<float>::quiet_NaN());
            poisoned    = true;
            switched_at = k + 1;
        }
    }
    Check(poisoned, "swap completed");
    // From the switch on, the output is b's convolution of the whole input.
    double worst = -999.0;
    bool   finite = true;
    for(size_t c = 0; c < C; c++)
    {
        const DVec ref = Reference(x, b.irs[c], x.size());
        worst = std::max(worst, ErrDb(o.y[c], ref, switched_at * B, x.size()));
        for(float v : o.y[c])
            finite &= std::isfinite(v);
    }
    printf("  B=%4zu outs=%zu%s: old IR storage poisoned after SwapPending() "
           "went false: %7.2f dB\n",
           B,
           C,
           rig.geo.background ? " (bg)" : "",
           worst);
    Check(finite && worst < -100.0, "old IR storage unused after the swap");
}

// A hand-made 4-stage geometry: 8 x 16, 6 x 64, 6 x 256, then 1024s.
Geometry Custom4(size_t last_count, bool bg)
{
    Geometry g;
    g.block          = 16;
    g.num_stages     = 4;
    const size_t p[] = {16, 64, 256, 1024};
    const size_t c[] = {8, 6, 6, last_count};
    for(size_t s = 0; s < 4; s++)
    {
        g.part[s]  = p[s];
        g.count[s] = c[s];
    }
    g.background = bg;
    return g;
}

// ---------- counters, reset handshake, bounded steps ----------------------

// The 32-bit block counter wraps after 2^32 callbacks (33 days at B = 32);
// the main-loop stage frame counter after 2^32 frames. Fast-forward both to
// just before their wraps and check the output stays exact across them.
void TestWrap(const CE::Config& cfg, Rng& rng)
{
    EngineRig rig;
    if(!rig.Init(cfg))
    {
        Check(false, "wrap Init");
        return;
    }
    const size_t   B  = cfg.block_size;
    const size_t   L  = std::min<size_t>(cfg.max_ir_len, rig.geo.Length());
    const uint32_t R2 = rig.geo.background ? Access::R2(rig.eng) : 1;
    const uint32_t U  = std::max(R2, Access::Rmax(rig.eng)); // every period
    Prepared       a, b;
    Vec            ir;
    if(cfg.external_ir)
    {
        std::vector<size_t> lens(cfg.num_outputs, L);
        Check(Prepare(rig, a, lens, rng) && Prepare(rig, b, lens, rng),
              "PrepareIR");
        rig.eng.SwapIR(&a.spec);
    }
    else
    {
        ir = Noise(L, rng, 0.3);
        Check(rig.eng.SetIR(ir.data(), ir.size()), "SetIR");
    }
    // Warm up on silence (the external IR switch completes meanwhile), then
    // put the block counter 80 U blocks and the frame counter 48 frames
    // before their wraps.
    const size_t warm = 16 * U;
    {
        const Vec zeros(B, 0.0f);
        Outs      junk(cfg.num_outputs, B);
        for(size_t k = 0; k < warm; k++)
        {
            float* p[CE::kMaxOutputs];
            junk.Ptrs(0, p);
            rig.eng.ProcessBlock(zeros.data(), p, B);
            rig.eng.DoBackgroundWork();
        }
        Check(!rig.eng.SwapPending(), "wrap warm-up: IR in place");
    }
    const uint32_t to_wrap = 80u * U;
    Access::FastForward(rig.eng,
                        (0u - to_wrap) - (uint32_t)warm,
                        (0u - 48u) - (uint32_t)(warm / R2));

    const size_t nb     = 120 * U + L / B + 64;
    const size_t swap_k = to_wrap - 3; // latched right at the block wrap
    const Vec    x      = Noise(nb * B, rng);
    Outs         o(cfg.num_outputs, x.size());
    for(size_t k = 0; k < nb; k++)
    {
        if(cfg.external_ir && k == swap_k)
            rig.eng.SwapIR(&b.spec); // a switch across the block wrap
        float* p[CE::kMaxOutputs];
        o.Ptrs(k * B, p);
        rig.eng.ProcessBlock(x.data() + k * B, p, B);
        rig.eng.DoBackgroundWork();
    }
    double err = -999.0;
    if(cfg.external_ir)
    {
        // Model the single switch: latched at the first rl boundary at or
        // after swap_k (the counters are multiples of rl here), done lead
        // blocks later.
        const SwapTiming tm(rig.geo);
        const size_t     q = (swap_k + tm.rl - 1) / tm.rl * tm.rl;
        const size_t     t = (q + tm.lead) * B;
        for(size_t c = 0; c < cfg.num_outputs; c++)
        {
            const DVec ra = Reference(x, a.irs[c], x.size());
            const DVec rb = Reference(x, b.irs[c], x.size());
            err = std::max(err, ErrDb(o.y[c], ra, 0, t));
            err = std::max(err, ErrDb(o.y[c], rb, t, x.size()));
        }
    }
    else
        err = ErrDb(o.y[0], Reference(x, ir, x.size()), 0, x.size());
    const daisyconvol::Perf pf = rig.eng.perf();
    printf("  B=%4zu%s%s: across the counter wraps %8.2f dB, underruns %u\n",
           B,
           rig.geo.background ? " (bg)" : "     ",
           cfg.external_ir ? " external" : "         ",
           err,
           pf.s2_underruns);
    Check(err < -100.0, "exact across the 32-bit counter wraps");
    Check(pf.s2_underruns == 0, "no underruns across the wraps");
}

// Simulated audio interrupt inside the main loop: the get_cycles hook runs
// while DoBackgroundStep is in the middle of a frame.
struct IsrSim
{
    CE*       eng        = nullptr;
    bool      in_bg      = false;
    int       calls      = 0;     // get_cycles calls in this bg step
    int       fire_at    = 0;     // fire on this call of a step (0: never)
    long      steps_left = -1;    // fire once this many steps have passed
    bool      reset      = false; // the simulated interrupt calls Reset()
    const float* isr_in  = nullptr; // ...or runs a ProcessBlock
    float**   isr_out    = nullptr;
    int       fired      = 0;
    uint32_t  clock      = 0;
};
IsrSim g_isr;

uint32_t HookCycles()
{
    IsrSim& s = g_isr;
    if(s.in_bg && s.fire_at && ++s.calls == s.fire_at && s.steps_left == 0)
    {
        s.steps_left = -1;
        s.fired++;
        s.in_bg = false; // no nesting
        if(s.reset)
            s.eng->Reset();
        if(s.isr_in)
            s.eng->ProcessBlock(s.isr_in, s.isr_out, s.eng->block_size());
        s.in_bg = true;
    }
    s.clock += 10;
    return s.clock;
}

// DoBackgroundStep() with the hook armed; returns its result.
bool BgStep(CE& eng)
{
    g_isr.in_bg = true;
    g_isr.calls = 0;
    const bool more = eng.DoBackgroundStep();
    g_isr.in_bg = false;
    if(g_isr.steps_left > 0)
        g_isr.steps_left--;
    return more;
}

void BgWork(CE& eng)
{
    while(BgStep(eng)) {}
}

// Reset() from the audio callback while the main loop is in the middle of a
// frame (between any two steps, or at the end of one): afterwards the output
// must be the exact convolution of the post-reset input.
void TestResetInsideBg(const CE::Config& cfg_in, Rng& rng)
{
    CE::Config cfg = cfg_in;
    cfg.get_cycles = HookCycles;
    const size_t C = cfg.num_outputs;
    double       worst_err = -999.0;
    int          fired     = 0;
    for(long at_step = 0; at_step < 12; at_step++)
    {
        EngineRig rig;
        if(!rig.Init(cfg))
        {
            Check(false, "reset test Init");
            return;
        }
        g_isr     = IsrSim();
        g_isr.eng = &rig.eng;
        const size_t   B  = cfg.block_size;
        const size_t   L  = std::min<size_t>(cfg.max_ir_len, rig.geo.Length());
        const uint32_t R2 = Access::R2(rig.eng);
        std::vector<Vec> irs;
        Prepared         a, b;
        if(cfg.external_ir)
        {
            std::vector<size_t> lens(C, L);
            Check(Prepare(rig, a, lens, rng) && Prepare(rig, b, lens, rng),
                  "PrepareIR");
            rig.eng.SwapIR(&a.spec);
            rig.eng.Reset();
        }
        else
        {
            const float* p[CE::kMaxOutputs];
            size_t       lens[CE::kMaxOutputs];
            for(size_t c = 0; c < C; c++)
            {
                irs.push_back(Noise(L, rng, 0.3));
                p[c]    = irs[c].data();
                lens[c] = L;
            }
            Check(rig.eng.SetIR(p, lens), "SetIR");
        }

        // Phase 1: some audio; then Reset() lands inside a frame's steps.
        const size_t n1 = 37 * R2;
        const Vec    x1 = Noise(n1 * B, rng);
        Outs         o1(C, x1.size());
        for(size_t k = 0; k < n1; k++)
        {
            float* p[CE::kMaxOutputs];
            o1.Ptrs(k * B, p);
            rig.eng.ProcessBlock(x1.data() + k * B, p, B);
            if(k + 1 == n1)
            {
                if(cfg.external_ir)
                    rig.eng.SwapIR(&b.spec); // pending at the reset
                g_isr.fire_at    = 2;        // end of the step
                g_isr.steps_left = at_step;
                g_isr.reset      = true;
            }
            if(k + 1 < n1)
                BgWork(rig.eng);
        }
        // The frame completed by the last block is processed step by step;
        // the reset fires at the end of step at_step (or never, if the
        // frame has fewer steps: then it is a plain reset).
        BgWork(rig.eng);
        if(!g_isr.fired)
            rig.eng.Reset();
        fired += g_isr.fired;
        g_isr.fire_at = 0;

        // Phase 2: fresh input; expected output = conv(x2, IR in effect).
        const size_t n2 = 40 * R2 + L / B;
        const Vec    x2 = Noise(n2 * B, rng);
        Outs         o2(C, x2.size());
        const daisyconvol::Perf pf0 = rig.eng.perf();
        for(size_t k = 0; k < n2; k++)
        {
            float* p[CE::kMaxOutputs];
            o2.Ptrs(k * B, p);
            rig.eng.ProcessBlock(x2.data() + k * B, p, B);
            BgWork(rig.eng);
        }
        for(size_t c = 0; c < C; c++)
        {
            const Vec& h = cfg.external_ir ? b.irs[c] : irs[c];
            worst_err    = std::max(
                worst_err,
                ErrDb(o2.y[c], Reference(x2, h, x2.size()), 0, x2.size()));
        }
        Check(rig.eng.perf().s2_underruns == pf0.s2_underruns,
              "no underruns after the reset");
        if(cfg.external_ir)
            Check(!rig.eng.SwapPending() && rig.eng.ActiveIR() == &b.spec,
                  "reset applied the pending IR");
    }
    printf("  B=%4zu outs=%zu%s: Reset() inside a main-loop frame (%d of 12 "
           "step positions): %8.2f dB\n",
           cfg.block_size,
           C,
           cfg.external_ir ? " external" : "",
           fired,
           worst_err);
    Check(fired >= 4, "reset landed inside frames");
    Check(worst_err < -100.0, "exact after Reset() inside a main-loop frame");
}

// After a Reset() the main loop acknowledges and then stalls: the periods
// it misses must be underruns (the main-loop stage contributes nothing),
// never leftovers from before the reset.
void TestResetThenStall(Rng& rng)
{
    EngineRig rig;
    rig.Init(32, 12000, true);
    const Vec ir = Noise(12000, rng, 0.3);
    rig.eng.SetIR(ir.data(), ir.size());
    const size_t B = 32, R2 = 64;
    const Vec    x1 = Noise(20 * R2 * B, rng);
    RunEngine(rig.eng, x1);
    rig.eng.Reset();               // from the callback
    rig.eng.DoBackgroundStep();    // the main loop acknowledges...
    const Vec x2 = Noise(6 * R2 * B, rng);
    Vec       y(x2.size());
    const daisyconvol::Perf pf0 = rig.eng.perf();
    for(size_t k = 0; k < 6 * R2; k++) // ...and stalls for 6 periods
        rig.eng.ProcessBlock(x2.data() + k * B, y.data() + k * B, B);
    // Periods 2..5 miss their frames: only the in-callback stages play.
    const size_t off = rig.geo.Offset(rig.geo.num_stages - 1);
    const Vec    head(ir.begin(), ir.begin() + off);
    const DVec   ref = Reference(x2, head, x2.size());
    const double err = ErrDb(y, ref, 2 * R2 * B, x2.size());
    const uint32_t und = rig.eng.perf().s2_underruns - pf0.s2_underruns;
    printf("  Reset(), acknowledged, then a stalled main loop: %8.2f dB "
           "against the in-callback stages alone, %u underruns\n",
           err,
           und);
    Check(err < -100.0 && und == 4,
          "missed periods after a reset are silent underruns");
}

// DoBackgroundStep(): bounded work items, any cadence that keeps up is
// exact; late frames are counted.
void TestSteps(Rng& rng)
{
    for(size_t outs : {1u, 2u})
    {
        EngineRig rig;
        CE::Config cfg = MakeConfig(32, 30000, true);
        cfg.num_outputs = outs;
        if(!rig.Init(cfg))
        {
            Check(false, "steps Init");
            return;
        }
        const size_t     B = 32;
        std::vector<Vec> irs;
        const float*     p[CE::kMaxOutputs];
        size_t           lens[CE::kMaxOutputs];
        for(size_t c = 0; c < outs; c++)
        {
            irs.push_back(Noise(30000, rng, 0.3));
            p[c]    = irs[c].data();
            lens[c] = 30000;
        }
        rig.eng.SetIR(p, lens);
        // One step per block is enough: a frame needs 1 + ceil(K / 2) + outs
        // steps and has R2 = 64 blocks.
        const size_t nb = 2000;
        const Vec    x  = Noise(nb * B, rng);
        Outs         o(outs, x.size());
        size_t       steps = 0, idle = 0;
        for(size_t k = 0; k < nb; k++)
        {
            float* q[CE::kMaxOutputs];
            o.Ptrs(k * B, q);
            rig.eng.ProcessBlock(x.data() + k * B, q, B);
            const size_t n = rng.Below(3); // 0..2 steps
            for(size_t i = 0; i < n; i++)
            {
                if(rig.eng.DoBackgroundStep())
                    steps++;
                else
                    idle++;
            }
        }
        double err = -999.0;
        for(size_t c = 0; c < outs; c++)
            err = std::max(err,
                           ErrDb(o.y[c], Reference(x, irs[c], x.size()), 0,
                                 x.size()));
        const daisyconvol::Perf pf = rig.eng.perf();
        printf("  outs=%zu: 0-2 DoBackgroundStep() per block (%zu steps): "
               "%8.2f dB, underruns %u\n",
               outs,
               steps,
               err,
               pf.s2_underruns);
        Check(err < -100.0 && pf.s2_underruns == 0,
              "exact with a few steps per block");
        Check(pf.stage2_frame.count == 0, "no perf without get_cycles");
    }

    // A frame finished after its whole output period: s2_late_skips.
    {
        EngineRig  rig;
        CE::Config cfg = MakeConfig(32, 20000, true);
        rig.Init(cfg);
        const Vec ir = Noise(20000, rng, 0.3);
        rig.eng.SetIR(ir.data(), ir.size());
        const size_t B = 32, R2 = 64;
        const Vec    x = Noise(40 * R2 * B, rng);
        Outs         o(1, x.size());
        size_t       k = 0;
        auto         Block = [&]() {
            float* q[CE::kMaxOutputs];
            o.Ptrs(k * B, q);
            rig.eng.ProcessBlock(x.data() + k * B, q, B);
            k++;
        };
        for(; k < 10 * R2 - 1;)
        {
            Block();
            rig.eng.DoBackgroundWork();
        }
        Block(); // completes frame 9
        // Start frame 9 (window + FFT), then stall for three periods and
        // finish it.
        Check(rig.eng.DoBackgroundStep(), "frame pending");
        for(size_t i = 0; i < 3 * R2; i++)
            Block();
        rig.eng.DoBackgroundWork();
        for(; k < 40 * R2;)
        {
            Block();
            rig.eng.DoBackgroundWork();
        }
        const daisyconvol::Perf pf = rig.eng.perf();
        printf("  late frame: late_skips=%u underruns=%u dropped=%u\n",
               pf.s2_late_skips,
               pf.s2_underruns,
               pf.s2_dropped_inputs);
        Check(pf.s2_late_skips == 1, "late frame counted");
        Check(pf.s2_underruns >= 1, "its period is an underrun");
        const DVec   ref    = Reference(x, ir, x.size());
        const size_t settle = (13 * R2 + 20000 / B + 3 * R2) * B;
        Check(ErrDb(o.y[0], ref, settle, x.size()) < -100.0,
              "exact again after the late frame");
    }
}

// ---------- API details ------------------------------------------------------

void TestBlockMismatchAndSanitize(Rng& rng)
{
    EngineRig rig;
    rig.Init(32, 4096, true);
    const Vec ir = Noise(3000, rng, 0.3);
    rig.eng.SetIR(ir.data(), ir.size());
    const size_t B = 32, nb = 400;
    Vec          x = Noise(nb * B, rng);
    // Non-finite input samples: replaced by 0.
    const size_t bad[] = {100, 2051, 7777};
    x[bad[0]]          = std::numeric_limits<float>::quiet_NaN();
    x[bad[1]]          = std::numeric_limits<float>::infinity();
    x[bad[2]]          = -std::numeric_limits<float>::infinity();
    Vec clean          = x;
    for(size_t i : bad)
        clean[i] = 0.0f;

    Vec y(x.size()), junk(64, 1.0f);
    for(size_t k = 0; k < nb; k++)
    {
        if(k == 50)
        {
            // Wrong lengths: silenced, counted, engine state untouched.
            rig.eng.ProcessBlock(junk.data(), junk.data(), B - 1);
            Check(junk[0] == 0.0f && junk[B - 2] == 0.0f && junk[B - 1] == 1.0f,
                  "wrong-length block silenced");
            std::fill(junk.begin(), junk.end(), 1.0f);
            float* outs[1] = {junk.data()};
            rig.eng.ProcessBlock(junk.data(), outs, 2 * B);
        }
        rig.eng.ProcessBlock(x.data() + k * B, y.data() + k * B, B);
    }
    const daisyconvol::Perf pf = rig.eng.perf();
    const DVec              ref = Reference(clean, ir, y.size());
    const double            err = ErrDb(y, ref, 0, y.size());
    printf("  sanitize: %u non-finite inputs, %8.2f dB; block mismatches %u\n",
           pf.nonfinite_inputs,
           err,
           pf.block_mismatches);
    Check(pf.nonfinite_inputs == 3, "non-finite inputs counted");
    Check(err < -100.0, "non-finite inputs replaced by 0");
    Check(pf.block_mismatches == 2, "block mismatches counted");

    // Without sanitizing, one NaN poisons the output.
    CE::Config cfg      = MakeConfig(32, 4096, true);
    cfg.sanitize_input  = false;
    EngineRig raw;
    raw.Init(cfg);
    raw.eng.SetIR(ir.data(), ir.size());
    for(size_t k = 0; k < nb; k++)
        raw.eng.ProcessBlock(x.data() + k * B, y.data() + k * B, B);
    size_t nonfinite = 0;
    for(float v : y)
        nonfinite += !std::isfinite(v);
    Check(nonfinite > 1000, "sanitize_input = false passes NaN through");

    // Single-output ProcessBlock on a 2-output engine.
    EngineRig two;
    cfg             = MakeConfig(32, 4096, true);
    cfg.num_outputs = 2;
    two.Init(cfg);
    std::fill(junk.begin(), junk.end(), 1.0f);
    two.eng.ProcessBlock(x.data(), junk.data(), B);
    Check(junk[0] == 0.0f && two.eng.perf().block_mismatches == 1,
          "single-output call on a 2-output engine silenced and counted");

    // In-place processing, single and multi-output.
    EngineRig ip;
    ip.Init(32, 4096, true);
    ip.eng.SetIR(ir.data(), ir.size());
    Vec buf = clean;
    for(size_t k = 0; k < nb; k++)
        ip.eng.ProcessBlock(buf.data() + k * B, buf.data() + k * B, B);
    Check(ErrDb(buf, ref, 0, buf.size()) < -100.0, "in-place (in == out)");
    const Vec  ir2 = Noise(1000, rng, 0.3);
    const float* irs[2]  = {ir.data(), ir2.data()};
    const size_t lens[2] = {ir.size(), ir2.size()};
    two.eng.SetIR(irs, lens);
    Vec buf0 = clean, buf1(clean.size());
    for(size_t k = 0; k < nb; k++)
    {
        float* outs[2] = {buf0.data() + k * B, buf1.data() + k * B};
        two.eng.ProcessBlock(buf0.data() + k * B, outs, B);
    }
    Check(ErrDb(buf0, ref, 0, buf0.size()) < -100.0
              && ErrDb(buf1, Reference(clean, ir2, buf1.size()), 0,
                       buf1.size())
                     < -100.0,
          "in-place, 2 outputs");
}

// perf(): snapshot, ResetPerf(), main-loop frame cycles exclude the time the
// audio callback took while it ran.
void TestPerf(Rng& rng)
{
    // Fake clock: every get_cycles call advances it by 10, so each main-loop
    // step costs exactly 10, and a ProcessBlock 40 (4 calls) of which it
    // measures 30 itself. A simulated interrupt (an extra ProcessBlock at
    // the end of a step) may therefore add only 10 to the frame cycles.
    EngineRig  rig;
    CE::Config cfg = MakeConfig(32, 20000, true);
    cfg.get_cycles = HookCycles;
    rig.Init(cfg);
    g_isr     = IsrSim();
    g_isr.eng = &rig.eng;
    const Vec ir = Noise(20000, rng, 0.3);
    rig.eng.SetIR(ir.data(), ir.size());
    const size_t B = 32, nb = 30 * 64;
    const Vec    x = Noise(nb * B, rng);
    Vec          y(x.size()), isr_y(B);
    float*       isr_out[1] = {isr_y.data()};
    Vec          silence(B, 0.0f);
    int          isrs = 0;
    for(size_t k = 0; k < nb; k++)
    {
        rig.eng.ProcessBlock(x.data() + k * B, y.data() + k * B, B);
        if(g_isr.fire_at == 0 && k % 3 == 0)
        {
            // at the end of the next main-loop step that does work
            g_isr.fire_at    = 2;
            g_isr.steps_left = 0;
            g_isr.isr_in     = silence.data();
            g_isr.isr_out    = isr_out;
        }
        while(BgStep(rig.eng)) {}
        if(g_isr.fired)
        {
            isrs += g_isr.fired;
            g_isr.fired   = 0;
            g_isr.fire_at = 0;
        }
    }
    const daisyconvol::Perf pf = rig.eng.perf();
    // Steps per frame: window FFT, ceil(K / 2) MAC steps, one IFFT.
    const Geometry& g     = rig.geo;
    const size_t    last  = g.num_stages - 1;
    const size_t    K     = (20000 - g.Offset(last) + g.part[last] - 1)
                     / g.part[last];
    const uint64_t  steps = 1 + (K + 1) / 2 + 1;
    const uint64_t  want  = 10 * steps * pf.stage2_frame.count + 10ull * isrs;
    printf("  perf: %u frames, %d simulated interrupts inside main-loop "
           "steps: frame cycles %llu (expected %llu)\n",
           pf.stage2_frame.count,
           isrs,
           (unsigned long long)pf.stage2_frame.total,
           (unsigned long long)want);
    Check(isrs > 0 && pf.stage2_frame.total == want,
          "stage2_frame excludes the callback's own cycles");
    Check(pf.callback.count > 0 && pf.callback.peak == 30, "callback perf");
    rig.eng.ResetPerf();
    const daisyconvol::Perf z = rig.eng.perf();
    Check(z.callback.count == 0 && z.stage2_frame.count == 0
              && z.s2_underruns == 0,
          "ResetPerf() reads as cleared at once");
    rig.eng.ProcessBlock(x.data(), y.data(), B);
    Check(rig.eng.perf().callback.count == 1,
          "callback counters restart after ResetPerf()");
}

// The same engine object re-initialized with other block sizes / modes.
void TestReInit(Rng& rng)
{
    EngineRig rig;
    double    worst = -999.0;
    for(size_t b : {32u, 64u, 16u, 256u, 128u})
    {
        for(size_t outs : {1u, 2u})
        {
            CE::Config cfg = MakeConfig(b, 20000, true);
            cfg.num_outputs = outs;
            Check(rig.Init(cfg), "re-Init");
            std::vector<Vec> irs;
            const float*     p[CE::kMaxOutputs];
            size_t           lens[CE::kMaxOutputs];
            for(size_t c = 0; c < outs; c++)
            {
                irs.push_back(Noise(20000 - 1000 * c, rng, 0.3));
                p[c]    = irs[c].data();
                lens[c] = irs[c].size();
            }
            rig.eng.SetIR(p, lens);
            const Vec  x = Noise(((20000 + 3 * 2048 + 4096) / b + 1) * b, rng);
            const Outs o = RunEngine(rig.eng, x);
            for(size_t c = 0; c < outs; c++)
                worst = std::max(worst, ErrDb(o.y[c],
                                              Reference(x, irs[c], x.size()),
                                              0,
                                              x.size()));
        }
    }
    printf("  re-Init across block sizes and output counts: %8.2f dB\n",
           worst);
    Check(worst < -100.0, "re-Init exact");
}

// Random mix of swaps, resets, wrong-length calls and main-loop cadences in
// external mode with a main-loop stage and 2 outputs, against a model that
// restarts the input history at every reset.
void TestFuzz(Rng& rng)
{
    CE::Config cfg = MakeConfig(32, 12000, true, true, nullptr, 2, true);
    EngineRig  rig;
    if(!rig.Init(cfg))
    {
        Check(false, "fuzz Init");
        return;
    }
    const size_t     B = 32, C = 2;
    const SwapTiming tm(rig.geo);
    const size_t     L = rig.geo.Length();
    std::vector<Prepared> ir(4);
    for(size_t i = 0; i < ir.size(); i++)
    {
        std::vector<size_t> lens = {L - 700 * i, 3000 + 2000 * i};
        Check(Prepare(rig, ir[i], lens, rng), "PrepareIR");
    }
    const size_t nb = 6000;
    const Vec    x  = Noise(nb * B, rng);
    Outs         o(C, x.size());

    // Model state.
    int              pending = -2, cur = -1, due = -2;
    size_t           due_at  = 0;
    size_t           seg_beg = 0; // block of the last reset
    std::vector<int> live(nb), seg(nb);
    std::vector<size_t> seg_starts = {0};
    size_t           since = 0; // blocks since reset (engine blk_)
    for(size_t k = 0; k < nb; k++)
    {
        const size_t r = rng.Below(1000);
        if(r < 12)
        {
            const int i = (int)rng.Below(ir.size() + 1) - 1; // -1 = silence
            rig.eng.SwapIR(i < 0 ? nullptr : &ir[i].spec);
            pending = i;
        }
        else if(r < 14)
        {
            // Reset() from the callback: pending, else in-progress, else
            // current IR applies at once.
            rig.eng.Reset();
            if(pending != -2)
                cur = pending;
            else if(due != -2)
                cur = due;
            pending = -2;
            due     = -2;
            seg_beg = k;
            seg_starts.push_back(k);
            since = 0;
        }
        if(due != -2 && since == due_at)
        {
            cur = due;
            due = -2;
        }
        if(since % tm.rl == 0 && pending != -2 && due == -2)
        {
            due     = pending;
            due_at  = since + tm.lead;
            pending = -2;
        }
        live[k] = cur;
        seg[k]  = (int)seg_starts.size() - 1;
        (void)seg_beg;

        float* p[CE::kMaxOutputs];
        o.Ptrs(k * B, p);
        if(rng.Below(100) == 0)
        {
            float* jp[CE::kMaxOutputs];
            Vec    j0(B), j1(B);
            jp[0] = j0.data();
            jp[1] = j1.data();
            rig.eng.ProcessBlock(x.data() + k * B, jp, B / 2); // ignored
        }
        rig.eng.ProcessBlock(x.data() + k * B, p, B);
        since++;
        // Enough main-loop work to keep up, in random-size pieces.
        const size_t steps = 2 + rng.Below(3);
        for(size_t i = 0; i < steps; i++)
            rig.eng.DoBackgroundStep();
        if(rng.Below(8) == 0)
            rig.eng.DoBackgroundWork();
    }
    // Reference: per reset segment, per IR, per output.
    double worst = -999.0;
    size_t checked = 0;
    for(size_t si = 0; si < seg_starts.size(); si++)
    {
        const size_t b0 = seg_starts[si];
        const size_t b1 = si + 1 < seg_starts.size() ? seg_starts[si + 1] : nb;
        const Vec    xs(x.begin() + b0 * B, x.begin() + b1 * B);
        std::vector<std::vector<DVec>> ref(ir.size(), std::vector<DVec>(C));
        for(size_t i = 0; i < ir.size(); i++)
            for(size_t c = 0; c < C; c++)
                ref[i][c] = Reference(xs, ir[i].irs[c], xs.size());
        double se = 0.0, sr = 0.0;
        for(size_t k = b0; k < b1; k++)
            for(size_t c = 0; c < C; c++)
                for(size_t i = 0; i < B; i++)
                {
                    const size_t n = (k - b0) * B + i;
                    const double rv
                        = live[k] < 0 ? 0.0 : ref[live[k]][c][n];
                    const double e = (double)o.y[c][k * B + i] - rv;
                    se += e * e;
                    sr += rv * rv;
                }
        if(sr > 0)
        {
            worst = std::max(worst, 10.0 * std::log10(se / sr));
            checked++;
        }
    }
    const daisyconvol::Perf pf = rig.eng.perf();
    printf("  fuzz: %zu reset segments, worst %8.2f dB, underruns %u, "
           "mismatches %u\n",
           checked,
           worst,
           pf.s2_underruns,
           pf.block_mismatches);
    Check(worst < -100.0, "fuzz: exact in every reset segment");
    Check(pf.s2_underruns == 0, "fuzz: no underruns");
}

// Twiddle tables relocated to RAM: same results, and the plans really use
// the copy. Only meaningful with a backend that has tables (CMSIS).
// Incremental PrepareIR: bit-identical spectra to PrepareIR, one partition
// per step, NaN detection mid-job, argument checks; and a job stepped while
// the engine runs (interleaved with DoBackgroundWork) gives exactly the same
// audio as an IR prepared at once.
void TestPrepareSteps(Rng& rng)
{
    struct Case
    {
        CE::Config  cfg;
        const char* label;
    };
    const Geometry g4 = []() {
        Geometry g;
        g.block      = 16;
        g.num_stages = 4;
        const size_t p[] = {16, 64, 256, 1024};
        const size_t c[] = {8, 6, 6, 3};
        for(size_t s = 0; s < 4; s++)
        {
            g.part[s]  = p[s];
            g.count[s] = c[s];
        }
        return g;
    }();
    const Case cases[] = {
        {MakeConfig(32, 4096, false, true, nullptr, 2), "cab, 2 outputs"},
        {MakeConfig(32, 48000, true, true, nullptr, 1, true),
         "1 s, main-loop stage"},
        {MakeConfig(64, 30000, true, true, nullptr, 3, true),
         "B=64, 3 outputs, main-loop stage"},
        {MakeConfig(16, g4.Length(), false, true, &g4), "custom 4 stages"},
    };
    for(const Case& cs : cases)
    {
        EngineRig rig;
        if(!rig.Init(cs.cfg))
        {
            Check(false, "PrepareSteps Init");
            continue;
        }
        const size_t        C = rig.Outputs();
        const size_t        L = rig.geo.Length();
        std::vector<Vec>    irs;
        const float*        ptrs[CE::kMaxOutputs];
        std::vector<size_t> lens;
        for(size_t c = 0; c < C; c++)
        {
            // different lengths per output, one longer than the geometry
            const size_t len = c == 0 ? L + 77 : (c == 1 ? L / 3 + 5 : 1);
            irs.push_back(Noise(len, rng, 0.5));
            ptrs[c] = irs[c].data();
            lens.push_back(len);
        }
        Vec           ref(rig.eng.spectra_floats(), 0.0f);
        Vec           inc(rig.eng.spectra_floats(), 7.0f);
        Vec           scratch(rig.eng.prepare_scratch_floats());
        CE::IrSpectra sref, sinc;
        Check(rig.eng.PrepareIR(ptrs, lens.data(), ref.data(), ref.size(),
                                scratch.data(), scratch.size(), &sref),
              "PrepareIR (reference)");

        CE::PrepareJob job;
        Check(rig.eng.BeginPrepareIR(&job, ptrs, lens.data(), inc.data(),
                                     inc.size(), scratch.data(),
                                     scratch.size(), &sinc),
              "BeginPrepareIR");
        size_t expect = 0;
        for(size_t c = 0; c < C; c++)
            for(size_t st = 0; st < rig.geo.num_stages; st++)
                expect += sref.k[c][st];
        size_t calls = 0;
        while(rig.eng.PrepareIRStep(&job))
        {
            calls++;
            Check(job.steps_done() == calls, "one partition per step");
        }
        calls++; // the last step returns false
        Check(job.done() && job.ok(), "job finished ok");
        Check(job.steps_total() == expect && calls == expect,
              "steps = partitions");
        Check(!rig.eng.PrepareIRStep(&job), "step after done is a no-op");
        // Same spectra, same layout (pointers relative to the storage).
        bool same = sinc.len == sref.len;
        for(size_t c = 0; c < C; c++)
            for(size_t st = 0; st < rig.geo.num_stages; st++)
                same &= sinc.k[c][st] == sref.k[c][st]
                        && sinc.h[c][st] - inc.data()
                               == sref.h[c][st] - ref.data();
        // Unused partition slots are not written by either path: compare
        // only the active ones.
        for(size_t c = 0; c < C && same; c++)
            for(size_t st = 0; st < rig.geo.num_stages; st++)
            {
                const size_t n = sref.k[c][st] * 2 * rig.geo.part[st];
                same &= std::memcmp(sinc.h[c][st], sref.h[c][st],
                                    n * sizeof(float))
                        == 0;
            }
        printf("  %-34s %4zu steps, bit-identical: %s\n",
               cs.label,
               expect,
               same ? "yes" : "NO");
        Check(same, "incremental spectra == PrepareIR");

        // NaN in the middle: the job fails at that partition.
        irs[0][irs[0].size() / 2] = std::nanf("");
        CE::PrepareJob bad;
        Check(rig.eng.BeginPrepareIR(&bad, ptrs, lens.data(), inc.data(),
                                     inc.size(), scratch.data(),
                                     scratch.size(), &sinc),
              "BeginPrepareIR does not scan samples");
        while(rig.eng.PrepareIRStep(&bad)) {}
        Check(bad.done() && !bad.ok() && bad.steps_done() < bad.steps_total(),
              "NaN fails the job mid-way");
        Check(!rig.eng.PrepareIR(ptrs, lens.data(), ref.data(), ref.size(),
                                 scratch.data(), scratch.size(), &sref),
              "PrepareIR still rejects NaN up front");

        // Argument checks.
        CE::PrepareJob a;
        Check(!rig.eng.BeginPrepareIR(&a, ptrs, lens.data(), inc.data(),
                                      inc.size() - 1, scratch.data(),
                                      scratch.size(), &sinc),
              "BeginPrepareIR rejects small storage");
        Check(!rig.eng.BeginPrepareIR(&a, ptrs, lens.data(), inc.data(),
                                      inc.size(), scratch.data(),
                                      scratch.size() - 1, &sinc),
              "BeginPrepareIR rejects small scratch");
        Check(!rig.eng.PrepareIRStep(&a),
              "a rejected job does not run");
    }

    // Stepped while running: two identical engines, one switches to an IR
    // prepared at once, the other to the same IR prepared one step per
    // block; switched on the same block, the outputs must be identical.
    const CE::Config cfg = MakeConfig(32, 48000, true, true, nullptr, 2, true);
    EngineRig        a, b;
    if(!a.Init(cfg) || !b.Init(cfg))
    {
        Check(false, "PrepareSteps live Init");
        return;
    }
    const std::vector<size_t> lens = {48000, 30000};
    Prepared                  first;
    Check(Prepare(a, first, lens, rng), "PrepareIR");
    a.eng.SwapIR(&first.spec);
    b.eng.SwapIR(&first.spec);
    a.eng.Reset();
    b.eng.Reset();

    std::vector<Vec>    irs2 = {Noise(48000, rng, 0.5), Noise(30000, rng, 0.5)};
    const float*        p2[2] = {irs2[0].data(), irs2[1].data()};
    Vec                 sa(a.eng.spectra_floats()), sb(b.eng.spectra_floats());
    Vec                 scra(a.eng.prepare_scratch_floats());
    Vec                 scrb(b.eng.prepare_scratch_floats());
    CE::IrSpectra       spa, spb;
    Check(a.eng.PrepareIR(p2, lens.data(), sa.data(), sa.size(), scra.data(),
                          scra.size(), &spa),
          "PrepareIR");
    CE::PrepareJob job;
    const size_t   B       = 32;
    const size_t   nblocks = 1200;
    const Vec      x       = Noise(nblocks * B, rng);
    Vec            ya[2], yb[2];
    for(size_t c = 0; c < 2; c++)
    {
        ya[c].assign(x.size(), 0.0f);
        yb[c].assign(x.size(), 0.0f);
    }
    bool   swapped = false, identical = true;
    size_t swap_blk = 0;
    for(size_t k = 0; k < nblocks; k++)
    {
        if(k == 100)
            Check(b.eng.BeginPrepareIR(&job, p2, lens.data(), sb.data(),
                                       sb.size(), scrb.data(), scrb.size(),
                                       &spb),
                  "BeginPrepareIR (live)");
        if(k >= 100 && !job.done())
            b.eng.PrepareIRStep(&job);
        if(!swapped && k > 100 && job.done())
        {
            Check(job.ok(), "live job ok");
            a.eng.SwapIR(&spa);
            b.eng.SwapIR(&spb);
            swapped  = true;
            swap_blk = k;
        }
        float* oa[2] = {ya[0].data() + k * B, ya[1].data() + k * B};
        float* ob[2] = {yb[0].data() + k * B, yb[1].data() + k * B};
        a.eng.ProcessBlock(x.data() + k * B, oa, B);
        b.eng.ProcessBlock(x.data() + k * B, ob, B);
        a.eng.DoBackgroundWork();
        b.eng.DoBackgroundWork();
    }
    for(size_t c = 0; c < 2; c++)
        identical &= std::memcmp(ya[c].data(), yb[c].data(),
                                 x.size() * sizeof(float))
                     == 0;
    printf("  live: prepared in %zu steps over %zu blocks, swapped at block "
           "%zu, outputs identical: %s\n",
           job.steps_total(),
           swap_blk - 100,
           swap_blk,
           identical ? "yes" : "NO");
    Check(swapped && identical, "stepped IR == IR prepared at once, live");
    Check(a.eng.perf().s2_underruns == 0 && b.eng.perf().s2_underruns == 0,
          "no underruns while preparing in steps");
}

void TestFftRelocation(Rng& rng)
{
    const size_t bytes = daisyconvol::FftTableBytes();
    if(bytes == 0)
    {
        printf("  FFT tables: none in this backend (reference FFT)\n");
        return;
    }
    // Every size has a plan by now: the constexpr sizing must match.
    size_t all = 0;
    for(size_t n = 32; n <= 4096; n *= 2)
        all += daisyconvol::FftTableBytesFor(n);
    Check(bytes == all, "FftTableBytesFor() matches the CMSIS tables");
    static std::vector<uint32_t> ram; // must outlive the plans
    ram.assign(bytes / 4 + 1, 0);
    Check(!daisyconvol::FftRelocateTables(ram.data(), bytes - 4),
          "relocation into a too small buffer refused");
    Check(daisyconvol::FftRelocateTables(ram.data(), bytes),
          "tables relocated");
    Check(daisyconvol::FftTableBytes() == 0, "nothing left to relocate");

    EngineRig rig;
    rig.Init(32, 30000, true);
    const Vec ir = Noise(30000, rng, 0.3);
    rig.eng.SetIR(ir.data(), ir.size());
    const Vec  x    = Noise(1500 * 32, rng);
    const DVec ref  = Reference(x, ir, x.size());
    const double ok = ErrDb(RunEngine(rig.eng, x).y[0], ref, 0, x.size());
    // Overwrite the first half: twiddles only (floats come first and are
    // most of the bytes; junk bin positions would read out of bounds).
    std::vector<uint32_t> saved = ram;
    std::fill(ram.begin(), ram.begin() + ram.size() / 2, 0x3F800000u); // 1.0f
    rig.eng.SetIR(ir.data(), ir.size());
    const double broken = ErrDb(RunEngine(rig.eng, x).y[0], ref, 0, x.size());
    ram = saved;
    printf("  FFT tables relocated (%zu bytes): %8.2f dB; with the RAM copy "
           "overwritten: %8.2f dB\n",
           bytes,
           ok,
           broken);
    Check(ok < -100.0, "exact with relocated tables");
    Check(!(broken < -60.0), "the plans read the relocated tables");
}

} // namespace

int main()
{
    Rng rng;

    printf("== geometry ==\n");
    TestGeometries();

    printf("== SetIR: every stage boundary ==\n");
    const size_t blocks[] = {16, 32, 64, 128, 256, 512, 1024};
    for(size_t b : blocks)
    {
        TestConfig(b, 4096, true, rng);  // cabinet
        TestConfig(b, 96000, true, rng); // 2 s reverb, main-loop stage
        if(b >= 64)                      // (small blocks: heavy callback)
            TestConfig(b, 96000, false, rng);
    }
    TestConfig(32, 144000, true, rng); // 3 s
    {
        const Geometry g = Custom4(20, true);
        Check(g.Valid(), "custom geometry valid");
        TestConfig(16, g.Length(), true, rng, &g);
    }

    printf("== several outputs (shared input delay lines) ==\n");
    for(size_t outs : {2u, 3u})
    {
        CE::Config cfg  = MakeConfig(32, 30000, true);
        cfg.num_outputs = outs;
        TestConfig(cfg, rng, outs == 2 ? "2 outputs" : "3 outputs");
    }
    {
        CE::Config cfg  = MakeConfig(128, 4096, true);
        cfg.num_outputs = CE::kMaxOutputs;
        TestConfig(cfg, rng, "4 outputs");
    }

    printf("== main-loop stall recovery ==\n");
    for(size_t b : {16u, 32u, 64u, 128u})
        TestUnderrunRecovery(b, rng);

    printf("== external IR switching ==\n");
    for(size_t b : {16u, 32u, 64u, 128u, 256u})
        TestExternalSwap(MakeConfig(b, CE::S01Len(b), false, true), rng);
    TestExternalSwap(MakeConfig(32, 24000, false, true), rng); // stage 1 long
    {
        const Geometry g = Custom4(3, false); // 3 time-sliced stages
        TestExternalSwap(MakeConfig(16, g.Length(), false, true, &g), rng);
    }
    printf("== external IR switching with a main-loop stage ==\n");
    for(size_t b : {16u, 32u, 64u, 128u, 256u, 1024u})
        TestExternalSwap(MakeConfig(b, 20000, true, true, nullptr, 1, true),
                         rng);
    TestExternalSwap(MakeConfig(32, 24000, true, true, nullptr, 2, true), rng);
    {
        const Geometry g = Custom4(8, true); // 4 stages, last in main loop
        TestExternalSwap(MakeConfig(16, g.Length(), true, true, &g), rng);
    }
    TestExternalSwap(MakeConfig(32, 4096, false, true, nullptr, 2), rng);
    TestStorageReuse(MakeConfig(32, 4096, false, true), rng);
    TestStorageReuse(MakeConfig(32, 20000, true, true, nullptr, 2, true), rng);

    printf("== 32-bit counter wraps ==\n");
    for(size_t b : {16u, 32u, 64u})
        TestWrap(MakeConfig(b, 20000, true), rng);
    TestWrap(MakeConfig(32, 20000, true, true, nullptr, 1, true), rng);
    TestWrap(MakeConfig(32, 4096, false, true), rng);

    printf("== Reset() from the callback inside a main-loop frame ==\n");
    TestResetInsideBg(MakeConfig(32, 12000, true), rng);
    TestResetInsideBg(MakeConfig(128, 12000, true), rng);
    {
        CE::Config cfg  = MakeConfig(32, 12000, true);
        cfg.num_outputs = 2;
        TestResetInsideBg(cfg, rng);
    }
    TestResetInsideBg(MakeConfig(32, 12000, true, true, nullptr, 2, true),
                      rng);
    TestResetThenStall(rng);

    printf("== main-loop steps ==\n");
    TestSteps(rng);

    printf("== API details ==\n");
    TestBlockMismatchAndSanitize(rng);
    TestPerf(rng);
    TestReInit(rng);
    TestFuzz(rng);
    printf("== incremental PrepareIR ==\n");
    TestPrepareSteps(rng);
    TestFftRelocation(rng);

    if(g_failures == 0)
    {
        printf("\nALL TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", g_failures);
    return 1;
}
