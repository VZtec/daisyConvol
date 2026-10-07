// direct.cpp - the partitioned engine against a plain time-domain
// convolution, y[n] = sum_k h[k] x[n - k] in double precision, for several
// IRs and block sizes. Prints the error of every case in dB (relative RMS
// over the whole output, tail included) and fails above -100 dB.
//
// The cases include the edges of the partitioning: a unit impulse (the
// output must be the input, with no added latency), IRs shorter than one
// partition and exactly one partition long, an impulse placed in the
// main-loop stage, and an IR of the maximum length the engine is configured
// for. conv.cpp covers the rest of the API (switching, resets, stalls,
// wraps...).
//
// Built twice by CMakeLists.txt: on the portable reference FFT and, with
// LIBDAISY_DIR, on the target's CMSIS-DSP FFT built for the host.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "daisyconvol/conv_engine.h"

namespace
{

using CE = daisyconvol::ConvEngine;

// Long enough for a main-loop stage at every block size (the in-callback
// stages cover 128 x B samples, at most 4096 here).
constexpr size_t kMaxIr = 12000;

// Deterministic PRNG (xorshift), uniform in [-1, 1).
struct Rng
{
    uint64_t s = 0x9E3779B97F4A7C15ull;
    double   Next()
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double)(int64_t)s / 9223372036854775808.0;
    }
};

std::vector<float> Noise(size_t n, Rng& rng)
{
    std::vector<float> v(n);
    for(auto& x : v)
        x = (float)rng.Next();
    return v;
}

// Exponentially decaying noise, like a room or cabinet response.
std::vector<float> Decay(size_t n, Rng& rng)
{
    std::vector<float> v(n);
    for(size_t i = 0; i < n; i++)
        v[i] = (float)(rng.Next() * std::exp(-6.0 * (double)i / (double)n));
    return v;
}

std::vector<float> Impulse(size_t n, size_t at)
{
    std::vector<float> v(n, 0.0f);
    v[at] = 1.0f;
    return v;
}

std::vector<double> Direct(const std::vector<float>& x,
                           const std::vector<float>& h)
{
    std::vector<double> y(x.size(), 0.0);
    for(size_t k = 0; k < h.size(); k++)
    {
        if(h[k] == 0.0f)
            continue;
        const double hk = h[k];
        for(size_t n = k; n < x.size(); n++)
            y[n] += hk * (double)x[n - k];
    }
    return y;
}

double ErrorDb(const std::vector<float>& y, const std::vector<double>& ref)
{
    double e = 0.0, r = 0.0;
    for(size_t i = 0; i < ref.size(); i++)
    {
        const double d = (double)y[i] - ref[i];
        e += d * d;
        r += ref[i] * ref[i];
    }
    if(e == 0.0)
        return -400.0; // exact
    return 10.0 * std::log10(e / r);
}

struct Case
{
    const char*        name;
    std::vector<float> h;
};

// Runs the engine (block size b, main-loop work after every block) over x
// with IR h; returns the error against ref (the direct convolution) in dB,
// or +999 if the engine refused the configuration or missed a deadline.
double Run(size_t                     b,
           const std::vector<float>&  h,
           const std::vector<float>&  x,
           const std::vector<double>& ref)
{
    CE::Config cfg;
    cfg.block_size = b;
    cfg.max_ir_len = kMaxIr;
    std::vector<float> fast(CE::FastFloats(cfg) + 1);
    std::vector<float> medium(CE::MediumFloats(cfg) + 1);
    std::vector<float> slow(CE::SlowFloats(cfg) + 1);
    CE                 eng;
    if(!eng.Init(cfg,
                 {fast.data(), fast.size() - 1, 0},
                 {medium.data(), medium.size() - 1, 0},
                 {slow.data(), slow.size() - 1, 0})
       || !eng.SetIR(h.data(), h.size()))
        return 999.0;

    std::vector<float> y(x.size());
    for(size_t i = 0; i + b <= x.size(); i += b)
    {
        eng.ProcessBlock(x.data() + i, y.data() + i, b);
        eng.DoBackgroundWork();
    }
    if(eng.perf().s2_underruns != 0)
        return 999.0;
    return ErrorDb(y, ref);
}

} // namespace

int main()
{
    Rng                rng;
    const size_t       blocks[] = {16, 32, 64, 128, 256};
    std::vector<Case>  cases;
    cases.push_back({"unit impulse (1 sample)", Impulse(1, 0)});
    cases.push_back({"3 samples (< one partition)", Noise(3, rng)});
    cases.push_back({"15 samples (< one partition)", Noise(15, rng)});
    cases.push_back({"16 samples (one partition at B=16)", Noise(16, rng)});
    cases.push_back({"1000 samples, decaying", Decay(1000, rng)});
    cases.push_back({"4096 samples (cabinet), decaying", Decay(4096, rng)});
    cases.push_back({"4097 samples (main-loop stage)", Noise(4097, rng)});
    cases.push_back({"impulse at 9000 (main-loop stage)", Impulse(9001, 9000)});
    cases.push_back({"12000 samples (maximum), decaying", Decay(kMaxIr, rng)});
    cases.push_back({"12000 samples (maximum), noise", Noise(kMaxIr, rng)});

    // Input: noise, then silence long enough for the longest tail to end
    // (the error covers the tail too).
    std::vector<float> x = Noise(16384, rng);
    x.resize(16384 + kMaxIr + 2048, 0.0f);

    printf("daisyConvol vs. direct time-domain convolution: error in dB "
           "(pass < -100)\n");
    printf("%-36s", "IR");
    for(size_t b : blocks)
    {
        char col[16];
        snprintf(col, sizeof col, "B=%zu", b);
        printf(" %8s", col);
    }
    printf("\n");

    int failures = 0;
    for(const Case& c : cases)
    {
        printf("%-36s", c.name);
        const std::vector<double> ref = Direct(x, c.h);
        for(size_t b : blocks)
        {
            const double db = Run(b, c.h, x, ref);
            printf(" %8.1f", db);
            failures += db < -100.0 ? 0 : 1;
        }
        printf("\n");
    }
    if(failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL CASES PASSED\n");
    return 0;
}
