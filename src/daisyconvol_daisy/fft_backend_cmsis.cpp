// fft_backend_cmsis.cpp - target implementation of fft_backend.h on top of
// CMSIS-DSP's complex FFT (the butterflies of arm_rfft_fast_f32).
//
// The instances are filled in manually instead of calling
// arm_rfft_fast_init_f32(): the copy bundled with libDaisy has every size
// except 2048/4096 commented out (a flash-size trim), and we need 64 and 512
// as well. The tables referenced here all exist in arm_common_tables.c; with
// -fdata-sections / --gc-sections only the referenced ones are linked in.
//
// What a convolution needs from an N-point real FFT is less than what
// arm_rfft_fast_f32 does (measured on the Daisy Seed, N = 512: forward
// 19.7k cycles = butterflies 11.9k + bit reversal 2.7k + real split 5.1k;
// inverse 21.7k, adding 2.3k of conjugation and 1/N scaling):
//   - forward: the CFFT runs without its bit-reversal pass, and the real
//     split reads each bin from where the butterflies left it (a position
//     table built from CMSIS's bit-reversal table), writing the usual
//     packed spectrum in natural order;
//   - inverse: IDFT(Z)[m] = DFT(Z)[-m] / M, so the same forward butterflies
//     run on the merged spectrum, with neither conjugations nor bit
//     reversal; only the second half of the output is gathered (what
//     overlap-save keeps), with the 1/M scaling, straight into its
//     destination.
// The split and the merge are CMSIS's stage_rfft_f32 / merge_rfft_f32
// formulas, two bins per iteration.

#include "daisyconvol/fft_backend.h"

#include <cstdint>
#include <cstring>

// FFT sizes compiled in (see FillInstance). Default: block_size = 32.
#if !defined(DAISYCONVOL_FFT_32) && !defined(DAISYCONVOL_FFT_64)    \
    && !defined(DAISYCONVOL_FFT_128) && !defined(DAISYCONVOL_FFT_256) \
    && !defined(DAISYCONVOL_FFT_512) && !defined(DAISYCONVOL_FFT_1024) \
    && !defined(DAISYCONVOL_FFT_2048) && !defined(DAISYCONVOL_FFT_4096)
#define DAISYCONVOL_FFT_64 1
#define DAISYCONVOL_FFT_512 1
#define DAISYCONVOL_FFT_4096 1
#endif

#include "arm_math.h"
// The table header has no extern "C" of its own: without it, compilers that
// mangle variable names in C++ (MSVC, for the desktop test build) look for
// the wrong symbols. GCC does not mangle them either way.
extern "C"
{
#include "arm_common_tables.h"
}

// The per-bin loops at -O2 even in an -Os build (as conv_engine.cpp does).
#if defined(__GNUC__) && !defined(__clang__)
#define FFT_HOT __attribute__((optimize("O2")))
#else
#define FFT_HOT
#endif

namespace daisyconvol
{

struct FftPlan
{
    arm_rfft_fast_instance_f32 inst;
    // pos[q]: complex index where the CFFT, run without bit reversal, leaves
    // bin q (n/2 entries).
    uint16_t* pos       = nullptr;
    size_t    n         = 0;
    bool      relocated = false;
};

namespace
{

// One plan per supported size (32..4096): plans are shared and never
// released, so engines re-initialized with other block sizes (or several
// engines) can never run out.
constexpr size_t kMaxPlans = 8;
FftPlan          plan_pool[kMaxPlans];

// The position tables of every size compiled in, n/2 entries each.
constexpr size_t kPosEntries = 1 // never a zero-length array
#if DAISYCONVOL_FFT_32
                               + 16
#endif
#if DAISYCONVOL_FFT_64
                               + 32
#endif
#if DAISYCONVOL_FFT_128
                               + 64
#endif
#if DAISYCONVOL_FFT_256
                               + 128
#endif
#if DAISYCONVOL_FFT_512
                               + 256
#endif
#if DAISYCONVOL_FFT_1024
                               + 512
#endif
#if DAISYCONVOL_FFT_2048
                               + 1024
#endif
#if DAISYCONVOL_FFT_4096
                               + 2048
#endif
    ;
uint16_t pos_pool[kPosEntries];
size_t   pos_used = 0;

bool FillInstance(arm_rfft_fast_instance_f32* s, size_t nfft)
{
    arm_cfft_instance_f32* si = &s->Sint;
    si->fftLen                = (uint16_t)(nfft / 2);
    s->fftLenRFFT             = (uint16_t)nfft;
    switch(nfft)
    {
#define CONV_RFFT_CASE(N, HALF)                                    \
    case N:                                                        \
        si->bitRevLength = ARMBITREVINDEXTABLE_##HALF##_TABLE_LENGTH; \
        si->pBitRevTable = (uint16_t*)armBitRevIndexTable##HALF;   \
        si->pTwiddle     = (float32_t*)twiddleCoef_##HALF;         \
        s->pTwiddleRFFT  = (float32_t*)twiddleCoef_rfft_##N;       \
        return true;
        // Each referenced size drags its twiddle/bitrev tables into flash,
        // so only the sizes enabled with DAISYCONVOL_FFT_<N> are compiled
        // (daisyconvol.mk derives them from DAISYCONVOL_BLOCKS). With none
        // defined, the block_size = 32 set is used.
#if DAISYCONVOL_FFT_32
        CONV_RFFT_CASE(32, 16)
#endif
#if DAISYCONVOL_FFT_64
        CONV_RFFT_CASE(64, 32)
#endif
#if DAISYCONVOL_FFT_128
        CONV_RFFT_CASE(128, 64)
#endif
#if DAISYCONVOL_FFT_256
        CONV_RFFT_CASE(256, 128)
#endif
#if DAISYCONVOL_FFT_512
        CONV_RFFT_CASE(512, 256)
#endif
#if DAISYCONVOL_FFT_1024
        CONV_RFFT_CASE(1024, 512)
#endif
#if DAISYCONVOL_FFT_2048
        CONV_RFFT_CASE(2048, 1024)
#endif
#if DAISYCONVOL_FFT_4096
        CONV_RFFT_CASE(4096, 2048)
#endif
#undef CONV_RFFT_CASE
        default: return false;
    }
}

// The bit-reversal pass swaps complex elements (its table holds byte
// offsets): applied to the identity, it gives for each bin q the position
// the butterflies leave it at.
void BuildPositions(FftPlan* p)
{
    const size_t m = p->n / 2;
    for(size_t i = 0; i < m; i++)
        p->pos[i] = (uint16_t)i;
    const uint16_t* tab = p->inst.Sint.pBitRevTable;
    for(size_t i = 0; i + 1 < p->inst.Sint.bitRevLength; i += 2)
    {
        const size_t   a = tab[i] >> 3, b = tab[i + 1] >> 3;
        const uint16_t t = p->pos[a];
        p->pos[a]        = p->pos[b];
        p->pos[b]        = t;
    }
}

// Tables of an nfft-point real FFT: complex twiddles of the N/2-point CFFT
// (2 * N/2 floats), real-stage twiddles (N floats), positions (N/2 uint16).
size_t TwiddleFloats(const FftPlan& p)
{
    return p.n + p.n;
}

size_t PlanTableBytes(const FftPlan& p)
{
    return TwiddleFloats(p) * sizeof(float32_t) + p.n / 2 * sizeof(uint16_t);
}

// Real split, stage_rfft_f32's formulas: bins k and m - k of the packed
// spectrum from Z[k] and Z[m - k] (z: the CFFT output, scrambled).
FFT_HOT void Split(const FftPlan* p, const float* z, float* out)
{
    const size_t     m   = p->n / 2;
    const uint16_t*  pos = p->pos;
    const float32_t* tw  = p->inst.pTwiddleRFFT;
    {
        const float* a = z + 2 * pos[0];
        out[0]         = 0.5f * ((a[0] + a[0]) + (a[1] + a[1])); // DC
        out[1]         = 0.5f * ((a[0] + a[0]) - (a[1] + a[1])); // Nyquist
    }
    for(size_t k = 1; k <= m / 2; k++)
    {
        const size_t kk = m - k;
        const float* a  = z + 2 * pos[k];
        const float* b  = z + 2 * pos[kk];
        const float  ar = a[0], ai = a[1], br = b[0], bi = b[1];
        {
            const float twr = tw[2 * k], twi = tw[2 * k + 1];
            const float t1a = br - ar;
            const float t1b = bi + ai;
            const float p0 = twr * t1a, p1 = twi * t1a;
            const float p2 = twr * t1b, p3 = twi * t1b;
            out[2 * k]     = 0.5f * (ar + br + p0 + p3);
            out[2 * k + 1] = 0.5f * (ai - bi + p1 - p2);
        }
        if(kk != k)
        {
            const float twr = tw[2 * kk], twi = tw[2 * kk + 1];
            const float t1a = ar - br;
            const float t1b = ai + bi;
            const float p0 = twr * t1a, p1 = twi * t1a;
            const float p2 = twr * t1b, p3 = twi * t1b;
            out[2 * kk]     = 0.5f * (br + ar + p0 + p3);
            out[2 * kk + 1] = 0.5f * (bi - ai + p1 - p2);
        }
    }
}

// Inverse of the split, merge_rfft_f32's formulas, in place: Z[k] and
// Z[m - k] from bins k and m - k.
FFT_HOT void Merge(const FftPlan* p, float* x)
{
    const size_t     m  = p->n / 2;
    const float32_t* tw = p->inst.pTwiddleRFFT;
    {
        const float dc = x[0], ny = x[1];
        x[0]           = 0.5f * (dc + ny);
        x[1]           = 0.5f * (dc - ny);
    }
    for(size_t k = 1; k <= m / 2; k++)
    {
        const size_t kk = m - k;
        float*       a  = x + 2 * k;
        float*       b  = x + 2 * kk;
        const float  ar = a[0], ai = a[1], br = b[0], bi = b[1];
        {
            const float twr = tw[2 * k], twi = tw[2 * k + 1];
            const float t1a = ar - br;
            const float t1b = ai + bi;
            const float r = twr * t1a, s = twi * t1b;
            const float t = twi * t1a, u = twr * t1b;
            a[0] = 0.5f * (ar + br - r - s);
            a[1] = 0.5f * (ai - bi + t - u);
        }
        if(kk != k)
        {
            const float twr = tw[2 * kk], twi = tw[2 * kk + 1];
            const float t1a = br - ar;
            const float t1b = bi + ai;
            const float r = twr * t1a, s = twi * t1b;
            const float t = twi * t1a, u = twr * t1b;
            b[0] = 0.5f * (br + ar - r - s);
            b[1] = 0.5f * (bi - ai + t - u);
        }
    }
}

// Second half of the inverse: complex samples m/2 .. m-1 of
// IDFT(Z) = DFT(Z)[-j] / m, i.e. bins m/2 .. 1 of the scrambled forward
// output y, scaled.
FFT_HOT void GatherHalf(const FftPlan* p, const float* y, float* out)
{
    const size_t    m     = p->n / 2;
    const uint16_t* pos   = p->pos;
    const float     scale = 1.0f / (float)m;
    for(size_t j = 0; j < m / 2; j++)
    {
        const float* v = y + 2 * pos[m / 2 - j];
        out[2 * j]     = v[0] * scale;
        out[2 * j + 1] = v[1] * scale;
    }
}

} // namespace

FftPlan* FftMakePlan(size_t nfft)
{
    for(auto& p : plan_pool)
        if(p.n == nfft)
            return &p;
    for(auto& p : plan_pool)
    {
        if(p.n == 0)
        {
            if(!FillInstance(&p.inst, nfft)
               || pos_used + nfft / 2 > kPosEntries)
                return nullptr;
            p.n   = nfft;
            p.pos = pos_pool + pos_used;
            pos_used += nfft / 2;
            BuildPositions(&p);
            return &p;
        }
    }
    return nullptr;
}

void FftForward(FftPlan* plan, float* time_scratch, float* spec_out)
{
    arm_cfft_f32(&plan->inst.Sint, time_scratch, 0, 0); // no bit reversal
    Split(plan, time_scratch, spec_out);
}

void FftInverseHalf(FftPlan* plan, float* spec_scratch, float* half_out)
{
    Merge(plan, spec_scratch);
    arm_cfft_f32(&plan->inst.Sint, spec_scratch, 0, 0); // forward, scrambled
    GatherHalf(plan, spec_scratch, half_out);
}

size_t FftTableBytes()
{
    size_t n = 0;
    for(const auto& p : plan_pool)
        if(p.n && !p.relocated)
            n += PlanTableBytes(p);
    return n;
}

bool FftRelocateTables(void* dst, size_t bytes)
{
    if(!dst || bytes < FftTableBytes())
        return false;
    // Floats first, then the uint16 position tables: every table stays
    // naturally aligned. The bit-reversal tables are only read by
    // FftMakePlan: they stay in flash.
    float32_t* f = static_cast<float32_t*>(dst);
    for(auto& p : plan_pool)
    {
        if(!p.n || p.relocated)
            continue;
        const size_t half = p.n; // CFFT twiddles: 2 * (n / 2) floats
        memcpy(f, p.inst.Sint.pTwiddle, half * sizeof(float32_t));
        p.inst.Sint.pTwiddle = f;
        f += half;
        memcpy(f, p.inst.pTwiddleRFFT, p.n * sizeof(float32_t));
        p.inst.pTwiddleRFFT = f;
        f += p.n;
    }
    uint16_t* u = reinterpret_cast<uint16_t*>(f);
    for(auto& p : plan_pool)
    {
        if(!p.n || p.relocated)
            continue;
        memcpy(u, p.pos, p.n / 2 * sizeof(uint16_t));
        p.pos = u;
        u += p.n / 2;
        p.relocated = true;
    }
    return true;
}

} // namespace daisyconvol
