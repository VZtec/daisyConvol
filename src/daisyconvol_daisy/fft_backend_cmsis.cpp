// fft_backend_cmsis.cpp - target implementation of fft_backend.h on top of
// CMSIS-DSP arm_rfft_fast_f32.
//
// The instances are filled in manually instead of calling
// arm_rfft_fast_init_f32(): the copy bundled with libDaisy has every size
// except 2048/4096 commented out (a flash-size trim), and we need 64 and 512
// as well. The tables referenced here all exist in arm_common_tables.c; with
// -fdata-sections / --gc-sections only the referenced ones are linked in.

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

namespace daisyconvol
{

struct FftPlan
{
    arm_rfft_fast_instance_f32 inst;
    size_t                     n         = 0;
    bool                       relocated = false;
};

namespace
{

// One plan per supported size (32..4096): plans are shared and never
// released, so engines re-initialized with other block sizes (or several
// engines) can never run out.
constexpr size_t kMaxPlans = 8;
FftPlan          plan_pool[kMaxPlans];

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

// Table sizes of an nfft-point real FFT: complex twiddles of the N/2-point
// CFFT (2 * N/2 floats), real-stage twiddles (N floats), bit reversal
// (bitRevLength uint16).
size_t TwiddleFloats(const FftPlan& p)
{
    return p.n + p.n;
}

size_t PlanTableBytes(const FftPlan& p)
{
    return TwiddleFloats(p) * sizeof(float32_t)
           + p.inst.Sint.bitRevLength * sizeof(uint16_t);
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
            if(!FillInstance(&p.inst, nfft))
                return nullptr;
            p.n = nfft;
            return &p;
        }
    }
    return nullptr;
}

void FftForward(FftPlan* plan, float* time_scratch, float* spec_out)
{
    arm_rfft_fast_f32(&plan->inst, time_scratch, spec_out, 0);
}

void FftInverse(FftPlan* plan, float* spec_scratch, float* time_out)
{
    arm_rfft_fast_f32(&plan->inst, spec_scratch, time_out, 1);
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
    // Floats first, then the uint16 bit-reversal tables: every table stays
    // naturally aligned.
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
        const size_t len = p.inst.Sint.bitRevLength;
        memcpy(u, p.inst.Sint.pBitRevTable, len * sizeof(uint16_t));
        p.inst.Sint.pBitRevTable = u;
        u += len;
        p.relocated = true;
    }
    return true;
}

} // namespace daisyconvol
