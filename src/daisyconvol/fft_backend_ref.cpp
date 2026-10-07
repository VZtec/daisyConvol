// fft_backend_ref.cpp - portable reference implementation of fft_backend.h
// for the desktop test harness. Not built for the target.
//
// Produces bit-identical *layout* to CMSIS arm_rfft_fast_f32 (packed real
// spectrum: [DC, Nyquist, re1, im1, re2, im2, ...], unscaled forward,
// 1/N-scaled inverse) so that the engine's packed-spectrum MAC kernel is
// exercised exactly as on the target. Internally double precision, with a
// shared static scratch: not reentrant (single-threaded tests only).

#include "fft_backend.h"

#include <cmath>
#include <cstring>

namespace daisyconvol
{

namespace
{

constexpr size_t kMaxN    = 4096;
constexpr double kPi      = 3.14159265358979323846;

// In-place iterative radix-2 complex FFT (DIT), e^{-j2pikn/N} forward.
void CFft(double* re, double* im, size_t n, bool inverse)
{
    // bit-reversal permutation
    for(size_t i = 1, j = 0; i < n; i++)
    {
        size_t bit = n >> 1;
        for(; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if(i < j)
        {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t        = im[i]; im[i] = im[j]; im[j] = t;
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

double scratch_re[kMaxN];
double scratch_im[kMaxN];

} // namespace

struct FftPlan
{
    size_t n = 0;
};

FftPlan* FftMakePlan(size_t nfft)
{
    static FftPlan pool[8];
    if(nfft < 4 || nfft > kMaxN || (nfft & (nfft - 1)) != 0)
        return nullptr;
    for(auto& p : pool)
        if(p.n == nfft)
            return &p;
    for(auto& p : pool)
    {
        if(p.n == 0)
        {
            p.n = nfft;
            return &p;
        }
    }
    return nullptr;
}

void FftForward(FftPlan* plan, float* time_scratch, float* spec_out)
{
    const size_t n = plan->n;
    for(size_t i = 0; i < n; i++)
    {
        scratch_re[i] = time_scratch[i];
        scratch_im[i] = 0.0;
    }
    CFft(scratch_re, scratch_im, n, false);
    spec_out[0] = (float)scratch_re[0];     // DC
    spec_out[1] = (float)scratch_re[n / 2]; // Nyquist
    for(size_t k = 1; k < n / 2; k++)
    {
        spec_out[2 * k]     = (float)scratch_re[k];
        spec_out[2 * k + 1] = (float)scratch_im[k];
    }
    // Match CMSIS: input buffer contents are unspecified afterwards.
    memset(time_scratch, 0, n * sizeof(float));
}

void FftInverse(FftPlan* plan, float* spec_scratch, float* time_out)
{
    const size_t n = plan->n;
    scratch_re[0]     = spec_scratch[0];
    scratch_im[0]     = 0.0;
    scratch_re[n / 2] = spec_scratch[1];
    scratch_im[n / 2] = 0.0;
    for(size_t k = 1; k < n / 2; k++)
    {
        scratch_re[k]     = spec_scratch[2 * k];
        scratch_im[k]     = spec_scratch[2 * k + 1];
        scratch_re[n - k] = scratch_re[k];
        scratch_im[n - k] = -scratch_im[k];
    }
    CFft(scratch_re, scratch_im, n, true);
    const double inv_n = 1.0 / (double)n;
    for(size_t i = 0; i < n; i++)
        time_out[i] = (float)(scratch_re[i] * inv_n);
    memset(spec_scratch, 0, n * sizeof(float));
}

// Twiddles are computed on the fly: nothing to relocate.
size_t FftTableBytes()
{
    return 0;
}

bool FftRelocateTables(void* dst, size_t bytes)
{
    (void)bytes;
    return dst != nullptr;
}

} // namespace daisyconvol
