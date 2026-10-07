// fft_backend.h
// Real-FFT backend interface for the partitioned convolution engine.
//
// Two implementations exist:
//   - fft_backend_cmsis.cpp : target build, wraps CMSIS-DSP arm_rfft_fast_f32
//                             (also builds on a PC for the test harness).
//   - fft_backend_ref.cpp   : desktop/test build, portable double-precision
//                             reference producing the exact same packed layout.
//
// Spectrum layout (CMSIS "rfft_fast" packing, N real samples -> N floats):
//   spec[0]          = X[0].re        (DC, real)
//   spec[1]          = X[N/2].re      (Nyquist, real)
//   spec[2k], spec[2k+1] = X[k].re, X[k].im   for k = 1 .. N/2-1
//
// Scaling: forward is unscaled, inverse includes the 1/N factor, so
// IFFT(FFT(x) * FFT(h)) yields the (circular) convolution of x and h.

#pragma once

#include <cstddef>

namespace daisyconvol
{

struct FftPlan; // opaque, defined by the backend

// Returns a plan for an N-point real FFT from a small static pool
// (no heap). Plans are shared: asking twice for the same size returns
// the same plan, and the pool holds every supported size. Returns nullptr
// if the size is unsupported. Not RT-safe; call at init time only.
FftPlan* FftMakePlan(size_t nfft);

// time_scratch: nfft floats, DESTROYED by the call. spec_out: nfft floats.
void FftForward(FftPlan* plan, float* time_scratch, float* spec_out);

// spec_scratch: nfft floats, DESTROYED by the call. time_out: nfft floats.
// Includes 1/N scaling.
void FftInverse(FftPlan* plan, float* spec_scratch, float* time_out);

// Bytes of twiddle / bit-reversal tables used by the plans created so far
// and not relocated yet (0 if the backend has no tables).
size_t FftTableBytes();

// Bytes of those tables for one nfft-point FFT in the CMSIS backend (0 for
// an unsupported size), for sizing a static buffer: the complex twiddles of
// the nfft/2-point CFFT and the real-stage twiddles (nfft floats each), plus
// the CFFT's bit-reversal table (uint16 entries).
constexpr size_t FftTableBytesFor(size_t nfft)
{
    const size_t bitrev = nfft == 32     ? 20
                          : nfft == 64   ? 48
                          : nfft == 128  ? 56
                          : nfft == 256  ? 208
                          : nfft == 512  ? 440
                          : nfft == 1024 ? 448
                          : nfft == 2048 ? 1800
                          : nfft == 4096 ? 3808
                                         : 0;
    return bitrev ? 2 * nfft * sizeof(float) + bitrev * 2 : 0;
}

// Copies those tables to dst (at least FftTableBytes() bytes, 4-byte
// aligned, e.g. AXI SRAM or DTCM) and makes the plans use the copies. With
// the app running from QSPI flash every FFT otherwise reads its tables
// through the QSPI interface. dst must stay valid for good. Call after the
// last ConvEngine::Init, with audio stopped and no DoBackgroundWork
// running. Returns false (and changes nothing) if dst is too small.
bool FftRelocateTables(void* dst, size_t bytes);

} // namespace daisyconvol
