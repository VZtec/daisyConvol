// conv_engine.cpp - see conv_engine.h for the architecture overview.

#include "conv_engine.h"

#include <cfloat>
#include <cmath>
#include <cstring>

namespace daisyconvol
{

float* Arena::Alloc(size_t nfloats)
{
    const size_t n = ConvEngine::AlignF(nfloats);
    if(used + n > size)
        return nullptr;
    float* p = mem + used;
    used += n;
    return p;
}

namespace
{

// No active partitions anywhere: what SwapIR(nullptr) switches to.
constexpr ConvEngine::IrSpectra kSilence{};

const ConvEngine::IrSpectra* Public(const ConvEngine::IrSpectra* p)
{
    return p == &kSilence ? nullptr : p;
}

inline size_t CeilDiv(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

// The target build may be -Os for flash size; force the per-sample kernels
// back to -O2. Also, newlib-nano's memcpy/memset are byte loops, so the hot
// copies below use word-wide loops of our own.
#if defined(__GNUC__) && !defined(__clang__)
#define CONV_HOT __attribute__((optimize("O2")))
#else
#define CONV_HOT
#endif

// acc + x * y and acc - x * y as single fused instructions (VFMA / VFMS)
// where the FPU has them, whatever -ffp-contract says.
#if defined(__GNUC__) && defined(__ARM_FEATURE_FMA)
#define CONV_FMA(acc, x, y) __builtin_fmaf((x), (y), (acc))
#define CONV_FMS(acc, x, y) __builtin_fmaf(-(x), (y), (acc))
#else
#define CONV_FMA(acc, x, y) ((acc) + (x) * (y))
#define CONV_FMS(acc, x, y) ((acc) - (x) * (y))
#endif

// The inner loop of the bin-major MAC (CmacParts) in Thumb-2 / FPv5
// assembly where it can be: 2 = VLDR.64 loads, 1 = VLDM loads, 0 = the C
// loop (e.g. -DDAISYCONVOL_MAC_ASM=0). Measured in cabsim (code in ITCM),
// engine cycles per block: C 45.5k, VLDM 36.2k, VLDR.64 35.5k. VLDR.64 is
// ahead where a stage's spectra are in cached RAM (AXI SRAM); with every
// arena in DTCM, VLDM is ahead by 1 %.
#ifndef DAISYCONVOL_MAC_ASM
#if defined(__GNUC__) && defined(__thumb2__) && defined(__ARM_ARCH_7EM__) \
    && defined(__ARM_FEATURE_FMA)
#define DAISYCONVOL_MAC_ASM 2
#else
#define DAISYCONVOL_MAC_ASM 0
#endif
#endif
// Assembly MAC of the time-sliced stages: PLD the next partition's two cache
// lines while the current one is multiplied (1), or not (0). Their spectra
// and delay lines are in the medium arena, AXI SRAM behind the D-cache in
// the recommended placement, and every line is a miss: in cabsim (code in
// ITCM) the engine goes from 35.4k to 32.4k cycles per block (stage 1 19.2k
// -> 16.1k). With the medium arena in DTCM there is nothing to prefetch and
// the PLDs cost 0.8 % (31.5k -> 31.8k). Stage 0 (fast arena) never PLDs.
#ifndef DAISYCONVOL_MAC_PLD
#define DAISYCONVOL_MAC_PLD 1
#endif

CONV_HOT void ZeroF32(float* __restrict dst, size_t n)
{
    size_t i = 0;
    for(; i + 4 <= n; i += 4)
    {
        dst[i]     = 0.0f;
        dst[i + 1] = 0.0f;
        dst[i + 2] = 0.0f;
        dst[i + 3] = 0.0f;
    }
    for(; i < n; i++)
        dst[i] = 0.0f;
}

CONV_HOT void CopyF32(float* __restrict dst, const float* __restrict src,
                      size_t n)
{
    size_t i = 0;
    for(; i + 4 <= n; i += 4)
    {
        dst[i]     = src[i];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i + 2];
        dst[i + 3] = src[i + 3];
    }
    for(; i < n; i++)
        dst[i] = src[i];
}

CONV_HOT void AddF32(float* __restrict dst, const float* __restrict src,
                     size_t n)
{
    for(size_t i = 0; i < n; i++)
        dst[i] += src[i];
}

// Copies n samples, replacing NaN / infinite ones by 0; returns how many.
CONV_HOT size_t CopySanitized(float* __restrict dst,
                              const float* __restrict src,
                              size_t n)
{
    size_t bad = 0;
    for(size_t i = 0; i < n; i++)
    {
        const float v  = src[i];
        const bool  ok = std::fabs(v) <= FLT_MAX; // false for NaN and +-inf
        dst[i]         = ok ? v : 0.0f;
        bad += ok ? 0 : 1;
    }
    return bad;
}

bool AllFinite(const float* x, size_t n)
{
    for(size_t i = 0; i < n; i++)
        if(!(std::fabs(x[i]) <= FLT_MAX))
            return false;
    return true;
}

// Complex multiply(-accumulate) over pairs of packed bins, two bins per
// iteration (4 fused multiply-adds per bin, two independent dependency
// chains). The Assign variants write instead of accumulating: the first
// partition of each MAC pass uses them, which removes any need to zero the
// accumulators.
CONV_HOT void CmacBins(float* __restrict acc,
                       const float* __restrict a,
                       const float* __restrict b,
                       size_t n)
{
    size_t i = 0;
    for(; i + 4 <= n; i += 4)
    {
        float r0 = acc[i], i0 = acc[i + 1], r1 = acc[i + 2], i1 = acc[i + 3];
        r0         = CONV_FMA(r0, a[i], b[i]);
        r1         = CONV_FMA(r1, a[i + 2], b[i + 2]);
        i0         = CONV_FMA(i0, a[i], b[i + 1]);
        i1         = CONV_FMA(i1, a[i + 2], b[i + 3]);
        r0         = CONV_FMS(r0, a[i + 1], b[i + 1]);
        r1         = CONV_FMS(r1, a[i + 3], b[i + 3]);
        i0         = CONV_FMA(i0, a[i + 1], b[i]);
        i1         = CONV_FMA(i1, a[i + 3], b[i + 2]);
        acc[i]     = r0;
        acc[i + 1] = i0;
        acc[i + 2] = r1;
        acc[i + 3] = i1;
    }
    for(; i < n; i += 2)
    {
        float r = acc[i], im = acc[i + 1];
        r          = CONV_FMA(r, a[i], b[i]);
        im         = CONV_FMA(im, a[i], b[i + 1]);
        r          = CONV_FMS(r, a[i + 1], b[i + 1]);
        im         = CONV_FMA(im, a[i + 1], b[i]);
        acc[i]     = r;
        acc[i + 1] = im;
    }
}

CONV_HOT void CmulBins(float* __restrict acc,
                       const float* __restrict a,
                       const float* __restrict b,
                       size_t n)
{
    size_t i = 0;
    for(; i + 4 <= n; i += 4)
    {
        const float r0 = a[i] * b[i], i0 = a[i] * b[i + 1];
        const float r1 = a[i + 2] * b[i + 2], i1 = a[i + 2] * b[i + 3];
        acc[i]         = CONV_FMS(r0, a[i + 1], b[i + 1]);
        acc[i + 1]     = CONV_FMA(i0, a[i + 1], b[i]);
        acc[i + 2]     = CONV_FMS(r1, a[i + 3], b[i + 3]);
        acc[i + 3]     = CONV_FMA(i1, a[i + 3], b[i + 2]);
    }
    for(; i < n; i += 2)
    {
        const float r = a[i] * b[i], im = a[i] * b[i + 1];
        acc[i]        = CONV_FMS(r, a[i + 1], b[i + 1]);
        acc[i + 1]    = CONV_FMA(im, a[i + 1], b[i]);
    }
}

// Full spectrum in CMSIS rfft packed layout (see fft_backend.h): bins 0
// (DC) and 1 (Nyquist) are real, the rest are complex pairs.
inline void CmacPacked(float* __restrict acc,
                       const float* __restrict a,
                       const float* __restrict b,
                       size_t n)
{
    acc[0] += a[0] * b[0];
    acc[1] += a[1] * b[1];
    CmacBins(acc + 2, a + 2, b + 2, n - 2);
}

inline void CmulPacked(float* __restrict acc,
                       const float* __restrict a,
                       const float* __restrict b,
                       size_t n)
{
    acc[0] = a[0] * b[0];
    acc[1] = a[1] * b[1];
    CmulBins(acc + 2, a + 2, b + 2, n - 2);
}

// The MAC of a whole stage, bin-major: acc = sum over the partitions
// k = 0..na+nb-1 of h_k * x_k, for the floats [i_beg, i_end) of the packed
// spectra. h_k is h + k*n; x_k runs backwards through a delay line from its
// newest slot and wraps once: xa - k*n for k < na (na >= 1), then
// xb - (k-na)*n. For each group of bins the sums stay in registers across
// all the partitions and acc is written once, instead of being read and
// written once per partition. Per bin, the operations and their order are
// CmulPacked's for partition 0, then CmacPacked's for each further one, so
// the result is the same bit for bit as a pass one partition at a time.
//
// The bins go in groups of 8 floats: a whole cache line when the spectra
// are 32-byte aligned (the arenas are), so each line is read in a single
// visit even where the partitions' power-of-two strides make them evict
// each other in the D-cache. i_beg and i_end are multiples of 8; the group
// at 0 holds DC, Nyquist and bins 1-3, the others four bins each.
//
// One complex bin at float k: CmacBins' and CmulBins' operations.
#define CONV_BIN_MAC(r, im, a, b, k)          \
    r  = CONV_FMA(r, (a)[k], (b)[k]);         \
    im = CONV_FMA(im, (a)[k], (b)[k + 1]);    \
    r  = CONV_FMS(r, (a)[k + 1], (b)[k + 1]); \
    im = CONV_FMA(im, (a)[k + 1], (b)[k])
#define CONV_BIN_MUL(r, im, a, b, k)          \
    r  = (a)[k] * (b)[k];                     \
    im = (a)[k] * (b)[k + 1];                 \
    r  = CONV_FMS(r, (a)[k + 1], (b)[k + 1]); \
    im = CONV_FMA(im, (a)[k + 1], (b)[k])

#if DAISYCONVOL_MAC_ASM
// One run of CmacParts' inner loop: ns >= 1 partitions of one group of 8
// floats into its 8 sums, a going up and b going down by stride bytes. The C
// loop compiles to 16 single-float loads that do not pair with the 16 FMAs
// (~47 cycles per partition from DTCM); here they are eight VLDR.64s (or
// two VLDMs), and the FMAs are the same, in the same order per bin, as
// CONV_BIN_MAC's: the output is the same bit for bit (checked on a Daisy
// Seed against the C loop and the partition-at-a-time MAC). With kDc the
// first two floats are DC and Nyquist (real products) instead of a complex
// bin. s0-s15 hold the operands; the sums go wherever the compiler puts
// them (s16-s31).
#if DAISYCONVOL_MAC_ASM == 1
#define CONV_MAC_LOADS                       \
    "vldmia %[a], {s0-s7}\n\t"               \
    "vldmia %[b], {s8-s15}\n\t"              \
    "add %[a], %[a], %[st]\n\t"              \
    "sub %[b], %[b], %[st]\n\t"
#else
#define CONV_MAC_LOADS                       \
    "vldr d0, [%[a]]\n\t"                    \
    "vldr d4, [%[b]]\n\t"                    \
    "vldr d1, [%[a], #8]\n\t"                \
    "vldr d5, [%[b], #8]\n\t"                \
    "vldr d2, [%[a], #16]\n\t"               \
    "vldr d6, [%[b], #16]\n\t"               \
    "vldr d3, [%[a], #24]\n\t"               \
    "vldr d7, [%[b], #24]\n\t"               \
    "add %[a], %[a], %[st]\n\t"              \
    "sub %[b], %[b], %[st]\n\t"
#endif
// After the pointers moved on: the next partition's lines (one partition
// ahead measured best; two ahead, or issued before the loads, lost 2-3 %).
#if DAISYCONVOL_MAC_PLD
#define CONV_MAC_PLD                         \
    "pld [%[a]]\n\t"                         \
    "pld [%[b]]\n\t"
#else
#define CONV_MAC_PLD
#endif
#define CONV_MAC_BIN0                        \
    "vfma.f32 %[r0], s0, s8\n\t"             \
    "vfma.f32 %[i0], s0, s9\n\t"
#define CONV_MAC_BIN0_END                    \
    "vfms.f32 %[r0], s1, s9\n\t"             \
    "vfma.f32 %[i0], s1, s8\n\t"
#define CONV_MAC_DC_NYQUIST                  \
    "vfma.f32 %[r0], s0, s8\n\t"             \
    "vfma.f32 %[i0], s1, s9\n\t"
#define CONV_MAC_BINS13                      \
    "vfma.f32 %[r1], s2, s10\n\t"            \
    "vfma.f32 %[i1], s2, s11\n\t"            \
    "vfma.f32 %[r2], s4, s12\n\t"            \
    "vfma.f32 %[i2], s4, s13\n\t"            \
    "vfma.f32 %[r3], s6, s14\n\t"            \
    "vfma.f32 %[i3], s6, s15\n\t"            \
    "vfms.f32 %[r1], s3, s11\n\t"            \
    "vfma.f32 %[i1], s3, s10\n\t"            \
    "vfms.f32 %[r2], s5, s13\n\t"            \
    "vfma.f32 %[i2], s5, s12\n\t"            \
    "vfms.f32 %[r3], s7, s15\n\t"            \
    "vfma.f32 %[i3], s7, s14\n\t"
#define CONV_MAC_OPERANDS                                              \
    : [r0] "+t"(r0), [i0] "+t"(i0), [r1] "+t"(r1), [i1] "+t"(i1),       \
      [r2] "+t"(r2), [i2] "+t"(i2), [r3] "+t"(r3), [i3] "+t"(i3),       \
      [a] "+r"(a), [b] "+r"(b), [k] "+r"(ns)                            \
    : [st] "r"(stride)                                                  \
    : "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", \
      "s11", "s12", "s13", "s14", "s15", "cc", "memory"
#define CONV_MAC_LOOP(PLD, FIRST, LAST)                                 \
    __asm__ volatile("1:\n\t" CONV_MAC_LOADS PLD FIRST CONV_MAC_BINS13 \
                         LAST "subs %[k], %[k], #1\n\t"                 \
                     "bne 1b\n\t" CONV_MAC_OPERANDS)
template <bool kDc, bool kPld>
CONV_HOT __attribute__((always_inline)) inline void MacRunAsm(float&       r0,
                                                              float&       i0,
                                                              float&       r1,
                                                              float&       i1,
                                                              float&       r2,
                                                              float&       i2,
                                                              float&       r3,
                                                              float&       i3,
                                                              const float* a,
                                                              const float* b,
                                                              size_t       ns,
                                                              size_t stride)
{
    if(kPld && kDc)
        CONV_MAC_LOOP(CONV_MAC_PLD, CONV_MAC_DC_NYQUIST, "");
    else if(kPld)
        CONV_MAC_LOOP(CONV_MAC_PLD, CONV_MAC_BIN0, CONV_MAC_BIN0_END);
    else if(kDc)
        CONV_MAC_LOOP("", CONV_MAC_DC_NYQUIST, "");
    else
        CONV_MAC_LOOP("", CONV_MAC_BIN0, CONV_MAC_BIN0_END);
}
#undef CONV_MAC_LOADS
#undef CONV_MAC_PLD
#undef CONV_MAC_BIN0
#undef CONV_MAC_BIN0_END
#undef CONV_MAC_DC_NYQUIST
#undef CONV_MAC_BINS13
#undef CONV_MAC_OPERANDS
#undef CONV_MAC_LOOP
#endif // DAISYCONVOL_MAC_ASM

template <bool kPld>
CONV_HOT void CmacParts(float* __restrict acc,
                        const float* __restrict h,
                        const float* __restrict xa,
                        size_t na,
                        const float* __restrict xb,
                        size_t nb,
                        size_t n,
                        size_t i_beg,
                        size_t i_end)
{
    // Partition 0 (h, xa) assigns; two runs accumulate the others: the rest
    // of the delay line down to its first slot, then from its last slot.
    const float* ha = h + n;
    const float* xa1 = na > 1 ? xa - n : nullptr;
    const size_t na1 = na - 1;
    const float* hb  = h + na * n;

    size_t i = i_beg;
    if(i == 0)
    {
        float d = h[0] * xa[0];
        float q = h[1] * xa[1];
        float r1, i1, r2, i2, r3, i3;
        CONV_BIN_MUL(r1, i1, h, xa, 2);
        CONV_BIN_MUL(r2, i2, h, xa, 4);
        CONV_BIN_MUL(r3, i3, h, xa, 6);
        for(size_t s = 0; s < 2; s++)
        {
            const float* hs = s ? hb : ha;
            const float* xs = s ? xb : xa1;
            const size_t ns = s ? nb : na1;
#if DAISYCONVOL_MAC_ASM
            if(ns)
                MacRunAsm<true, kPld>(
                    d, q, r1, i1, r2, i2, r3, i3, hs, xs, ns, n * sizeof(float));
#else
            for(size_t j = 0; j < ns; j++)
            {
                const float* a = hs + j * n;
                const float* b = xs - j * n;
                d              = CONV_FMA(d, a[0], b[0]);
                q              = CONV_FMA(q, a[1], b[1]);
                CONV_BIN_MAC(r1, i1, a, b, 2);
                CONV_BIN_MAC(r2, i2, a, b, 4);
                CONV_BIN_MAC(r3, i3, a, b, 6);
            }
#endif
        }
        acc[0] = d;
        acc[1] = q;
        acc[2] = r1;
        acc[3] = i1;
        acc[4] = r2;
        acc[5] = i2;
        acc[6] = r3;
        acc[7] = i3;
        i      = 8;
    }

    for(; i < i_end; i += 8)
    {
        float r0, i0, r1, i1, r2, i2, r3, i3;
        {
            const float* a = h + i;
            const float* b = xa + i;
            CONV_BIN_MUL(r0, i0, a, b, 0);
            CONV_BIN_MUL(r1, i1, a, b, 2);
            CONV_BIN_MUL(r2, i2, a, b, 4);
            CONV_BIN_MUL(r3, i3, a, b, 6);
        }
        for(size_t s = 0; s < 2; s++)
        {
            const float* hs = s ? hb : ha;
            const float* xs = s ? xb : xa1;
            const size_t ns = s ? nb : na1;
#if DAISYCONVOL_MAC_ASM
            if(ns)
                MacRunAsm<false, kPld>(r0,
                                 i0,
                                 r1,
                                 i1,
                                 r2,
                                 i2,
                                 r3,
                                 i3,
                                 hs + i,
                                 xs + i,
                                 ns,
                                 n * sizeof(float));
#else
            for(size_t j = 0; j < ns; j++)
            {
                const float* a = hs + j * n + i;
                const float* b = xs - j * n + i;
                CONV_BIN_MAC(r0, i0, a, b, 0);
                CONV_BIN_MAC(r1, i1, a, b, 2);
                CONV_BIN_MAC(r2, i2, a, b, 4);
                CONV_BIN_MAC(r3, i3, a, b, 6);
            }
#endif
        }
        acc[i]     = r0;
        acc[i + 1] = i0;
        acc[i + 2] = r1;
        acc[i + 3] = i1;
        acc[i + 4] = r2;
        acc[i + 5] = i2;
        acc[i + 6] = r3;
        acc[i + 7] = i3;
    }
}
#undef CONV_BIN_MAC
#undef CONV_BIN_MUL

// The MAC of a stage's ka partitions of spectra h against its
// frequency-domain delay line, whose newest slot is w: partition k pairs
// with slot w - k, or w + slots - k once that would be negative. Floats
// [i_beg, i_end) of acc; nothing with no active partition.
template <bool kPld>
inline void CmacRing(float* __restrict acc,
                     const float* __restrict h,
                     const float* fdl,
                     size_t       n,
                     size_t       w,
                     size_t       slots,
                     size_t       ka,
                     size_t       i_beg,
                     size_t       i_end)
{
    if(ka == 0)
        return;
    const size_t na = ka < w + 1 ? ka : w + 1; // slots w down to 0
    const size_t nb = ka - na;                 // then from slots - 1 down
    CmacParts<kPld>(acc,
                    h,
                    fdl + w * n,
                    na,
                    nb ? fdl + (slots - 1) * n : nullptr,
                    nb,
                    n,
                    i_beg,
                    i_end);
}

// Wrap-safe "x >= y" for uint32 sequence counters.
inline bool SeqGe(uint32_t x, uint32_t y)
{
    return (int32_t)(x - y) >= 0;
}

// Partition spectra of one stage: partition k (IR offset off + k*part) is
// stored as FFT([h[off + k*part .. +part), zeros(part)]). Returns the number
// of partitions the IR reaches (capped at max_parts).
size_t PartitionIR(const float* ir,
                   size_t       len,
                   size_t       off,
                   size_t       part,
                   size_t       max_parts,
                   FftPlan*     plan,
                   float*       scratch,
                   float*       h)
{
    const size_t nfft = 2 * part;
    const size_t rem  = len > off ? len - off : 0;
    size_t       act  = rem ? CeilDiv(rem, part) : 0;
    if(act > max_parts)
        act = max_parts;
    for(size_t k = 0; k < act; k++)
    {
        const size_t n = (rem - k * part) < part ? (rem - k * part) : part;
        memset(scratch, 0, nfft * sizeof(float));
        memcpy(scratch, ir + off + k * part, n * sizeof(float));
        FftForward(plan, scratch, h + k * nfft);
    }
    return act;
}

} // namespace

bool ConvEngine::Init(const Config& cfg, Arena fast, Arena medium, Arena slow)
{
    inited_ = false;
    num_outputs_
        = cfg.num_outputs >= 1 && cfg.num_outputs <= kMaxOutputs
              ? cfg.num_outputs
              : 1;
    if(!ConfigOk(cfg))
        return false;

    const Geometry g = ConfigGeometry(cfg);
    const size_t   b = cfg.block_size;
    const size_t   C = num_outputs_;
    geo_             = g;
    b_               = b;
    n0_              = 2 * b;
    max_ir_          = cfg.max_ir_len;
    external_        = cfg.external_ir;
    sanitize_        = cfg.sanitize_input;
    get_cycles_      = cfg.get_cycles;
    k0_slots_        = g.count[0];

    plan0_  = FftMakePlan(n0_);
    fdl0_   = fast.Alloc(k0_slots_ * n0_);
    in0_    = fast.Alloc(n0_);
    acc0_   = fast.Alloc(n0_);
    bool ok = plan0_ && fdl0_ && in0_ && acc0_;
    for(size_t c = 0; c < C; c++)
    {
        h0_[c]  = fast.Alloc(k0_slots_ * n0_);
        k0a_[c] = 0;
        ok      = ok && h0_[c];
    }

    num_sliced_ = 0;
    rmax_       = 1;
    bg_         = false;
    bg_run_     = false;
    plan2_      = nullptr;
    scr_bg_ = ring2_ = fdl2_ = nullptr;
    p2_ = n2_ = r2_ = k2_slots_ = 0;
    for(size_t c = 0; c < kMaxOutputs; c++)
    {
        h2_own_[c] = acc2_[c] = staging2_[c] = nullptr;
        h2_[c]                               = nullptr;
        k2a_[c]                              = 0;
    }
    for(size_t s = 1; ok && s < g.num_stages; s++)
    {
        const size_t p = g.part[s];
        if(g.IsBackground(s))
        {
            bg_       = true;
            p2_       = p;
            n2_       = 2 * p;
            r2_       = p / b;
            k2_slots_ = g.count[s];
            plan2_    = FftMakePlan(n2_);
            scr_bg_   = fast.Alloc(n2_);
            ring2_    = medium.Alloc(4 * p2_);
            fdl2_     = slow.Alloc(k2_slots_ * n2_);
            ok        = plan2_ && scr_bg_ && ring2_ && fdl2_;
            for(size_t c = 0; ok && c < C; c++)
            {
                acc2_[c]     = fast.Alloc(n2_);
                staging2_[c] = medium.Alloc(2 * p2_);
                h2_own_[c] = external_ ? nullptr : slow.Alloc(k2_slots_ * n2_);
                ok         = acc2_[c] && staging2_[c] && (external_ || h2_own_[c]);
            }
            continue;
        }
        Sliced& st = sliced_[num_sliced_++];
        st         = Sliced();
        st.p       = p;
        st.n       = 2 * p;
        st.r       = p / b;
        st.slots   = g.count[s];
        st.plan    = FftMakePlan(st.n);
        st.in      = fast.Alloc(2 * p);
        st.fdl     = medium.Alloc(st.slots * st.n);
        ok         = st.plan && st.in && st.fdl;
        for(size_t c = 0; ok && c < C; c++)
        {
            st.acc[c]   = fast.Alloc(st.n);
            st.stage[c] = fast.Alloc(2 * p);
            float* own  = external_ ? nullptr : medium.Alloc(st.slots * st.n);
            st.h[c]     = own;
            ok = st.acc[c] && st.stage[c] && (external_ || own);
        }
        if(st.r > rmax_)
            rmax_ = st.r;
    }
    scr_rt_ = fast.Alloc(RtScratchFloats(g));
    if(!ok || !scr_rt_)
        return false;

    // Arenas may live in NOLOAD sections (.dtcmram_bss, .sdram_bss) that the
    // startup code does not clear: zero everything we took.
    memset(fast.mem, 0, fast.used * sizeof(float));
    if(medium.used)
        memset(medium.mem, 0, medium.used * sizeof(float));
    if(slow.used)
        memset(slow.mem, 0, slow.used * sizeof(float));

    // External IR switches latch on a period boundary of the largest stage
    // and complete once every stage has switched; a main-loop stage needs
    // them two of its periods ahead (see LatchIR).
    rl_     = bg_ ? r2_ : rmax_;
    lead_   = bg_ ? 2 * r2_ : rmax_;
    bg_run_ = bg_ && external_;

    ir_len_.store(0, std::memory_order_relaxed);
    ir_pending_.store(nullptr, std::memory_order_relaxed);
    ir_new_ = ir_cur_ = nullptr;
    switching_ = h0_switching_ = false;
    active_pub_.store(nullptr, std::memory_order_relaxed);
    busy_.store(false, std::memory_order_relaxed);
    ResetState(true);
    perf_ = Perf();
    perf_reset_req_.store(false, std::memory_order_relaxed);
    inited_ = true;
    return true;
}

size_t ConvEngine::spectra_floats() const
{
    return num_outputs_ * SpectraFloats(geo_);
}

// sync: audio stopped and not inside DoBackgroundWork (Init, SetIR), so the
// main-loop stage state is cleared here too. Otherwise (Reset() from the
// callback) the main loop clears that state itself (BgReset) and
// acknowledges; until then the stage publishes no frames.
void ConvEngine::ResetState(bool sync)
{
    ZeroF32(fdl0_, k0_slots_ * n0_);
    ZeroF32(in0_, n0_);
    w0_ = k0_slots_ - 1; // advances to slot 0 on the first block
    for(size_t i = 0; i < num_sliced_; i++)
    {
        Sliced& st = sliced_[i];
        ZeroF32(st.fdl, st.slots * st.n);
        ZeroF32(st.in, 2 * st.p);
        for(size_t c = 0; c < num_outputs_; c++)
            ZeroF32(st.stage[c], 2 * st.p);
        st.w    = st.slots - 1;
        st.wbuf = 0;
    }
    blk_     = 0;
    q2_      = 0;
    s2_age_  = 0;
    s2_have_ = false;
    s2_half_ = 0;

    // With the delay lines empty there is no old-IR output left to finish:
    // IR switches still pending or in progress apply now.
    if(external_)
        ApplyPendingIR(sync);

    if(!bg_)
        return;
    // The ring is written by the callback only; a frame the main loop may
    // be copying from it right now is discarded (reset_req_).
    ZeroF32(ring2_, 4 * p2_);
    frames_produced_.store(0, std::memory_order_relaxed);
    if(sync)
    {
        ZeroF32(fdl2_, k2_slots_ * n2_);
        for(size_t c = 0; c < num_outputs_; c++)
            ZeroF32(staging2_[c], 2 * p2_);
        s2_next_ = 0;
        w2_      = 0;
        bg_step_ = kBgIdle;
        results_ready_.store(0, std::memory_order_relaxed);
        reset_seen_ = reset_req_.load(std::memory_order_relaxed);
        reset_ack_.store(reset_seen_, std::memory_order_relaxed);
        bg_sw_seen_ = bg_sw_req_.load(std::memory_order_relaxed);
        bg_sw_ack_.store(bg_sw_seen_, std::memory_order_relaxed);
        s2_wait_ = false;
        if(external_)
        {
            for(size_t c = 0; c < num_outputs_; c++)
            {
                const size_t last = geo_.num_stages - 1;
                h2_[c]  = ir_cur_ ? ir_cur_->h[c][last] : nullptr;
                k2a_[c] = ir_cur_ ? ir_cur_->k[c][last] : 0;
            }
        }
    }
    else if(bg_run_)
    {
        if(external_)
            bg_reset_ir_.store(ir_cur_, std::memory_order_relaxed);
        reset_req_.store(reset_req_.load(std::memory_order_relaxed) + 1,
                         std::memory_order_release);
        s2_wait_ = true;
    }
}

void ConvEngine::Reset()
{
    if(inited_)
        ResetState(false);
}

// Applies whatever SwapIR is pending or in progress to every in-callback
// stage at once (the main-loop stage adopts it on reset, BgReset).
void ConvEngine::ApplyPendingIR(bool sync)
{
    // busy_ goes up before the pending pointer is taken so SwapPending()
    // never reads both as clear in between.
    if(ir_pending_.load(std::memory_order_relaxed) || switching_)
        busy_.store(true);
    const IrSpectra* p = ir_pending_.exchange(nullptr);
    if(!p)
        p = switching_ ? ir_new_ : ir_cur_;
    switching_    = false;
    h0_switching_ = false;
    for(size_t c = 0; c < num_outputs_; c++)
    {
        k0a_[c] = p ? p->k[c][0] : 0;
        if(p)
            CopyF32(h0_[c], p->h[c][0], p->k[c][0] * n0_);
    }
    for(size_t i = 0; i < num_sliced_; i++)
    {
        Sliced& st   = sliced_[i];
        st.switching = false;
        for(size_t c = 0; c < num_outputs_; c++)
        {
            st.h[c]  = p ? p->h[c][i + 1] : nullptr;
            st.ka[c] = p ? p->k[c][i + 1] : 0;
        }
    }
    ir_cur_ = p;
    ir_len_.store(p ? p->len : 0, std::memory_order_relaxed);
    // With a main-loop stage (Reset() while audio runs), the main loop may
    // still read the previous spectra until it acknowledges the reset.
    if(sync || !bg_run_)
    {
        active_pub_.store(Public(p));
        busy_.store(false);
    }
}

bool ConvEngine::SetIR(const float* ir, size_t len)
{
    return num_outputs_ == 1 && SetIR(&ir, &len);
}

bool ConvEngine::SetIR(const float* const* irs, const size_t* lens)
{
    if(!inited_ || external_ || !irs || !lens)
        return false;
    for(size_t c = 0; c < num_outputs_; c++)
        if(!irs[c] || lens[c] == 0 || lens[c] > max_ir_
           || !AllFinite(irs[c], lens[c]))
            return false;

    ResetState(true);
    size_t len_max = 0;
    bool   any_bg  = false;
    for(size_t c = 0; c < num_outputs_; c++)
    {
        const float* ir  = irs[c];
        const size_t len = lens[c];
        k0a_[c] = PartitionIR(ir, len, 0, b_, k0_slots_, plan0_, scr_rt_, h0_[c]);
        for(size_t i = 0; i < num_sliced_; i++)
        {
            Sliced& st = sliced_[i];
            st.ka[c]   = PartitionIR(ir,
                                   len,
                                   geo_.Offset(i + 1),
                                   st.p,
                                   st.slots,
                                   st.plan,
                                   scr_rt_,
                                   const_cast<float*>(st.h[c]));
        }
        if(bg_)
        {
            k2a_[c] = PartitionIR(ir,
                                  len,
                                  geo_.Offset(geo_.num_stages - 1),
                                  p2_,
                                  k2_slots_,
                                  plan2_,
                                  scr_bg_,
                                  h2_own_[c]);
            h2_[c]  = h2_own_[c];
            any_bg  = any_bg || k2a_[c] > 0;
        }
        if(len > len_max)
            len_max = len;
    }
    bg_run_ = any_bg;
    ir_len_.store(len_max, std::memory_order_relaxed);
    perf_ = Perf();
    perf_reset_req_.store(false, std::memory_order_relaxed);
    return true;
}

bool ConvEngine::PrepareIR(const float* const* irs,
                           const size_t*       lens,
                           float*              storage,
                           size_t              storage_floats,
                           float*              scratch,
                           size_t              scratch_floats,
                           IrSpectra*          out) const
{
    // All checks first (including the samples), so a failure leaves the
    // storage and *out untouched; then the same work as the incremental
    // path, in one go.
    if(!inited_ || !external_ || !irs || !lens)
        return false;
    const size_t L = geo_.Length();
    for(size_t c = 0; c < num_outputs_; c++)
        if(!irs[c] || lens[c] == 0
           || !AllFinite(irs[c], lens[c] < L ? lens[c] : L))
            return false;
    PrepareJob job;
    if(!BeginPrepareIR(&job,
                       irs,
                       lens,
                       storage,
                       storage_floats,
                       scratch,
                       scratch_floats,
                       out))
        return false;
    while(PrepareIRStep(&job)) {}
    return job.ok();
}

bool ConvEngine::BeginPrepareIR(PrepareJob*         job,
                                const float* const* irs,
                                const size_t*       lens,
                                float*              storage,
                                size_t              storage_floats,
                                float*              scratch,
                                size_t              scratch_floats,
                                IrSpectra*          out) const
{
    if(!job)
        return false;
    *job = PrepareJob();
    if(!inited_ || !external_ || !irs || !lens || !storage || !scratch
       || !out)
        return false;
    if(storage_floats < spectra_floats()
       || scratch_floats < prepare_scratch_floats())
        return false;
    for(size_t c = 0; c < num_outputs_; c++)
        if(!irs[c] || lens[c] == 0)
            return false;

    // The layout of *out is known up front: the partition pointers and
    // active counts are filled in now, the spectra by the steps.
    const size_t L = geo_.Length();
    *out           = IrSpectra();
    float* h       = storage;
    for(size_t c = 0; c < num_outputs_; c++)
    {
        const size_t len = lens[c] < L ? lens[c] : L;
        job->irs_[c]     = irs[c];
        job->lens_[c]    = len;
        for(size_t st = 0; st < geo_.num_stages; st++)
        {
            const size_t off  = geo_.Offset(st);
            const size_t part = geo_.part[st];
            const size_t rem  = len > off ? len - off : 0;
            size_t       act  = rem ? CeilDiv(rem, part) : 0;
            if(act > geo_.count[st])
                act = geo_.count[st];
            out->h[c][st] = h;
            out->k[c][st] = act;
            job->total_ += act;
            h += AlignF(geo_.count[st] * 2 * part);
        }
        if(len > out->len)
            out->len = len;
    }
    job->storage_ = storage;
    job->scratch_ = scratch;
    job->out_     = out;
    job->state_   = job->total_ ? PrepareJob::kRunning : PrepareJob::kOk;
    return true;
}

bool ConvEngine::PrepareIRStep(PrepareJob* job) const
{
    if(!job || job->state_ != PrepareJob::kRunning)
        return false;
    IrSpectra* out = job->out_;
    // Next partition that exists: (output c, stage s, partition k).
    while(job->c_ < num_outputs_)
    {
        if(job->s_ >= geo_.num_stages)
        {
            job->c_++;
            job->s_ = job->k_ = 0;
        }
        else if(job->k_ >= out->k[job->c_][job->s_])
        {
            job->s_++;
            job->k_ = 0;
        }
        else
            break;
    }
    if(job->c_ >= num_outputs_) // (not reached: finishes on the last step)
    {
        job->state_ = PrepareJob::kOk;
        return false;
    }

    // Partition k of stage s: FFT([h[off + k*part .. +part), zeros(part)]),
    // exactly as PartitionIR does it.
    const size_t c    = job->c_, st = job->s_, k = job->k_;
    const size_t part = geo_.part[st];
    const size_t nfft = 2 * part;
    const size_t off  = geo_.Offset(st);
    const size_t rem  = job->lens_[c] - off;
    const size_t n    = (rem - k * part) < part ? (rem - k * part) : part;
    const float* src  = job->irs_[c] + off + k * part;
    if(!AllFinite(src, n))
    {
        job->state_ = PrepareJob::kFailed;
        return false;
    }
    FftPlan* plan = st == 0                  ? plan0_
                    : geo_.IsBackground(st) ? plan2_
                                            : sliced_[st - 1].plan;
    memset(job->scratch_, 0, nfft * sizeof(float));
    memcpy(job->scratch_, src, n * sizeof(float));
    FftForward(plan, job->scratch_, const_cast<float*>(out->h[c][st]) + k * nfft);

    job->k_++;
    if(++job->steps_ == job->total_)
    {
        job->state_ = PrepareJob::kOk;
        return false;
    }
    return true;
}

bool ConvEngine::PrepareIR(const float* ir,
                           size_t       len,
                           float*       storage,
                           float*       scratch,
                           IrSpectra*   out) const
{
    return num_outputs_ == 1
           && PrepareIR(&ir, &len, storage, SIZE_MAX, scratch, SIZE_MAX, out);
}

void ConvEngine::SwapIR(const IrSpectra* s)
{
    if(external_)
        ir_pending_.store(s ? s : &kSilence, std::memory_order_release);
}

bool ConvEngine::SwapPending() const
{
    // Pending first: the callback raises busy_ before taking the pointer.
    return ir_pending_.load() != nullptr || busy_.load();
}

const ConvEngine::IrSpectra* ConvEngine::ActiveIR() const
{
    return active_pub_.load();
}

// Callback side of SwapIR, run at the start of every block.
//
// A time-sliced stage with period R blocks computes frame f during period
// f+1 and plays it during f+2, so its output lags its coefficients by one of
// its periods; stage 0 has no lag. A new IR latched at block q therefore
// switches each time-sliced stage at T - R and stage 0 at T = q + lead: up
// to block T the output is entirely the old IR's convolution, from T on the
// new one's. Without a main-loop stage q is a period boundary of the
// largest time-sliced stage and lead is that period. The main-loop stage
// plays frame j during its period j+2, and the main loop may already be
// computing the frame completed just before q; so there q is a boundary of
// that stage's period m, lead = two of its periods, and the main loop
// switches at frame m (published with bg_sw_req_ before frame m is), whose
// output starts exactly at T.
void ConvEngine::LatchIR()
{
    for(size_t i = 0; i < num_sliced_; i++)
    {
        Sliced& st = sliced_[i];
        if(st.switching && blk_ == st.switch_at)
        {
            for(size_t c = 0; c < num_outputs_; c++)
            {
                st.h[c]  = st.h_next[c];
                st.ka[c] = st.ka_next[c];
            }
            st.switching = false;
        }
    }
    if(h0_switching_ && blk_ == h0_switch_at_)
    {
        for(size_t c = 0; c < num_outputs_; c++)
        {
            CopyF32(h0_[c], ir_new_->h[c][0], ir_new_->k[c][0] * n0_);
            k0a_[c] = ir_new_->k[c][0];
        }
        h0_switching_ = false;
    }
    if(switching_ && !h0_switching_)
    {
        bool done = !bg_
                    || bg_sw_ack_.load(std::memory_order_acquire)
                           == bg_sw_req_.load(std::memory_order_relaxed);
        for(size_t i = 0; done && i < num_sliced_; i++)
            done = !sliced_[i].switching;
        if(done)
        {
            ir_cur_    = ir_new_;
            switching_ = false;
            if(!s2_wait_)
            {
                active_pub_.store(Public(ir_cur_));
                busy_.store(false);
            }
        }
    }

    if(switching_ || s2_wait_ || (blk_ & (rl_ - 1)) != 0
       || !ir_pending_.load(std::memory_order_relaxed))
        return;
    busy_.store(true); // before taking the pointer, see SwapPending()
    const IrSpectra* p = ir_pending_.exchange(nullptr); // never null here
    ir_new_            = p;
    switching_         = true;
    ir_len_.store(p->len, std::memory_order_relaxed);
    const uint32_t t = blk_ + (uint32_t)lead_;
    for(size_t i = 0; i < num_sliced_; i++)
    {
        Sliced& st = sliced_[i];
        for(size_t c = 0; c < num_outputs_; c++)
        {
            st.h_next[c]  = p->h[c][i + 1];
            st.ka_next[c] = p->k[c][i + 1];
        }
        st.switch_at = t - (uint32_t)st.r;
        st.switching = true;
        if(st.switch_at == blk_) // the largest stage switches right away
        {
            for(size_t c = 0; c < num_outputs_; c++)
            {
                st.h[c]  = st.h_next[c];
                st.ka[c] = st.ka_next[c];
            }
            st.switching = false;
        }
    }
    h0_switch_at_ = t;
    h0_switching_ = true;
    if(bg_)
    {
        const size_t last = geo_.num_stages - 1;
        for(size_t c = 0; c < num_outputs_; c++)
        {
            bg_sw_h_[c] = p->h[c][last];
            bg_sw_k_[c] = p->k[c][last];
        }
        bg_sw_frame_ = q2_; // frame of the period starting now, see above
        bg_sw_req_.store(bg_sw_req_.load(std::memory_order_relaxed) + 1,
                         std::memory_order_release);
    }
}

// One block of a time-sliced stage. Frame f = input samples [f*P, (f+1)*P).
// During period p we compute frame p-1: FFT+FDL push at phase 0, MACs at
// phases 1..R-2, IFFT at phase R-1. Its output segment [(p+1)*P, (p+2)*P)
// is consumed during period p+1 from the staging half not being written.
// The window/FDL work runs even with no active partitions, so an IR switch
// never finds a stale delay line.
void ConvEngine::ProcessSliced(Sliced& st, const float* x)
{
    const size_t ph = blk_ & (st.r - 1);
    if(ph == 0)
    {
        CopyF32(scr_rt_, st.in, 2 * st.p);
        st.w = (st.w + 1 == st.slots) ? 0 : st.w + 1;
        FftForward(st.plan, scr_rt_, st.fdl + st.w * st.n);
        CopyF32(st.in, st.in + st.p, st.p); // slide window
        st.wbuf ^= 1;
    }
    else if(ph <= st.r - 2)
    {
        // MAC chunk: the bins, in groups of 8 floats, spread evenly over
        // the R-2 middle phases (the first groups % (R-2) phases take one
        // more), each phase running all of an output's ka partitions over
        // its bins. Every bin is written once, so acc never needs clearing.
        const size_t m      = st.r - 2;
        const size_t j      = ph - 1;
        const size_t groups = st.n / 8;
        const size_t base   = groups / m, extra = groups % m;
        const size_t g_beg  = j * base + (j < extra ? j : extra);
        const size_t g_end  = g_beg + base + (j < extra ? 1 : 0);
        for(size_t c = 0; c < num_outputs_; c++)
            CmacRing<true>(st.acc[c], // medium arena: prefetch
                           st.h[c],
                           st.fdl,
                           st.n,
                           st.w,
                           st.slots,
                           st.ka[c],
                           8 * g_beg,
                           8 * g_end);
    }
    else // last phase: IFFT and stage the P-sample output segments
    {
        for(size_t c = 0; c < num_outputs_; c++)
        {
            float* dst = st.stage[c] + st.wbuf * st.p;
            if(st.ka[c] == 0)
            {
                ZeroF32(dst, st.p);
                continue;
            }
            FftInverseHalf(st.plan, st.acc[c], dst);
        }
    }
    // Append the incoming block (belongs to the frame in progress).
    CopyF32(st.in + st.p + ph * b_, x, b_);
}

// Callback side of the main-loop stage. Frame j = samples [j*P2, (j+1)*P2):
// the main loop computes it during period j+1 and its output segment is
// played during period j+2. q2_ counts the periods on its own: blk_ / R2
// would restart at 0 when blk_ wraps (2^32 blocks), which the wrap-safe
// comparisons cannot absorb.
void ConvEngine::ProcessBgInput(const float* x, uint32_t ph2)
{
    if(s2_wait_
       && reset_ack_.load(std::memory_order_acquire)
              == reset_req_.load(std::memory_order_relaxed))
    {
        // The main loop has cleared its state after our Reset(): publish
        // the frames completed meanwhile.
        s2_wait_ = false;
        frames_produced_.store(q2_, std::memory_order_release);
        if(external_ && !switching_)
        {
            active_pub_.store(Public(ir_cur_));
            busy_.store(false);
        }
    }
    if(ph2 == 0)
    {
        // Latch once per period: either the whole period reads the staged
        // buffer or the whole period reads silence. This also guarantees
        // the main loop never writes a buffer the callback is reading (see
        // DoBackgroundStep).
        s2_have_ = false;
        if(s2_age_ >= 2)
        {
            const uint32_t needed = q2_ - 2; // frame consumed this period
            if(!s2_wait_
               && SeqGe(results_ready_.load(std::memory_order_acquire),
                        needed + 1))
            {
                s2_have_ = true;
                s2_half_ = needed & 1;
            }
            else
            {
                perf_.s2_underruns++;
            }
        }
    }
    CopyF32(ring2_ + (blk_ & (4 * r2_ - 1)) * b_, x, b_);
    if(ph2 == r2_ - 1)
    {
        q2_++;
        if(s2_age_ < 2)
            s2_age_++;
        if(!s2_wait_)
            frames_produced_.store(q2_, std::memory_order_release);
    }
}

void ConvEngine::Silence(float* const* outs, size_t n)
{
    for(size_t c = 0; c < num_outputs_; c++)
        if(outs[c])
            ZeroF32(outs[c], n);
}

void ConvEngine::ProcessBlock(const float* in, float* out)
{
    ProcessBlock(in, out, b_);
}

void ConvEngine::ProcessBlock(const float* in, float* out, size_t n)
{
    if(num_outputs_ != 1)
    {
        ZeroF32(out, n);
        if(inited_)
        {
            perf_.block_mismatches++;
            perf_seq_.store(perf_seq_.load(std::memory_order_relaxed) + 1,
                            std::memory_order_release);
        }
        return;
    }
    float* const outs[1] = {out};
    Process(in, outs, n);
}

void ConvEngine::ProcessBlock(const float* in, float* const* outs, size_t n)
{
    Process(in, outs, n);
}

void ConvEngine::Process(const float* in, float* const* outs, size_t n)
{
    if(!inited_ || n != b_)
    {
        Silence(outs, n);
        if(inited_)
        {
            perf_.block_mismatches++;
            perf_seq_.store(perf_seq_.load(std::memory_order_relaxed) + 1,
                            std::memory_order_release);
        }
        return;
    }

    const bool     timed = get_cycles_ != nullptr;
    const uint32_t t0    = timed ? get_cycles_() : 0;
    const size_t   B     = b_;
    const size_t   C     = num_outputs_;

    if(perf_reset_req_.load(std::memory_order_relaxed)
       && perf_reset_req_.exchange(false, std::memory_order_acquire))
    {
        perf_.stage0.Reset();
        perf_.stage1_slice.Reset();
        perf_.callback.Reset();
        perf_.s2_underruns     = 0;
        perf_.nonfinite_inputs = 0;
        perf_.block_mismatches = 0;
    }

    if(external_)
        LatchIR();

    // ---------------- input: slide the 2B window, append this block -------
    // Everything below reads the block from x = in0_ + B, so in and out may
    // be the same buffer.
    CopyF32(in0_, in0_ + B, B);
    float* const x = in0_ + B;
    if(sanitize_)
        perf_.nonfinite_inputs += (uint32_t)CopySanitized(x, in, B);
    else
        CopyF32(x, in, B);

    // ---------------- stage 0: uniform partitions, fully in-callback ------
    // FFT the 2B window into the newest FDL slot, MAC each output's active
    // partitions, IFFT; the last B samples are that output's block.
    CopyF32(scr_rt_, in0_, n0_);
    w0_ = (w0_ + 1 == k0_slots_) ? 0 : w0_ + 1;
    FftForward(plan0_, scr_rt_, fdl0_ + w0_ * n0_);
    for(size_t c = 0; c < C; c++)
    {
        const size_t ka = k0a_[c];
        if(ka == 0)
        {
            ZeroF32(outs[c], B);
            continue;
        }
        CmacRing<false>( // the fast arena: DTCM, nothing to prefetch
            acc0_,
            h0_[c],
            fdl0_,
            n0_,
            w0_,
            k0_slots_,
            ka,
            0,
            n0_);
        FftInverseHalf(plan0_, acc0_, outs[c]); // overlap-save: last B
    }

    const uint32_t t1 = timed ? get_cycles_() : 0;

    // ---------------- time-sliced stages ----------------------------------
    for(size_t i = 0; i < num_sliced_; i++)
        ProcessSliced(sliced_[i], x);

    const uint32_t t2 = timed ? get_cycles_() : 0;

    // ---------------- main-loop stage: publish input, latch results -------
    const uint32_t ph2 = bg_ ? blk_ & (uint32_t)(r2_ - 1) : 0;
    if(bg_run_)
        ProcessBgInput(x, ph2);

    // ---------------- sum stage outputs ------------------------------------
    for(size_t i = 0; i < num_sliced_; i++)
    {
        const Sliced& st = sliced_[i];
        const size_t  ph = blk_ & (st.r - 1);
        for(size_t c = 0; c < C; c++)
            AddF32(outs[c], st.stage[c] + (st.wbuf ^ 1) * st.p + ph * B, B);
    }
    if(bg_run_ && s2_have_)
        for(size_t c = 0; c < C; c++)
            AddF32(outs[c], staging2_[c] + s2_half_ * p2_ + ph2 * B, B);

    blk_++;

    if(timed)
    {
        const uint32_t t3 = get_cycles_();
        perf_.stage0.Update(t1 - t0);
        perf_.stage1_slice.Update(t2 - t1);
        perf_.callback.Update(t3 - t0);
        cb_cycles_.store(cb_cycles_.load(std::memory_order_relaxed) + (t3 - t0),
                         std::memory_order_relaxed);
    }
    perf_seq_.store(perf_seq_.load(std::memory_order_relaxed) + 1,
                    std::memory_order_release);
}

void ConvEngine::DoBackgroundWork()
{
    while(DoBackgroundStep()) {}
}

// Reset() from the callback: drop the frame in progress and everything only
// the main loop may touch, adopt the callback's IR (external mode), then
// acknowledge.
void ConvEngine::BgReset(uint32_t req)
{
    ZeroF32(fdl2_, k2_slots_ * n2_);
    s2_next_ = 0;
    w2_      = 0;
    bg_step_ = kBgIdle;
    results_ready_.store(0, std::memory_order_relaxed);
    if(external_)
    {
        const IrSpectra* p    = bg_reset_ir_.load(std::memory_order_relaxed);
        const size_t     last = geo_.num_stages - 1;
        for(size_t c = 0; c < num_outputs_; c++)
        {
            h2_[c]  = p ? p->h[c][last] : nullptr;
            k2a_[c] = p ? p->k[c][last] : 0;
        }
        // A switch request still open is superseded by that IR.
        bg_sw_seen_ = bg_sw_req_.load(std::memory_order_acquire);
        bg_sw_ack_.store(bg_sw_seen_, std::memory_order_relaxed);
    }
    reset_seen_ = req;
    reset_ack_.store(req, std::memory_order_release);
}

// Partition k of every output against its FDL slot. The MAC streams the
// spectra and the delay line from SDRAM; reading both directly interleaves
// far-apart streams at cache-line granularity, which thrashes SDRAM rows
// (~120 cyc/bin measured). Instead each half-spectrum chunk is staged into
// fast memory with sequential copies (the delay line chunk once for all
// outputs; scr_bg_ is free between the FFT and the IFFTs), then multiplied
// from there.
void ConvEngine::BgMacPartition(size_t k)
{
    size_t idx = w2_ + k2_slots_ - k;
    if(idx >= k2_slots_)
        idx -= k2_slots_;
    const float* fsrc = fdl2_ + idx * n2_;
    const size_t half = n2_ / 2;
    float*       fc   = scr_bg_;
    float*       hc   = scr_bg_ + half;
    for(size_t part = 0; part < 2; part++)
    {
        CopyF32(fc, fsrc + part * half, half);
        for(size_t c = 0; c < num_outputs_; c++)
        {
            if(k >= k2a_[c])
                continue;
            CopyF32(hc, h2_[c] + k * n2_ + part * half, half);
            float* acc = acc2_[c] + part * half;
            if(k == 0 && part == 0)
                CmulPacked(acc, hc, fc, half);
            else if(k == 0)
                CmulBins(acc, hc, fc, half);
            else if(part == 0)
                CmacPacked(acc, hc, fc, half);
            else
                CmacBins(acc, hc, fc, half);
        }
    }
}

void ConvEngine::BgFinishFrame()
{
    s2_next_ = bg_j_ + 1;
    w2_      = (w2_ + 1 == k2_slots_) ? 0 : w2_ + 1;
    bg_step_ = kBgIdle;
}

bool ConvEngine::DoBackgroundStep()
{
    if(!inited_ || !bg_run_)
        return false;

    const bool     timed = get_cycles_ != nullptr;
    const uint32_t w0    = timed ? get_cycles_() : 0;
    const uint32_t c0    = cb_cycles_.load(std::memory_order_relaxed);

    const uint32_t req = reset_req_.load(std::memory_order_acquire);
    if(req != reset_seen_)
    {
        BgReset(req);
        return true;
    }

    bool frame_done = false;
    switch(bg_step_)
    {
        case kBgIdle:
        {
            const uint32_t produced
                = frames_produced_.load(std::memory_order_acquire);
            if(SeqGe(s2_next_, produced))
                return false; // nothing to do
            const uint32_t j = s2_next_;
            bg_j_            = j;
            bg_frame_cycles_ = 0;

            // External IR switch from this frame on (see LatchIR). Every
            // earlier frame is done, so the old spectra are free once the
            // new ones are in place.
            if(external_)
            {
                const uint32_t sw = bg_sw_req_.load(std::memory_order_acquire);
                if(sw != bg_sw_seen_ && SeqGe(j, bg_sw_frame_))
                {
                    for(size_t c = 0; c < num_outputs_; c++)
                    {
                        h2_[c]  = bg_sw_h_[c];
                        k2a_[c] = bg_sw_k_[c];
                    }
                    bg_sw_seen_ = sw;
                    bg_sw_ack_.store(sw, std::memory_order_release);
                }
            }

            // Input window = samples [(j-1)*P2, (j+1)*P2). The ring holds 4
            // periods, so the window is two adjacent P2 quarters; it stays
            // valid while the writer has not entered period j+3
            // (produced <= j+2). The ring is zeroed on reset, so frame 0
            // needs no special case.
            bool ok = !SeqGe(produced, j + 3);
            if(ok)
            {
                CopyF32(scr_bg_, ring2_ + ((j - 1) & 3) * p2_, p2_);
                CopyF32(scr_bg_ + p2_, ring2_ + (j & 3) * p2_, p2_);
                // Re-validate after copying: if the writer advanced into our
                // window mid-copy the snapshot may be torn. The fence keeps
                // the copy's loads before the re-check (seqlock pattern).
                std::atomic_thread_fence(std::memory_order_acquire);
                ok = !SeqGe(frames_produced_.load(std::memory_order_acquire),
                            j + 3);
            }
            float* fdls = fdl2_ + w2_ * n2_;
            if(!ok)
            {
                // Way too late: input gone. Zero the FDL slot (equivalent to
                // a dropped input frame - bounded artifact, no garbage) and
                // skip the output.
                ZeroF32(fdls, n2_);
                perf_.s2_dropped_inputs++;
                BgFinishFrame();
                frame_done = true;
                break;
            }
            FftForward(plan2_, scr_bg_, fdls);
            bg_k_    = 0;
            bg_step_ = kBgMac;
            break;
        }
        case kBgMac:
        {
            size_t kmax = 0;
            for(size_t c = 0; c < num_outputs_; c++)
                if(k2a_[c] > kmax)
                    kmax = k2a_[c];
            size_t kend = bg_k_ + kBgPartsPerStep;
            if(kend > kmax)
                kend = kmax;
            for(size_t k = bg_k_; k < kend; k++)
                BgMacPartition(k);
            bg_k_ = kend;
            if(bg_k_ >= kmax)
            {
                bg_c_    = 0;
                bg_step_ = kBgInverse;
            }
            break;
        }
        case kBgInverse:
        {
            // Output segment [(j+2)*P2, (j+3)*P2) is latched by the callback
            // at phase 0 of period j+2. The ping-pong buffer (j & 1) was
            // last consumed during period j (over: frame j is complete), so
            // no reader can hold it now.
            const size_t c   = bg_c_;
            float*       dst = staging2_[c] + (bg_j_ & 1) * p2_;
            if(k2a_[c] == 0)
                ZeroF32(dst, p2_);
            else
                FftInverseHalf(plan2_, acc2_[c], dst);
            if(++bg_c_ < num_outputs_)
                break;
            // A Reset() while we worked makes this frame meaningless: leave
            // it unpublished, the next step clears the state.
            if(reset_req_.load(std::memory_order_acquire) != reset_seen_)
                break;
            results_ready_.store(bg_j_ + 1, std::memory_order_release);
            if(SeqGe(frames_produced_.load(std::memory_order_acquire),
                     bg_j_ + 3))
                perf_.s2_late_skips++; // its whole output period is over
            BgFinishFrame();
            frame_done = true;
            break;
        }
    }

    if(timed)
    {
        // Own cycles only: the audio callback may have preempted this step.
        const uint32_t wall = get_cycles_() - w0;
        const uint32_t cb   = cb_cycles_.load(std::memory_order_relaxed) - c0;
        bg_frame_cycles_ += wall > cb ? wall - cb : 0;
        if(frame_done)
            perf_.stage2_frame.Update(bg_frame_cycles_);
    }
    return true;
}

Perf ConvEngine::perf() const
{
    Perf p;
    for(;;)
    {
        const uint32_t seq = perf_seq_.load(std::memory_order_acquire);
        p                  = perf_;
        std::atomic_thread_fence(std::memory_order_acquire);
        if(perf_seq_.load(std::memory_order_relaxed) == seq)
            break;
    }
    if(perf_reset_req_.load(std::memory_order_acquire))
    {
        // Cleared by ResetPerf(); the callback has not got to it yet.
        p.stage0.Reset();
        p.stage1_slice.Reset();
        p.callback.Reset();
        p.s2_underruns     = 0;
        p.nonfinite_inputs = 0;
        p.block_mismatches = 0;
    }
    return p;
}

void ConvEngine::ResetPerf()
{
    perf_.stage2_frame.Reset();
    perf_.s2_late_skips     = 0;
    perf_.s2_dropped_inputs = 0;
    perf_reset_req_.store(true, std::memory_order_release);
}

} // namespace daisyconvol
