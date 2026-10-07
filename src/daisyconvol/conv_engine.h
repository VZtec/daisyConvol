// conv_engine.h
// Non-uniform partitioned convolution engine (Gardner / zita-convolver style)
// for single-core embedded targets. Hardware independent: all memory is
// provided by the caller through three arenas, all FFT work goes through
// fft_backend.h, cycle counting is an injected function pointer.
//
// Partitioning (B = audio block size, a power of two, 16..2048):
//   stage 0     partition B, computed completely in every callback: the head
//               of the IR, with zero added latency
//   stages 1..  larger partitions P (powers of two, <= 2048), each starting
//               at IR offset 2P. Either time-sliced across the P/B callbacks
//               of its period (FFT, MACs, IFFT: deadline-exact, inside the
//               callback) or, for the last stage only, run in the main loop
//               through DoBackgroundWork() / DoBackgroundStep() with a
//               lock-free SPSC handoff and a deadline window of one full
//               period.
//
// The 2P offset rule is what gives every stage a full period of scheduling
// slack, and it fixes how many partitions each intermediate stage has
// (2 * P_next / P - 2); only the last stage grows with the IR length.
// MakeGeometry() picks a geometry from (B, max IR length) - x8 steps, each
// stage in the callback when its FFT is cheap enough for the block budget -
// or the caller supplies one (Config::geometry). For B = 32 the default is
// 16 x 32, 14 x 256, then 2048-sample partitions in the main loop.
//
// One input, up to kMaxOutputs outputs (Config::num_outputs), each with its
// own IR: the input FFTs and delay lines are shared, e.g. a stereo reverb
// from a mono guitar for much less than two engines.
//
// RT guarantees inside ProcessBlock(): no allocation, no locks, no blocking
// calls, bounded worst case (stage 0 + one slice of each time-sliced stage +
// O(B) copies).
//
// Threading model: ProcessBlock() runs in the audio interrupt,
// DoBackgroundWork() in the main loop. Communication is single-producer/
// single-consumer via std::atomic<uint32_t> counters (lock-free on
// Cortex-M7) plus ping-pong output buffers; Reset() from the callback is a
// request/acknowledge handshake with the main loop. Init/SetIR must be
// called with audio stopped.
//
// External IR mode (Config::external_ir): instead of SetIR, the caller
// precomputes the partition spectra of any number of IRs with PrepareIR()
// into its own storage, then switches between them while audio runs with
// SwapIR(). The switch keeps the input delay lines and is exact: output is
// the old IR's convolution up to a period boundary and the new IR's from
// there on (a hard cut, no gap). With a main-loop stage the switch is
// announced two of its periods ahead, so that stage switches too.
//
// C++14, no exceptions, no RTTI, no heap.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "fft_backend.h"

namespace daisyconvol
{

// Simple bump allocator over caller-provided float storage.
struct Arena
{
    float* mem  = nullptr;
    size_t size = 0; // in floats
    size_t used = 0;

    // Chunk sizes are rounded up to 8 floats, so every chunk is 32-byte
    // (cache line) aligned when mem is: declare the backing arrays
    // alignas(32).
    float* Alloc(size_t nfloats);
};

struct StagePerf
{
    uint32_t last  = 0;
    uint32_t peak  = 0;
    uint64_t total = 0;
    uint32_t count = 0;

    void Update(uint32_t cycles)
    {
        last = cycles;
        if(cycles > peak)
            peak = cycles;
        total += cycles;
        count++;
    }
    uint32_t Avg() const { return count ? (uint32_t)(total / count) : 0; }
    void     Reset() { *this = StagePerf(); }
};

struct Perf
{
    StagePerf stage0;       // per callback
    StagePerf stage1_slice; // per callback: all time-sliced stages together
    StagePerf callback;     // whole ProcessBlock
    // Per main-loop stage frame: cycles DoBackgroundWork/Step spent on it,
    // minus the time ProcessBlock took meanwhile (other interrupts that
    // preempt the main loop still count).
    StagePerf stage2_frame;

    // Periods where the main-loop stage result was not ready at its deadline
    // (main loop too slow / not called often enough). Audio degrades to
    // "missing tail segment", never corrupts.
    uint32_t s2_underruns = 0;
    // Frames the main loop finished only after their whole output period had
    // passed: computed for nothing (that period is also an underrun).
    uint32_t s2_late_skips = 0;
    // Frames whose *input* window was already overwritten in the ring
    // (main loop more than two periods behind); their FDL slot is zeroed.
    uint32_t s2_dropped_inputs = 0;
    // Input samples that were NaN or infinite, replaced by 0
    // (Config::sanitize_input).
    uint32_t nonfinite_inputs = 0;
    // ProcessBlock calls with the wrong length, or the single-output
    // overload on a multi-output engine: that output was silenced.
    uint32_t block_mismatches = 0;
};

// Partition layout. Stage s covers IR samples
// [Offset(s), Offset(s) + count[s] * part[s]).
struct Geometry
{
    static constexpr size_t kMaxStages = 6;
    static constexpr size_t kMaxPart   = 2048; // 4096-point FFT (CMSIS limit)
    static constexpr size_t kMinBlock  = 16;   // 32-point FFT (CMSIS limit)

    size_t block      = 0;
    size_t num_stages = 0;
    size_t part[kMaxStages]  = {}; // partition size, part[0] == block
    size_t count[kMaxStages] = {}; // partitions
    bool   background = false;     // the last stage runs in the main loop

    constexpr size_t Offset(size_t s) const { return s == 0 ? 0 : 2 * part[s]; }
    constexpr size_t Ratio(size_t s) const { return part[s] / block; }
    constexpr size_t Length() const
    {
        return num_stages ? Offset(num_stages - 1)
                                + count[num_stages - 1] * part[num_stages - 1]
                          : 0;
    }
    constexpr bool IsBackground(size_t s) const
    {
        return background && s + 1 == num_stages && s > 0;
    }

    static constexpr bool Pow2(size_t x) { return x && !(x & (x - 1)); }

    // Checks every structural rule the engine relies on.
    constexpr bool Valid() const
    {
        if(!Pow2(block) || block < kMinBlock || block > kMaxPart)
            return false;
        if(num_stages == 0 || num_stages > kMaxStages || part[0] != block)
            return false;
        // stage 0 always runs in the callback
        if(background && num_stages < 2)
            return false;
        for(size_t s = 0; s < num_stages; s++)
        {
            if(!Pow2(part[s]) || part[s] > kMaxPart || count[s] == 0)
                return false;
            if(s > 0 && part[s] <= part[s - 1])
                return false;
            // time-sliced stages need FFT, >= 1 MAC and IFFT phases; the
            // main-loop stage needs a period of >= 2 blocks
            if(s > 0 && Ratio(s) < (IsBackground(s) ? 2u : 4u))
                return false;
            if(s + 1 < num_stages
               && Offset(s) + count[s] * part[s] != Offset(s + 1))
                return false;
        }
        return true;
    }
};

namespace detail
{
// C++14 stand-in for std::atomic<T>::is_always_lock_free (C++17), for
// integer types: the ATOMIC_*_LOCK_FREE macro of the same size (2 = always).
template <typename T>
constexpr bool IntAlwaysLockFree()
{
    return sizeof(T) == sizeof(int)         ? ATOMIC_INT_LOCK_FREE == 2
           : sizeof(T) == sizeof(long)      ? ATOMIC_LONG_LOCK_FREE == 2
           : sizeof(T) == sizeof(long long) ? ATOMIC_LLONG_LOCK_FREE == 2
                                            : false;
}
} // namespace detail

class ConvEngine
{
  public:
    // Outputs sharing one input (Config::num_outputs).
    static constexpr size_t kMaxOutputs = 4;
    // CPU cycles per sample the default geometry budgets for: 480 MHz / 48 kHz.
    static constexpr uint32_t kDefaultCyclesPerSample = 10000;

    struct Config
    {
        size_t block_size = 32;     // B: power of two, 16..2048
        size_t max_ir_len = 144000; // samples: sizes the geometry and memory
        // Allow a main-loop stage (DoBackgroundWork) for long IRs. Without
        // it everything runs in the callback, which for long IRs at small
        // blocks is much more CPU per callback.
        bool allow_background = true;
        // IR spectra are provided by the caller (PrepareIR + SwapIR) instead
        // of SetIR.
        bool external_ir = false;
        // Custom geometry (must be Valid(), block == block_size and cover
        // max_ir_len). nullptr: MakeGeometry(block_size, max_ir_len, ...).
        const Geometry* geometry = nullptr;
        // Optional cycle counter (e.g. DWT->CYCCNT). nullptr disables perf.
        uint32_t (*get_cycles)() = nullptr;
        // Outputs, each convolving the same input with its own IR (1..4).
        size_t num_outputs = 1;
        // Replace NaN / infinite input samples by 0 (counted in
        // Perf::nonfinite_inputs); one bad sample would otherwise silence
        // or blow up the output for the whole IR length. ~2 cycles/sample.
        bool sanitize_input = true;
        // External IR mode: let the default geometry use a main-loop stage
        // for long IRs (it switches too, see SwapIR). Off: long external IRs
        // extend the last in-callback stage instead (the pre-existing
        // behavior). Ignored with a custom geometry.
        bool external_background = false;
        // CPU budget the default geometry assumes, in cycles per sample
        // (clock / sample rate): e.g. 5000 at 96 kHz.
        uint32_t cycles_per_sample = kDefaultCyclesPerSample;
    };

    static constexpr size_t AlignF(size_t n) { return (n + 7u) & ~size_t(7); }

    // Rough CMSIS real-FFT cost model (cycles), used to decide whether a
    // stage fits in the callback: in-callback if one FFT <= 15 % of a block's
    // budget (cycles_per_sample * b).
    static constexpr size_t FftCost(size_t n)
    {
        size_t lg = 0;
        while((size_t(1) << lg) < n)
            lg++;
        return 7 * n * lg;
    }
    static constexpr bool
    FitsInCallback(size_t   b,
                   size_t   part,
                   uint32_t cycles_per_sample = kDefaultCyclesPerSample)
    {
        return FftCost(2 * part) * 100 <= 15 * size_t(cycles_per_sample) * b;
    }

    // A time-sliced (in-callback) last stage reads its whole spectra and
    // delay line from memory every period; past this size it would not fit
    // AXI SRAM next to anything else, so MakeGeometry moves it to the main
    // loop, whose MAC streams SDRAM efficiently.
    static constexpr size_t kMaxSlicedBytes = 256 * 1024;

    // Default geometry: partitions grow x8 per stage up to 2048 samples. A
    // stage runs time-sliced in the callback when its FFT fits the block
    // budget and its period is >= 4 blocks; otherwise it becomes the last
    // stage, in the main loop, with 2048-sample partitions. Without
    // allow_background the previous stage is extended instead. Returns an
    // invalid geometry for unsupported b.
    static constexpr Geometry
    MakeGeometry(size_t   b,
                 size_t   max_ir,
                 bool     allow_background  = true,
                 uint32_t cycles_per_sample = kDefaultCyclesPerSample)
    {
        Geometry g;
        g.block = b;
        if(!Geometry::Pow2(b) || b < Geometry::kMinBlock
           || b > Geometry::kMaxPart || max_ir == 0)
            return g;
        g.part[0]    = b;
        g.num_stages = 1;
        for(;;)
        {
            const size_t s   = g.num_stages - 1;
            const size_t p   = g.part[s];
            const size_t off = g.Offset(s);
            size_t pn = 8 * p < Geometry::kMaxPart ? 8 * p : Geometry::kMaxPart;
            bool   next    = pn > p && g.num_stages < Geometry::kMaxStages;
            bool   next_bg = false;
            if(next
               && !(FitsInCallback(b, pn, cycles_per_sample) && pn / b >= 4))
            {
                if(allow_background)
                {
                    // The main-loop stage is bound by memory bandwidth
                    // (spectra + delay line streamed from SDRAM once per
                    // frame): the largest partition moves the least data
                    // per second. The stage before it gets more partitions.
                    next_bg = true;
                    pn      = Geometry::kMaxPart;
                }
                else
                    next = false;
            }
            const size_t rem = max_ir > off ? max_ir - off : 0;
            if(!next || max_ir <= 2 * pn)
            {
                g.count[s] = rem ? (rem + p - 1) / p : 1;
                break;
            }
            g.count[s]           = (2 * pn - off) / p;
            g.part[g.num_stages] = pn;
            g.num_stages++;
            if(next_bg)
            {
                g.background              = true;
                g.count[g.num_stages - 1] = (max_ir - 2 * pn + pn - 1) / pn;
                break;
            }
        }
        const size_t last = g.num_stages - 1;
        if(allow_background && last > 0 && !g.background
           && g.count[last] * 2 * g.part[last] * 2 * sizeof(float)
                  > kMaxSlicedBytes)
            g.background = true;
        return g;
    }

    // The geometry Init() uses for cfg.
    static constexpr Geometry ConfigGeometry(const Config& c)
    {
        return c.geometry ? *c.geometry
                          : MakeGeometry(c.block_size,
                                         c.max_ir_len,
                                         c.allow_background
                                             && (!c.external_ir
                                                 || c.external_background),
                                         c.cycles_per_sample);
    }

    // IR coverage of the default in-callback stages 0 and 1 (16 + 14
    // partitions): 4096 samples at B = 32.
    static constexpr size_t S01Len(size_t b) { return 128 * b; }

    // ---- Arena sizes, in floats, for statically sizing the backing arrays.
    // fast   -> DTCM      (hot per-callback state + scratch)
    // medium -> AXI SRAM  (time-sliced stages' spectra/FDL, main-loop stage
    //                      ring & staging)
    // slow   -> SDRAM     (main-loop stage IR spectra + delay line)
    // Where each arena lives is up to the caller. 0 = arena not needed (pass
    // an empty Arena; zero-length arrays are not valid C++). All sizes are 0
    // for an invalid configuration.
    //
    // Preferred: from the same Config passed to Init(), e.g.
    //   constexpr CE::Config kCfg = [] { CE::Config c; c.block_size = 32;
    //                                    c.max_ir_len = 96000; return c; }();
    //   alignas(32) float fast_mem[CE::FastFloats(kCfg)];
    static constexpr size_t FastFloats(const Config& c)
    {
        return ConfigOk(c) ? FastFloatsFor(ConfigGeometry(c), c.num_outputs)
                           : 0;
    }
    static constexpr size_t MediumFloats(const Config& c)
    {
        return ConfigOk(c) ? MediumFloatsFor(ConfigGeometry(c),
                                             c.num_outputs,
                                             c.external_ir)
                           : 0;
    }
    static constexpr size_t SlowFloats(const Config& c)
    {
        return ConfigOk(c) ? SlowFloatsFor(
                   ConfigGeometry(c), c.num_outputs, c.external_ir)
                           : 0;
    }
    // External IR mode: storage for one IrSpectra (all outputs).
    static constexpr size_t SpectraFloats(const Config& c)
    {
        return ConfigOk(c) ? c.num_outputs * SpectraFloats(ConfigGeometry(c))
                           : 0;
    }
    // Scratch for PrepareIR (the largest FFT of the geometry).
    static constexpr size_t PrepareScratchFloats(const Config& c)
    {
        return ConfigOk(c) ? MaxFftFloats(ConfigGeometry(c)) : 0;
    }
    // Buffer size, in bytes, for daisyconvol::FftRelocateTables() when this
    // is the only engine (its FFT sizes; see fft_backend.h).
    static constexpr size_t FftTableBytes(const Config& c)
    {
        size_t n = 0;
        for(size_t s = 0; ConfigOk(c) && s < ConfigGeometry(c).num_stages; s++)
            n += FftTableBytesFor(2 * ConfigGeometry(c).part[s]);
        return n;
    }

    // Single output, from a geometry.
    static constexpr size_t FastFloats(const Geometry& g)
    {
        return FastFloatsFor(g, 1);
    }
    static constexpr size_t MediumFloats(const Geometry& g,
                                         bool            external_ir = false)
    {
        return MediumFloatsFor(g, 1, external_ir);
    }
    static constexpr size_t SlowFloats(const Geometry& g,
                                       bool            external_ir = false)
    {
        return SlowFloatsFor(g, 1, external_ir);
    }
    static constexpr size_t SpectraFloats(const Geometry& g)
    {
        size_t n = 0;
        for(size_t s = 0; g.Valid() && s < g.num_stages; s++)
            n += AlignF(g.count[s] * 2 * g.part[s]);
        return n;
    }
    static constexpr size_t PrepareScratchFloats(const Geometry& g)
    {
        return MaxFftFloats(g);
    }

    // Default geometry of a single-output config with these fields (the
    // historical rule: external IR mode never gets a main-loop stage here).
    static constexpr Geometry DefaultGeometry(size_t b,
                                              size_t max_ir,
                                              bool   allow_background,
                                              bool   external_ir)
    {
        return MakeGeometry(b, max_ir, allow_background && !external_ir);
    }

    // Single output, SetIR mode, main-loop stage allowed.
    static constexpr size_t FastFloats(size_t b, size_t max_ir)
    {
        return FastFloats(DefaultGeometry(b, max_ir, true, false));
    }
    static constexpr size_t MediumFloats(size_t b, size_t max_ir)
    {
        return MediumFloats(DefaultGeometry(b, max_ir, true, false));
    }
    static constexpr size_t SlowFloats(size_t b, size_t max_ir)
    {
        return SlowFloats(DefaultGeometry(b, max_ir, true, false));
    }
    // External IR mode, single output, default geometry (no main-loop
    // stage).
    static constexpr size_t SpectraFloats(size_t b)
    {
        return SpectraFloats(MakeGeometry(b, S01Len(b), false));
    }
    static constexpr size_t SpectraFloats(size_t b, size_t max_ir)
    {
        return SpectraFloats(MakeGeometry(b, max_ir, false));
    }
    static constexpr size_t PrepareScratchFloats(size_t b)
    {
        return PrepareScratchFloats(MakeGeometry(b, S01Len(b), false));
    }
    static constexpr size_t PrepareScratchFloats(size_t b, size_t max_ir)
    {
        return PrepareScratchFloats(MakeGeometry(b, max_ir, false));
    }

    // External IR mode: precomputed spectra of one IR per output, per stage.
    // The storage (SpectraFloats) is owned by the caller and must stay valid
    // and unchanged while the engine may use it (see SwapPending()).
    struct IrSpectra
    {
        const float* h[kMaxOutputs][Geometry::kMaxStages] = {};
        size_t       k[kMaxOutputs][Geometry::kMaxStages] = {}; // partitions
        size_t       len = 0; // IR samples covered (longest output)
    };

    // Not RT-safe. Arenas are fully zeroed here (libDaisy's .dtcmram_bss /
    // .sdram_bss sections are NOLOAD and not cleared by the startup code).
    // Fails if the geometry is invalid, an FFT size is not available in the
    // backend (see daisyconvol.mk) or an arena is too small.
    bool Init(const Config& cfg, Arena fast, Arena medium, Arena slow);

    // Precomputes all IR partition spectra and resets the engine state.
    // Not RT-safe (runs FFTs, clears SDRAM): call with audio stopped.
    // One IR per output (irs[c], lens[c]); the single-IR form needs
    // num_outputs == 1. Fails if a length is 0 or > max_ir_len, an IR has a
    // NaN / infinite sample, or in external IR mode.
    bool SetIR(const float* ir, size_t len);
    bool SetIR(const float* const* irs, const size_t* lens);

    // External IR mode: computes the spectra of one IR per output into
    // storage (spectra_floats() floats) and fills *out. IRs longer than the
    // geometry covers are truncated (out->len tells). Touches no engine
    // state (only the FFT plans), so it may run while audio is running;
    // scratch (prepare_scratch_floats()) is destroyed. Not RT-safe. Fails if
    // storage or scratch is too small, or an IR has a NaN / infinite sample.
    bool PrepareIR(const float* const* irs,
                   const size_t*       lens,
                   float*              storage,
                   size_t              storage_floats,
                   float*              scratch,
                   size_t              scratch_floats,
                   IrSpectra*          out) const;
    // Single output; storage and scratch sizes are not checked.
    bool PrepareIR(const float* ir,
                   size_t       len,
                   float*       storage,
                   float*       scratch,
                   IrSpectra*   out) const;

    // Incremental PrepareIR, for preparing IRs while audio runs without
    // blocking the main loop (a 1 s IR takes tens of ms at once, longer
    // than the main-loop stage's deadline slack). BeginPrepareIR checks the
    // arguments (O(1)); each PrepareIRStep then computes one partition
    // spectrum (one FFT of at most 4096 points, ~0.5 ms at 480 MHz) and
    // returns true while work remains. Interleave the steps with
    // DoBackgroundStep() and the rest of the loop. The IRs, storage, scratch
    // and *out must stay valid and untouched until the job is done; *out is
    // usable (SwapIR) only once done() && ok(). A NaN / infinite sample
    // fails the job at the partition that holds it (the storage is then
    // partly written). Same results as PrepareIR, bit for bit.
    class PrepareJob
    {
      public:
        bool done() const { return state_ != kRunning; }
        bool ok() const { return state_ == kOk; }
        // Partition spectra computed so far / in total.
        size_t steps_done() const { return steps_; }
        size_t steps_total() const { return total_; }

      private:
        friend class ConvEngine;
        enum State : uint8_t
        {
            kIdle,
            kRunning,
            kOk,
            kFailed
        };
        const float* irs_[kMaxOutputs]  = {};
        size_t       lens_[kMaxOutputs] = {};
        float*       storage_ = nullptr;
        float*       scratch_ = nullptr;
        IrSpectra*   out_     = nullptr;
        size_t       c_ = 0, s_ = 0, k_ = 0; // next output / stage / part
        float*       h_     = nullptr;       // this stage's spectra
        size_t       steps_ = 0, total_ = 0;
        State        state_ = kIdle;
    };
    // Fails (job not started) on the same argument errors as PrepareIR,
    // except non-finite samples, which PrepareIRStep detects.
    bool BeginPrepareIR(PrepareJob*         job,
                        const float* const* irs,
                        const size_t*       lens,
                        float*              storage,
                        size_t              storage_floats,
                        float*              scratch,
                        size_t              scratch_floats,
                        IrSpectra*          out) const;
    bool PrepareIRStep(PrepareJob* job) const;

    // External IR mode, main loop: switch to s (nullptr = silence). Lock-free
    // and RT-safe. The switch is latched at the next period boundary of the
    // largest stage and completes one period later (without a main-loop
    // stage: within two periods of the largest time-sliced stage, 16 blocks
    // with the default geometries); with a main-loop stage it is latched at
    // a boundary of that stage's period and completes two of its periods
    // later (up to three, 128 ms at 2048 samples). Swapping again before
    // that is fine (last one wins). No-op outside external IR mode.
    void SwapIR(const IrSpectra* s);

    // External IR mode: true while a SwapIR() has not fully taken effect.
    // Once false, every IrSpectra other than ActiveIR() (and its storage) is
    // unused and may be rewritten or freed.
    bool SwapPending() const;
    // External IR mode: the IR in effect (nullptr: none / silence).
    const IrSpectra* ActiveIR() const;

    // Clears the delay lines, keeps the IR (a pending SwapIR takes effect at
    // once). Call it from the audio callback (between ProcessBlock calls),
    // e.g. to drop a stale tail after the input was interrupted, or with
    // audio stopped. Bounded and allocation free: it clears the callback's
    // own state (O(size of the in-callback delay lines)); a main-loop stage
    // is cleared by the main loop, which acknowledges it within one
    // DoBackgroundWork/Step call, and contributes silence until then.
    void Reset();

    // Audio callback path. in: n samples; out: n samples, overwritten.
    // n must be block_size() (otherwise the output is silenced and
    // Perf::block_mismatches counted). in and out may be the same buffer.
    // Bounded worst case, no allocation, no locks.
    void ProcessBlock(const float* in, float* out, size_t n);
    // Same, all outputs: outs[c] for c < num_outputs() (e.g. libDaisy's
    // AudioHandle::OutputBuffer for a stereo engine).
    void ProcessBlock(const float* in, float* const* outs, size_t n);
    // Single output, n == block_size().
    void ProcessBlock(const float* in, float* out);

    // Main-loop path: runs all pending main-loop stage work (FFT -> MAC over
    // the FDL -> IFFT -> publish). A frame must be finished within one
    // period of that stage after it is complete (2048 samples = 42.7 ms at
    // 48 kHz); the frame itself takes ~12 ms of that at 480 MHz for a 2 s
    // IR, so the rest of the main loop must stay well below the remainder.
    // Calling more often is harmless and cheap. No-op without a main-loop
    // stage.
    void DoBackgroundWork();
    // One bounded piece of that work (the window FFT, two partitions of
    // MAC, or one output's IFFT: ~0.3-0.7 ms at 480 MHz), so the main loop
    // can interleave its own work. Returns true if more work may be
    // pending.
    bool DoBackgroundStep();

    // Consistent snapshot of the counters (call from the main loop).
    Perf perf() const;
    // Clears the counters; the callback's own ones are cleared at its next
    // block (and read as 0 until then).
    void ResetPerf();

    const Geometry& geometry() const { return geo_; }
    size_t          block_size() const { return b_; }
    size_t          num_outputs() const { return num_outputs_; }
    size_t          max_ir_len() const { return max_ir_; }
    size_t ir_len() const { return ir_len_.load(std::memory_order_relaxed); }
    // External IR mode: PrepareIR storage and scratch sizes, in floats.
    size_t spectra_floats() const;
    size_t prepare_scratch_floats() const { return MaxFftFloats(geo_); }

  private:
    friend struct ConvEngineTestAccess; // desktop test harness only

    static_assert(detail::IntAlwaysLockFree<uint32_t>(),
                  "the callback / main-loop handoff needs lock-free atomics");
    static_assert(ATOMIC_POINTER_LOCK_FREE == 2,
                  "the callback / main-loop handoff needs lock-free atomics");
    static_assert(detail::IntAlwaysLockFree<size_t>(),
                  "the callback / main-loop handoff needs lock-free atomics");
    static_assert(ATOMIC_BOOL_LOCK_FREE == 2,
                  "the callback / main-loop handoff needs lock-free atomics");

    static constexpr bool ConfigOk(const Config& c)
    {
        return c.num_outputs >= 1 && c.num_outputs <= kMaxOutputs
               && ConfigGeometry(c).Valid()
               && ConfigGeometry(c).block == c.block_size
               && c.max_ir_len > 0
               && ConfigGeometry(c).Length() >= c.max_ir_len;
    }

    // Callback-side FFT scratch: the largest in-callback FFT.
    static constexpr size_t RtScratchFloats(const Geometry& g)
    {
        size_t n = g.Valid() ? 2 * g.block : 0;
        for(size_t s = 1; g.Valid() && s < g.num_stages; s++)
            if(!g.IsBackground(s) && 2 * g.part[s] > n)
                n = 2 * g.part[s];
        return n;
    }
    static constexpr size_t MaxFftFloats(const Geometry& g)
    {
        return g.Valid() ? 2 * g.part[g.num_stages - 1] : 0;
    }

    static constexpr size_t FastFloatsFor(const Geometry& g, size_t outs)
    {
        if(!g.Valid())
            return 0;
        const size_t n0 = 2 * g.block;
        size_t       n  = AlignF(g.count[0] * n0)      // fdl0
                   + 2 * AlignF(n0)                    // in0, acc0
                   + outs * AlignF(g.count[0] * n0);   // h0 per output
        for(size_t s = 1; s < g.num_stages; s++)
        {
            const size_t nf = 2 * g.part[s];
            if(g.IsBackground(s))
                n += AlignF(nf)             // background scratch
                     + outs * AlignF(nf);   // acc per output
            else
                n += AlignF(2 * g.part[s])  // input window
                     + outs * (AlignF(nf)   // acc per output
                               + AlignF(2 * g.part[s])); // staging (2 x P)
        }
        return n + AlignF(RtScratchFloats(g)); // callback-side scratch
    }
    static constexpr size_t
    MediumFloatsFor(const Geometry& g, size_t outs, bool external_ir)
    {
        if(!g.Valid())
            return 0;
        size_t n = 0;
        for(size_t s = 1; s < g.num_stages; s++)
        {
            const size_t nf = 2 * g.part[s];
            if(g.IsBackground(s))
                n += AlignF(4 * g.part[s])            // input ring
                     + outs * AlignF(2 * g.part[s]);  // staging (2 x P)
            else
                n += AlignF(g.count[s] * nf)                           // fdl
                     + (external_ir ? 0 : outs * AlignF(g.count[s] * nf)); // h
        }
        return n;
    }
    static constexpr size_t
    SlowFloatsFor(const Geometry& g, size_t outs, bool external_ir)
    {
        if(!g.Valid() || !g.background)
            return 0;
        const size_t s  = g.num_stages - 1;
        const size_t nh = AlignF(g.count[s] * 2 * g.part[s]);
        return nh + (external_ir ? 0 : outs * nh); // fdl + h per output
    }

    // One time-sliced (in-callback) stage, s >= 1.
    struct Sliced
    {
        size_t   p = 0, n = 0, r = 0; // partition, FFT size, P/B
        size_t   slots = 0;           // allocated partitions
        FftPlan* plan  = nullptr;
        float*   fdl   = nullptr;
        float*   in    = nullptr; // 2P sliding window
        size_t   w     = 0;       // newest FDL slot
        uint32_t wbuf  = 0;       // staging half being written
        // per output
        const float* h[kMaxOutputs]     = {}; // own storage or external
        size_t       ka[kMaxOutputs]    = {}; // active partitions
        float*       acc[kMaxOutputs]   = {};
        float*       stage[kMaxOutputs] = {}; // 2 x P staging, ping-pong
        // external IR switch pending for this stage
        const float* h_next[kMaxOutputs]  = {};
        size_t       ka_next[kMaxOutputs] = {};
        uint32_t     switch_at            = 0;
        bool         switching            = false;
    };

    // Main-loop stage frame, in steps (DoBackgroundStep).
    enum BgStep : uint8_t
    {
        kBgIdle,    // next: pick a frame, snapshot its window, FFT
        kBgMac,     // next: MAC partitions bg_k_...
        kBgInverse, // next: IFFT of output bg_c_, publish after the last
    };
    static constexpr size_t kBgPartsPerStep = 2;

    void Process(const float* in, float* const* outs, size_t n);
    void Silence(float* const* outs, size_t n);
    void ResetState(bool sync);
    void ApplyPendingIR(bool sync);
    void LatchIR();
    void ProcessSliced(Sliced& st, const float* x);
    void ProcessBgInput(const float* x, uint32_t ph2);
    void BgReset(uint32_t req);
    void BgMacPartition(size_t k);
    void BgFinishFrame();

    bool     inited_   = false;
    bool     external_ = false;
    bool     sanitize_ = true;
    Geometry geo_;
    size_t   b_ = 0, n0_ = 0;
    size_t   num_outputs_ = 1;
    size_t   max_ir_      = 0;
    size_t   k0_slots_    = 0;
    size_t   num_sliced_  = 0;
    size_t   rmax_        = 1; // largest time-sliced period, in blocks
    size_t   rl_          = 1; // IR switches latch every rl_ blocks...
    size_t   lead_        = 1; // ...and complete lead_ blocks later
    uint32_t (*get_cycles_)() = nullptr;
    std::atomic<size_t> ir_len_{0};

    // stage 0
    FftPlan* plan0_ = nullptr;
    float*   fdl0_  = nullptr;
    float*   in0_   = nullptr; // [previous block | current block]
    float*   acc0_  = nullptr;
    float*   h0_[kMaxOutputs]  = {};
    size_t   k0a_[kMaxOutputs] = {};
    size_t   w0_     = 0;
    float*   scr_rt_ = nullptr; // callback-side scratch

    Sliced sliced_[Geometry::kMaxStages];

    // main-loop stage (last stage when geo_.background)
    bool     bg_     = false;
    bool     bg_run_ = false; // has work: external mode, or SetIR reached it
    size_t   p2_ = 0, n2_ = 0, r2_ = 0, k2_slots_ = 0;
    FftPlan* plan2_  = nullptr;
    float*   scr_bg_ = nullptr; // n2 floats, main-loop scratch
    float*   ring2_  = nullptr; // 4 x P2 input ring
    float*   fdl2_   = nullptr;
    float*   h2_own_[kMaxOutputs]   = {}; // SetIR spectra
    float*   acc2_[kMaxOutputs]     = {};
    float*   staging2_[kMaxOutputs] = {}; // 2 x P2, ping-pong
    // spectra in use: owned by the main loop while audio runs
    const float* h2_[kMaxOutputs]  = {};
    size_t       k2a_[kMaxOutputs] = {};

    // callback-side state
    uint32_t blk_     = 0;     // callback counter (wrap-safe comparisons)
    uint32_t q2_      = 0;     // main-loop stage frames completed (wraps)
    uint32_t s2_age_  = 0;     // its periods since reset, saturates at 2
    bool     s2_wait_ = false; // Reset() not acknowledged by the main loop
    bool     s2_have_ = false; // this period plays a staged frame...
    uint32_t s2_half_ = 0;     // ...from this staging half

    // external IR switching (see LatchIR), callback side
    std::atomic<const IrSpectra*> ir_pending_{nullptr}; // set by SwapIR
    const IrSpectra* ir_new_       = nullptr; // latched, switching
    const IrSpectra* ir_cur_       = nullptr; // what the stages use
    bool             switching_    = false;
    bool             h0_switching_ = false;
    uint32_t         h0_switch_at_ = 0;
    std::atomic<const IrSpectra*> active_pub_{nullptr}; // ActiveIR()
    std::atomic<bool>             busy_{false};         // SwapPending()

    // background-side state
    uint8_t  bg_step_         = kBgIdle;
    uint32_t bg_j_            = 0; // frame in progress
    size_t   bg_k_            = 0; // next MAC partition
    size_t   bg_c_            = 0; // next output to inverse-transform
    uint32_t bg_frame_cycles_ = 0;
    uint32_t s2_next_         = 0; // next main-loop frame to process
    size_t   w2_              = 0; // its FDL slot
    uint32_t reset_seen_      = 0; // last reset_req_ handled
    uint32_t bg_sw_seen_      = 0; // last bg_sw_req_ applied

    // SPSC handoff (callback -> main loop -> callback)
    std::atomic<uint32_t> frames_produced_{0};
    std::atomic<uint32_t> results_ready_{0};
    // Reset() from the callback: req bumped by the callback, acknowledged
    // by the main loop once its own state is cleared.
    std::atomic<uint32_t>         reset_req_{0};
    std::atomic<uint32_t>         reset_ack_{0};
    std::atomic<const IrSpectra*> bg_reset_ir_{nullptr}; // external: adopt
    // External IR switch of the main-loop stage: the fields are written by
    // the callback before bg_sw_req_ is released; the main loop applies
    // them from frame bg_sw_frame_ on and acknowledges.
    const float*          bg_sw_h_[kMaxOutputs] = {};
    size_t                bg_sw_k_[kMaxOutputs] = {};
    uint32_t              bg_sw_frame_          = 0;
    std::atomic<uint32_t> bg_sw_req_{0};
    std::atomic<uint32_t> bg_sw_ack_{0};

    // Perf: the callback's counters are published with perf_seq_ (perf()
    // retries a torn copy); ResetPerf() asks the callback to clear them.
    Perf                  perf_;
    std::atomic<uint32_t> perf_seq_{0};
    std::atomic<bool>     perf_reset_req_{false};
    std::atomic<uint32_t> cb_cycles_{0}; // ProcessBlock cycles, running sum
};

} // namespace daisyconvol
