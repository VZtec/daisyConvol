// irswitch.cpp - external IR mode example: seamless IR switching while
// audio runs (see conv_engine.h, "External IR mode").
//
// Mono in (IN_L) -> one of three IRs -> both outputs, 48 kHz, block 32. The
// IRs are prepared with PrepareIR() into our own storage; every 2 s (or on
// 'n' over USB serial) the main loop switches to the next one with
// SwapIR(). Once SwapPending() is false the IR we switched away from is
// free, and the main loop regenerates it (a new variant) while audio keeps
// running, in bounded steps (BeginPrepareIR / PrepareIRStep). The
// main-loop stage work is done in bounded DoBackgroundStep() pieces
// interleaved with the rest of the loop.
//
// make IR_LEN=4096 (default): cabinet-like IRs, everything in the callback.
// make IR_LEN=48000: 1 s rooms; the engine adds a main-loop stage, which
// switches too (announced two of its periods ahead: ~85 ms at B = 32).

#include <cmath>

#include "daisy_seed.h"

#include "daisyconvol/daisyconvol.h"
#include "daisyconvol_daisy/seed.h"
#include "usb_log.h" // non-blocking log lines (see there)

using namespace daisy;

namespace
{

#ifndef CONV_BLOCK
#define CONV_BLOCK 32
#endif
#ifndef CONV_IR_LEN
#define CONV_IR_LEN 4096
#endif
constexpr size_t kBlockSize  = CONV_BLOCK;
constexpr size_t kIrLen      = CONV_IR_LEN;
constexpr float  kSampleRate = 48000.0f;
constexpr size_t kNumIrs     = 3;
static_assert(kBlockSize >= 16 && kBlockSize <= 256,
              "BLOCK must be a power of two, 16..256 with libDaisy");

using CE = daisyconvol::ConvEngine;

constexpr CE::Config kConfig = [] {
    CE::Config c;
    c.block_size  = kBlockSize;
    c.max_ir_len  = kIrLen;
    c.external_ir = true;
    // Long IRs: let the last stage run in the main loop (and switch there).
    c.external_background = kIrLen > 4096;
    return c;
}();
static_assert(CE::FastFloats(kConfig) > 0, "invalid engine config");

constexpr size_t NonZero(size_t n)
{
    return n ? n : 1;
}
constexpr size_t kFastFloats    = CE::FastFloats(kConfig);
constexpr size_t kMediumFloats  = CE::MediumFloats(kConfig);
constexpr size_t kSlowFloats    = CE::SlowFloats(kConfig);
constexpr size_t kSpectraFloats = CE::SpectraFloats(kConfig);
constexpr size_t kScratchFloats = CE::PrepareScratchFloats(kConfig);


DAISYCONVOL_DTCM alignas(32) float  fast_mem[NonZero(kFastFloats)];
alignas(32) float                   medium_mem[NonZero(kMediumFloats)];
alignas(32) DAISYCONVOL_SDRAM float slow_mem[NonZero(kSlowFloats)];

// IR spectra: read by the in-callback stages every block, so keep short
// ones in AXI SRAM; long ones (mostly main-loop stage spectra) in SDRAM.
#if CONV_IR_LEN > 4096
#define SPECTRA_SECTION DAISYCONVOL_SDRAM
#else
#define SPECTRA_SECTION
#endif
SPECTRA_SECTION alignas(32) float spectra[kNumIrs][kSpectraFloats];
alignas(32) float                 scratch[kScratchFloats];
DAISYCONVOL_SDRAM float           ir_buf[kIrLen];
CE::IrSpectra                     irs[kNumIrs];
uint32_t                          variant[kNumIrs];

alignas(32) uint8_t fft_tables[CE::FftTableBytes(kConfig)];

DaisySeed hw;
CE        engine;

// IR n, variant v: decaying noise through a one-pole low-pass whose cutoff
// depends on n (dark / mid / bright), energy-normalized. Short ones sound
// like different cabinets, long ones like different rooms.
//
// Regenerating an IR while audio runs must not block the main loop: for a
// 1 s IR, generating it and computing its spectra takes ~43 ms at once,
// a whole period of the main-loop stage (its frames would miss their
// deadline). So it is a small job done in bounded steps: generate 2048
// samples, normalize 2048 samples, or one PrepareIRStep (one partition
// FFT) per call.
struct Regen
{
    enum Phase
    {
        kIdle,
        kGenerate,
        kNormalize,
        kPrepare
    };
    Phase          phase = kIdle;
    size_t         n = 0, i = 0;
    uint32_t       seed   = 0;
    float          a = 0.0f, decay = 0.0f, lp = 0.0f, norm = 1.0f;
    double         energy = 0.0;
    CE::PrepareJob job;
    bool           ok        = true;
    uint32_t       start_us  = 0;
};
Regen regen;

constexpr size_t kGenChunk = 2048;

void StartRegen(size_t n)
{
    const float coef[] = {0.15f, 0.45f, 0.85f}; // low-pass amount
    regen.phase        = Regen::kGenerate;
    regen.n            = n;
    regen.i            = 0;
    regen.seed         = 0x9E3779B9u * (uint32_t)(n + 1) + 7919u * variant[n];
    regen.a            = coef[n % 3];
    regen.decay        = kIrLen > 4096 ? 3.0f : 60.0f; // 1/s
    regen.lp           = 0.0f;
    regen.energy       = 0.0;
    regen.ok           = true;
    regen.start_us     = System::GetUs();
}

// One bounded piece of the regeneration; returns true while work remains.
bool RegenStep()
{
    switch(regen.phase)
    {
        case Regen::kIdle: return false;
        case Regen::kGenerate:
        {
            const size_t end = regen.i + kGenChunk < kIrLen ? regen.i + kGenChunk
                                                            : kIrLen;
            for(size_t i = regen.i; i < end; i++)
            {
                regen.seed = regen.seed * 1664525u + 1013904223u;
                const float noise = (float)(int32_t)regen.seed / 2147483648.0f;
                regen.lp += regen.a * (noise - regen.lp);
                ir_buf[i] = regen.lp * expf(-regen.decay * (float)i / kSampleRate);
                regen.energy += (double)ir_buf[i] * ir_buf[i];
            }
            regen.i = end;
            if(end == kIrLen)
            {
                regen.norm  = 1.0f / sqrtf((float)regen.energy + 1e-12f);
                regen.i     = 0;
                regen.phase = Regen::kNormalize;
            }
            return true;
        }
        case Regen::kNormalize:
        {
            const size_t end = regen.i + kGenChunk < kIrLen ? regen.i + kGenChunk
                                                            : kIrLen;
            for(size_t i = regen.i; i < end; i++)
                ir_buf[i] *= regen.norm;
            regen.i = end;
            if(end == kIrLen)
            {
                const float* p[1]    = {ir_buf};
                const size_t lens[1] = {kIrLen};
                regen.ok             = engine.BeginPrepareIR(&regen.job,
                                                 p,
                                                 lens,
                                                 spectra[regen.n],
                                                 kSpectraFloats,
                                                 scratch,
                                                 kScratchFloats,
                                                 &irs[regen.n]);
                regen.phase = regen.ok ? Regen::kPrepare : Regen::kIdle;
                return regen.ok;
            }
            return true;
        }
        case Regen::kPrepare:
            if(engine.PrepareIRStep(&regen.job))
                return true;
            regen.ok    = regen.job.ok();
            regen.phase = Regen::kIdle;
            return false;
    }
    return false;
}

// At boot (audio not running yet) the same job, run to completion.
bool Prepare(size_t n)
{
    StartRegen(n);
    while(RegenStep()) {}
    return regen.ok;
}

size_t IndexOf(const CE::IrSpectra* s)
{
    for(size_t n = 0; n < kNumIrs; n++)
        if(s == &irs[n])
            return n;
    return kNumIrs;
}

volatile bool dfu_requested  = false;
volatile bool next_requested = false;

void OnUsbReceive(uint8_t* buf, uint32_t* len)
{
    for(uint32_t i = 0; i < *len; i++)
    {
        if(buf[i] == 'D')
            dfu_requested = true;
        if(buf[i] == 'n')
            next_requested = true;
    }
}

void AudioCallback(AudioHandle::InputBuffer  in,
                   AudioHandle::OutputBuffer out,
                   size_t                    size)
{
    engine.ProcessBlock(in[0], out[0], size);
    for(size_t i = 0; i < size; i++)
        out[1][i] = out[0][i];
}

} // namespace

int main(void)
{
    hw.Init(true);
    hw.SetAudioBlockSize(kBlockSize);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    daisyconvol_daisy::EnableCycleCounter();
    daisyconvol_daisy::EnableFlushToZero();

    CE::Config cfg = kConfig;
    cfg.get_cycles = daisyconvol_daisy::CycleCount;
    bool ok        = hw.AudioBlockSize() == kBlockSize
              && engine.Init(cfg,
                             {fast_mem, kFastFloats, 0},
                             {medium_mem, kMediumFloats, 0},
                             {slow_mem, kSlowFloats, 0});
    for(size_t n = 0; ok && n < kNumIrs; n++)
        ok = Prepare(n);
    ok = ok && daisyconvol::FftRelocateTables(fft_tables, sizeof(fft_tables));

    hw.StartLog(false);
    hw.usb_handle.SetReceiveCallback(OnUsbReceive, UsbHandle::FS_INTERNAL);
    while(!ok)
    {
        usblog::Line(hw, "irswitch: init failed");
        hw.DelayMs(1000);
    }
    {
        const daisyconvol::Geometry& g = engine.geometry();
        usblog::Line(hw, "irswitch: block %u, %u IRs of %u samples, %u stages%s",
                     (unsigned)kBlockSize,
                     (unsigned)kNumIrs,
                     (unsigned)kIrLen,
                     (unsigned)g.num_stages,
                     g.background ? " (last in the main loop)" : "");
    }

    // The first IR takes effect a few blocks after audio starts.
    engine.SwapIR(&irs[0]);
    hw.StartAudio(AudioCallback);

    size_t   current     = 0;      // IR asked for
    size_t   stale       = kNumIrs; // IR switched away from, to regenerate
    bool     waiting     = true;   // a swap is in progress
    uint32_t swap_time   = System::GetUs();
    uint32_t last_switch = System::GetNow();
    uint32_t last_print  = System::GetNow();
    uint32_t last_led    = System::GetNow();
    bool     led         = false;
    while(1)
    {
        // Main-loop stage work in bounded pieces (none without a main-loop
        // stage), then the rest of the loop.
        for(int i = 0; i < 4 && engine.DoBackgroundStep(); i++) {}

        if(dfu_requested)
            System::ResetToBootloader(
                System::BootloaderMode::DAISY_INFINITE_TIMEOUT);

        const uint32_t now = System::GetNow();
        if(waiting && !engine.SwapPending())
        {
            waiting = false;
            usblog::Line(hw, "  now playing IR %u (switch took %lu us)",
                         (unsigned)IndexOf(engine.ActiveIR()),
                         System::GetUs() - swap_time);
            // The IR we left is unused now: make a new variant of it, in
            // bounded steps (see Regen).
            if(stale < kNumIrs)
            {
                variant[stale]++;
                StartRegen(stale);
                stale = kNumIrs;
            }
        }
        if(regen.phase != Regen::kIdle && !RegenStep())
            usblog::Line(hw, "  regenerated IR %u in %lu us%s",
                         (unsigned)regen.n,
                         System::GetUs() - regen.start_us,
                         regen.ok ? "" : " (FAILED)");
        // Never switch to an IR that is being regenerated.
        if(!waiting && regen.phase == Regen::kIdle
           && (next_requested || now - last_switch >= 2000))
        {
            next_requested = false;
            last_switch    = now;
            stale          = current;
            current        = (current + 1) % kNumIrs;
            swap_time      = System::GetUs();
            waiting        = true;
            engine.SwapIR(&irs[current]);
        }
        if(now - last_led >= 250)
        {
            last_led = now;
            led      = !led;
            hw.SetLed(led);
        }
        if(now - last_print >= 1000)
        {
            last_print    = now;
            const auto pf = engine.perf();
            usblog::Line(hw, "cb avg %lu peak %lu cyc | bg frame avg %lu | undr "
                         "%lu",
                         pf.callback.Avg(),
                         pf.callback.peak,
                         pf.stage2_frame.Avg(),
                         pf.s2_underruns);
        }
    }
}
