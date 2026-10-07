// convolver.cpp - libDaisy integration example for the partitioned
// convolution engine (see conv_engine.h).
//
// Mono in (IN_L) -> IR (ir_data.h, or a synthetic reverb) -> both outputs,
// 48 kHz, block size 32 (make BLOCK=...). With make STEREO=1 the engine has
// two outputs, each with its own IR (a stereo reverb from a mono input, for
// much less than two engines). The audio callback runs the in-callback
// stages; the main loop runs the last stage via DoBackgroundWork() and
// prints the per-stage cycle load (DWT cycle counter) once per second over
// USB serial.

#include <cmath>

#include "daisy_seed.h"

#include "daisyconvol/daisyconvol.h"
#include "daisyconvol_daisy/seed.h"
#include "usb_log.h" // non-blocking log lines (see there)

// Real IR embedded from a WAV: generate ir_data.h with tools/wav2ir.py and
// the Makefile defines CONV_USE_EMBEDDED_IR.
#ifdef CONV_USE_EMBEDDED_IR
#include "ir_data.h"
#endif

using namespace daisy;

namespace
{

// Audio block size: make BLOCK=<power of two, 16..256> (default 32).
#ifndef CONV_BLOCK
#define CONV_BLOCK 32
#endif
constexpr size_t kBlockSize  = CONV_BLOCK;
constexpr float  kSampleRate = 48000.0f;
// libDaisy's AudioHandle caps the block size at 256 (kAudioMaxBufferSize /
// 4) and DaisySeed::SetAudioBlockSize() drops the error: a larger engine
// block would read and write past the callback's buffers.
static_assert(kBlockSize >= 16 && kBlockSize <= 256
                  && (kBlockSize & (kBlockSize - 1)) == 0,
              "BLOCK must be a power of two, 16..256 with libDaisy");
// Maximum IR length in samples: make IR_LEN=<n> (default 3 s). The
// synthetic reverb used without ir_data.h is that long.
#ifndef CONV_IR_LEN
#define CONV_IR_LEN (3 * 48000)
#endif
constexpr size_t kMaxIrLen = CONV_IR_LEN;
#ifndef CONV_STEREO
#define CONV_STEREO 0
#endif
constexpr size_t kOutputs = CONV_STEREO ? 2 : 1;

using CE = daisyconvol::ConvEngine;

constexpr CE::Config kConfig = [] {
    CE::Config c;
    c.block_size  = kBlockSize;
    c.max_ir_len  = kMaxIrLen;
    c.num_outputs = kOutputs;
    return c;
}();
static_assert(CE::FastFloats(kConfig) > 0, "invalid engine config");

// Arena sizes, never 0 (zero-length arrays are not C++).
constexpr size_t NonZero(size_t n)
{
    return n ? n : 1;
}
constexpr size_t kFastFloats   = CE::FastFloats(kConfig);
constexpr size_t kMediumFloats = CE::MediumFloats(kConfig);
constexpr size_t kSlowFloats   = CE::SlowFloats(kConfig);

// The three memory arenas. Default .bss lands in AXI SRAM (512 KB) on
// libDaisy's linker script; DTCM and SDRAM are explicit NOLOAD sections
// (the engine zeroes the arenas in Init()).
// The fast arena grows with the block size (stage-0 spectra and the
// time-sliced stages' windows): past 64 frames it no longer fits the 128 KB
// DTCM next to the stack (with two outputs, past 32), so it goes to AXI
// SRAM.
#if CONV_BLOCK <= (CONV_STEREO ? 32 : 64)
DAISYCONVOL_DTCM
#endif
alignas(32) float                   fast_mem[NonZero(kFastFloats)];
alignas(32) float                   medium_mem[NonZero(kMediumFloats)];
alignas(32) DAISYCONVOL_SDRAM float slow_mem[NonZero(kSlowFloats)];

// The FFT twiddle / bit-reversal tables live in flash, which here is QSPI:
// they are copied to AXI SRAM at startup (every FFT then reads them from
// RAM; 46 KB at B = 32 with a reverb). make TABLES_IN_RAM=0 skips it.
#ifndef CONV_TABLES_IN_RAM
#define CONV_TABLES_IN_RAM 1
#endif
#if CONV_TABLES_IN_RAM
alignas(32) uint8_t fft_tables[CE::FftTableBytes(kConfig)];
#endif

#ifndef CONV_USE_EMBEDDED_IR
DAISYCONVOL_SDRAM float ir_buf[kOutputs][kMaxIrLen];
#endif

DaisySeed hw;
CE        engine;

#ifndef CONV_USE_EMBEDDED_IR
// Synthetic IR: dry impulse + exponentially decaying noise tail,
// energy-normalized. Stands in for an IR loaded from QSPI/SD. Each output
// gets its own noise (seed), which makes the stereo version wide.
void GenerateIr(float* ir, size_t len, uint32_t seed)
{
    double energy = 0.0;
    for(size_t i = 0; i < len; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        const float noise = (float)(int32_t)seed / 2147483648.0f;
        const float t     = (float)i / kSampleRate;
        ir[i]             = noise * expf(-4.5f * t) * 0.05f;
        energy += (double)ir[i] * ir[i];
    }
    const float norm = 1.0f / sqrtf((float)energy + 1e-12f);
    for(size_t i = 0; i < len; i++)
        ir[i] *= norm;
    ir[0] += 0.5f; // some dry signal
}
#endif // !CONV_USE_EMBEDDED_IR

// A 'D' received over USB serial resets into the Daisy bootloader (DFU),
// so the board can be reflashed without touching BOOT/RESET.
volatile bool dfu_requested = false;

void OnUsbReceive(uint8_t* buf, uint32_t* len)
{
    for(uint32_t i = 0; i < *len; i++)
        if(buf[i] == 'D')
            dfu_requested = true;
}

void AudioCallback(AudioHandle::InputBuffer  in,
                   AudioHandle::OutputBuffer out,
                   size_t                    size)
{
#if CONV_STEREO
    engine.ProcessBlock(in[0], out, size); // out[0], out[1]
#else
    engine.ProcessBlock(in[0], out[0], size);
    for(size_t i = 0; i < size; i++)
        out[1][i] = out[0][i];
#endif
}

} // namespace

int main(void)
{
    hw.Init(true); // 480 MHz: the cycle budgets below assume it
    hw.SetAudioBlockSize(kBlockSize);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    daisyconvol_daisy::EnableCycleCounter();
    daisyconvol_daisy::EnableFlushToZero();

    CE::Config cfg = kConfig;
    cfg.get_cycles = daisyconvol_daisy::CycleCount;

    daisyconvol::Arena fast{fast_mem, kFastFloats, 0};
    daisyconvol::Arena medium{medium_mem, kMediumFloats, 0};
    daisyconvol::Arena slow{slow_mem, kSlowFloats, 0};

    bool ok = hw.AudioBlockSize() == kBlockSize
              && engine.Init(cfg, fast, medium, slow);
    if(ok)
    {
#ifdef CONV_USE_EMBEDDED_IR
        // ir_data.h: kIrData lives in QSPI (.rodata). SetIR reads it once
        // here and builds the partition spectra; the audio path never
        // touches QSPI afterwards. Length is clamped to the engine maximum.
        const size_t ir_len = kIrDataLen < kMaxIrLen ? kIrDataLen : kMaxIrLen;
#if CONV_STEREO && defined(IR_DATA_STEREO)
        const float* irs[2] = {kIrDataL, kIrDataR};
#elif CONV_STEREO
        const float* irs[2] = {kIrData, kIrData}; // mono file: same IR
#else
        const float* irs[1] = {kIrData};
#endif
        size_t lens[kOutputs];
        for(size_t c = 0; c < kOutputs; c++)
            lens[c] = ir_len;
        ok = engine.SetIR(irs, lens);
#else
        const float* irs[kOutputs];
        size_t       lens[kOutputs];
        for(size_t c = 0; c < kOutputs; c++)
        {
            GenerateIr(ir_buf[c], kMaxIrLen, 0x12345678u + 7919u * c);
            irs[c]  = ir_buf[c];
            lens[c] = kMaxIrLen;
        }
        ok = engine.SetIR(irs, lens); // before StartAudio: not RT-safe
#endif
    }
#if CONV_TABLES_IN_RAM
    const size_t table_bytes = daisyconvol::FftTableBytes();
    const bool   relocated
        = ok && daisyconvol::FftRelocateTables(fft_tables, sizeof(fft_tables));
#endif

    hw.StartLog(false);
    hw.usb_handle.SetReceiveCallback(OnUsbReceive,
                                     UsbHandle::FS_INTERNAL);
    if(!ok)
    {
        while(1)
        {
            usblog::Line(hw, "conv engine init failed (audio block %u, want %u)",
                         (unsigned)hw.AudioBlockSize(),
                         (unsigned)kBlockSize);
            hw.DelayMs(1000);
        }
    }

    {
        const daisyconvol::Geometry& g = engine.geometry();
        usblog::Line(hw, "daisyConvol: block %u, IR %u samples, %u output(s), %u "
                     "stages",
                     (unsigned)kBlockSize,
                     (unsigned)engine.ir_len(),
                     (unsigned)kOutputs,
                     (unsigned)g.num_stages);
        for(size_t s = 0; s < g.num_stages; s++)
            usblog::Line(hw, "  stage %u: %u x %u%s",
                         (unsigned)s,
                         (unsigned)g.count[s],
                         (unsigned)g.part[s],
                         g.IsBackground(s) ? " (main loop)" : "");
#if CONV_TABLES_IN_RAM
        usblog::Line(hw, "  FFT tables: %u bytes %s",
                     (unsigned)table_bytes,
                     relocated ? "copied to RAM" : "left in flash (buffer "
                                                   "too small)");
#endif
    }
    hw.StartAudio(AudioCallback);

    // Cycle budget per callback at 480 MHz / 48 kHz.
    constexpr uint32_t kBudget = (uint32_t)(kBlockSize * (480e6f / 48e3f));

    uint32_t last_print = System::GetNow();
    while(1)
    {
        // All main-loop stage work happens here. Deadline window: one period
        // of that stage (2048 samples = 42.7 ms); a frame costs ~12 ms
        // for a 2 s IR, so anything that blocks this loop for tens of ms
        // shows up as underruns. DoBackgroundStep() does it in ~0.5 ms
        // pieces when the loop has other work to interleave.
        engine.DoBackgroundWork();

        if(dfu_requested)
            System::ResetToBootloader(
                System::BootloaderMode::DAISY_INFINITE_TIMEOUT);

        const uint32_t now = System::GetNow();
        if(now - last_print >= 1000)
        {
            last_print    = now;
            const auto pf = engine.perf();
            usblog::Line(hw, "cb avg %lu peak %lu cyc (%lu%%) | s0 %lu | sliced "
                         "%lu | bg frame avg %lu peak %lu | undr %lu late %lu "
                         "drop %lu | nan %lu",
                         pf.callback.Avg(),
                         pf.callback.peak,
                         pf.callback.peak * 100u / kBudget,
                         pf.stage0.peak,
                         pf.stage1_slice.peak,
                         pf.stage2_frame.Avg(),
                         pf.stage2_frame.peak,
                         pf.s2_underruns,
                         pf.s2_late_skips,
                         pf.s2_dropped_inputs,
                         pf.nonfinite_inputs);
        }
    }
}
