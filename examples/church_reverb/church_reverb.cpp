// church_reverb.cpp - convolution church reverb on a bare Daisy Seed.
//
// Mono in (IN_L) -> dry + a 5 s church IR -> both outputs. The first 4096
// samples of the IR run in the audio callback; the rest (2048-sample
// partitions, in SDRAM) in the main loop, which does that work in bounded
// DoBackgroundStep() pieces. make MONO=0 gives a stereo IR (left / right
// outputs, sharing the input delay lines and their FFTs): ~1.6x the
// main-loop work, so ~3 s fits.
//
// IR: reverb_church_ir.h next to this file (Iglesia San Pedro, Chivilcoy,
// Argentina, far position; VZtec, CC BY 4.0), generated with
// tools/wav2ir.py (see README.md).
// make BLOCK=<16..256> IR_LEN=<samples> MONO=<0|1> (default 32, 240000 = 5 s,
// mono).
//
// Controls (USB serial, one key each):
//   [ ]  wet less / more               - +  output level
//   d    dry on / off                  b  bypass
//   D    reset into the bootloader     h  help
// Once per second it prints the callback and main-loop load.

#include <cmath>

#include "daisy_seed.h"

#include "daisyconvol/daisyconvol.h"
#include "daisyconvol_daisy/seed.h"
#include "usb_log.h" // non-blocking log lines (see there)

// The church IR (tools/wav2ir.py --name ReverbChurchIr).
#include "reverb_church_ir.h"

using namespace daisy;

namespace
{

#ifndef REVERB_BLOCK
#define REVERB_BLOCK 32
#endif
#ifndef REVERB_IR_LEN
#define REVERB_IR_LEN 240000 // 5 s
#endif
constexpr size_t kBlockSize  = REVERB_BLOCK;
constexpr size_t kMaxIrLen   = REVERB_IR_LEN;
constexpr float  kSampleRate = 48000.0f;
// MONO (default): one engine output, the same wet signal on both sides.
// make MONO=0: two outputs (left / right IRs), ~1.6x the main-loop work.
#ifndef REVERB_MONO
#define REVERB_MONO 1
#endif
constexpr size_t kChannels   = REVERB_MONO ? 1 : 2; // engine outputs
static_assert(kBlockSize >= 16 && kBlockSize <= 256
                  && (kBlockSize & (kBlockSize - 1)) == 0,
              "BLOCK must be a power of two, 16..256 with libDaisy");

using CE = daisyconvol::ConvEngine;

constexpr CE::Config kConfig = [] {
    CE::Config c;
    c.block_size  = kBlockSize;
    c.max_ir_len  = kMaxIrLen;
    c.num_outputs = kChannels;
    return c;
}();
static_assert(CE::FastFloats(kConfig) > 0, "invalid engine config");

constexpr size_t NonZero(size_t n)
{
    return n ? n : 1;
}
constexpr size_t kFastFloats   = CE::FastFloats(kConfig);
constexpr size_t kMediumFloats = CE::MediumFloats(kConfig);
constexpr size_t kSlowFloats   = CE::SlowFloats(kConfig);

// Fast arena in DTCM while it fits next to the stack; FFT tables (the
// 4096-point ones are ~40 KB) and the medium arena in AXI SRAM; the
// main-loop stage's spectra and delay line in SDRAM.
#if REVERB_BLOCK <= 32
DAISYCONVOL_DTCM
#endif
alignas(32) float                   fast_mem[NonZero(kFastFloats)];
alignas(32) float                   medium_mem[NonZero(kMediumFloats)];
alignas(32) DAISYCONVOL_SDRAM float slow_mem[NonZero(kSlowFloats)];
alignas(32) uint8_t                 fft_tables[CE::FftTableBytes(kConfig)];


DaisySeed hw;
CE        engine;

// Controls, written by the main loop, read by the callback.
volatile float wet      = 0.35f;
volatile float level    = 1.0f;
volatile bool  dry_on   = true;
volatile bool  bypassed = false;

volatile uint32_t cb_cycles_peak = 0;
volatile uint64_t cb_cycles_sum  = 0;
volatile uint32_t cb_count       = 0;

float wet_buf[2][kBlockSize]; // [1] unused with MONO

void AudioCallback(AudioHandle::InputBuffer  in,
                   AudioHandle::OutputBuffer out,
                   size_t                    size)
{
    const uint32_t t0 = DWT->CYCCNT;

    float* wets[2] = {wet_buf[0], wet_buf[1]};
    engine.ProcessBlock(in[0], wets, size); // silences on a size mismatch

    const float g  = level;
    const float gw = g * wet;
    const float gd = dry_on ? g * (1.0f - wet) : 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        const float x = in[0][i];
        if(bypassed)
        {
            out[0][i] = out[1][i] = x;
            continue;
        }
        out[0][i] = gd * x + gw * wet_buf[0][i];
        out[1][i] = gd * x + gw * wet_buf[kChannels - 1][i];
    }

    const uint32_t dt = DWT->CYCCNT - t0;
    if(dt > cb_cycles_peak)
        cb_cycles_peak = dt;
    cb_cycles_sum = cb_cycles_sum + dt;
    cb_count      = cb_count + 1;
}


volatile char pending_key = 0;

void OnUsbReceive(uint8_t* buf, uint32_t* len)
{
    if(*len)
        pending_key = (char)buf[*len - 1];
}

void PrintHelp()
{
    usblog::Line(hw, "keys: [ ] wet | - + level | d dry | b bypass | h help");
}

void PrintState()
{
    usblog::Line(hw, "wet %d%% | dry %s | level %d%%%s",
                 (int)(wet * 100.0f + 0.5f),
                 dry_on ? "on" : "off",
                 (int)(level * 100.0f + 0.5f),
                 bypassed ? " | BYPASS" : "");
}

} // namespace

int main(void)
{
    hw.Init(true); // 480 MHz
    hw.SetAudioBlockSize(kBlockSize);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    daisyconvol_daisy::EnableCycleCounter();
    daisyconvol_daisy::EnableFlushToZero(); // also in the audio interrupt

    CE::Config cfg = kConfig;
    cfg.get_cycles = daisyconvol_daisy::CycleCount;
    bool ok        = hw.AudioBlockSize() == kBlockSize
              && engine.Init(cfg,
                             {fast_mem, kFastFloats, 0},
                             {medium_mem, kMediumFloats, 0},
                             {slow_mem, kSlowFloats, 0});
    const size_t n = kReverbChurchIrLen < kMaxIrLen ? kReverbChurchIrLen : kMaxIrLen;
#ifdef REVERB_CHURCH_IR_STEREO
    const float* irs[2] = {kReverbChurchIrL, kReverbChurchIrR};
#else
    const float* irs[2] = {kReverbChurchIr, kReverbChurchIr};
#endif
    const size_t lens[2] = {n, n};
    ok = ok && engine.SetIR(irs, lens) // not RT-safe: before StartAudio
         && daisyconvol::FftRelocateTables(fft_tables, sizeof(fft_tables));

#ifdef REVERB_SELFTEST
    // make SELFTEST=1: an impulse through the engine before audio starts,
    // compared with the IR itself, head (in-callback stages) and tail
    // (main-loop stage) separately.
    double st_e[2] = {0, 0}, st_r[2] = {0, 0}, st_tail_out = 0, st_tail_ir = 0;
    if(ok)
    {
        float zero[kBlockSize] = {}, imp[kBlockSize] = {};
        imp[0] = 1.0f;
        const size_t nb = (n + kBlockSize - 1) / kBlockSize + 8;
        for(size_t k = 0; k < nb; k++)
        {
            float* o[2] = {wet_buf[0], wet_buf[1]};
            engine.ProcessBlock(k ? zero : imp, o, kBlockSize);
            engine.DoBackgroundWork();
            for(size_t i = 0; i < kBlockSize; i++)
            {
                const size_t t = k * kBlockSize + i;
                for(size_t c = 0; c < 2; c++)
                {
                    const size_t ch = c < kChannels ? c : 0; // MONO: R = L
                    const double r  = t < n ? irs[ch][t] : 0.0;
                    const double d  = wet_buf[ch][i] - r;
                    st_e[c] += d * d;
                    st_r[c] += r * r;
                }
                if(t >= 4096)
                {
                    st_tail_out += (double)wet_buf[0][i] * wet_buf[0][i];
                    st_tail_ir += t < n ? (double)irs[0][t] * irs[0][t] : 0.0;
                }
            }
        }
        engine.Reset();
        engine.ResetPerf();
    }
#endif

    hw.StartLog(false);
    hw.usb_handle.SetReceiveCallback(OnUsbReceive, UsbHandle::FS_INTERNAL);
    if(!ok)
        for(;;)
        {
            usblog::Line(hw, "church_reverb: engine init failed");
            System::Delay(1000);
        }

    hw.StartAudio(AudioCallback);

    const daisyconvol::Geometry& geo = engine.geometry();
    // Main-loop stage partition: one frame per P samples, 48000 / P frames
    // per second (23.4 at P = 2048); 0 without that stage.
    const uint32_t bg_part
        = geo.background ? (uint32_t)geo.part[geo.num_stages - 1] : 0;
    constexpr uint32_t kBudget = kBlockSize * 10000; // cycles per block

    uint32_t last = System::GetNow();
    bool     led  = false;
    bool     told = false;
    for(;;)
    {
        // Main-loop stage work in ~0.5 ms pieces; the rest of the loop (keys,
        // log, LED) runs between them. The frame must still finish within
        // one period of that stage (2048 samples = 42.7 ms).
        engine.DoBackgroundStep();

        const uint32_t now = System::GetNow();
        if(!told && now > 1000) // once the USB serial port is up
        {
            told = true;
            usblog::Line(hw, "church_reverb: IR %u samples, %s, block %u",
                         (unsigned)engine.ir_len(),
                         kChannels == 1 ? "mono" : "stereo",
                         (unsigned)kBlockSize);
            for(size_t s = 0; s < geo.num_stages; s++)
                usblog::Line(hw, "  stage %u: %u x %u%s",
                             (unsigned)s,
                             (unsigned)geo.count[s],
                             (unsigned)geo.part[s],
                             geo.IsBackground(s) ? " (main loop)" : "");
#ifdef REVERB_SELFTEST
            usblog::Line(hw, "selftest: error L %d dB R %d dB | tail (after 4096) "
                         "out/IR energy %d / %d dB",
                         (int)(10.0 * log10(st_e[0] / st_r[0] + 1e-30)),
                         (int)(10.0 * log10(st_e[1] / st_r[1] + 1e-30)),
                         (int)(10.0 * log10(st_tail_out + 1e-30)),
                         (int)(10.0 * log10(st_tail_ir + 1e-30)));
#endif
            PrintHelp();
            PrintState();
        }

        const char k = pending_key;
        if(k)
        {
            pending_key = 0;
            switch(k)
            {
                case '[': wet = fmaxf(0.0f, wet - 0.05f); break;
                case ']': wet = fminf(1.0f, wet + 0.05f); break;
                case '-': level = fmaxf(0.0f, level - 0.1f); break;
                case '+':
                case '=': level = fminf(2.0f, level + 0.1f); break;
                case 'd': dry_on = !dry_on; break;
                case 'b': bypassed = !bypassed; break;
                case 'D':
                    System::ResetToBootloader(
                        System::BootloaderMode::DAISY_INFINITE_TIMEOUT);
                    break;
                default: PrintHelp(); break;
            }
            PrintState();
        }

        if(now - last >= 1000)
        {
            last = now;
            __disable_irq();
            const uint32_t nb   = cb_count;
            const uint64_t sum  = cb_cycles_sum;
            const uint32_t peak = cb_cycles_peak;
            cb_count = 0;
            cb_cycles_sum  = 0;
            cb_cycles_peak = 0;
            __enable_irq();
            const uint32_t avg = nb ? (uint32_t)(sum / nb) : 0;
            const auto     pf  = engine.perf();
            // CPU share, in tenths of a percent of 480 MHz.
            const uint32_t cb_pm = avg * 1000 / kBudget;
            const uint32_t bg_pm
                = bg_part ? (uint32_t)((uint64_t)pf.stage2_frame.Avg() * 48000
                                       / bg_part / 480000)
                          : 0;
            usblog::Line(hw, "callback avg %lu peak %lu (%lu%%) | main-loop frame "
                         "avg %lu peak %lu | CPU %lu.%lu%% + %lu.%lu%% | undr "
                         "%lu late %lu drop %lu",
                         avg,
                         peak,
                         peak * 100 / kBudget,
                         pf.stage2_frame.Avg(),
                         pf.stage2_frame.peak,
                         cb_pm / 10,
                         cb_pm % 10,
                         bg_pm / 10,
                         bg_pm % 10,
                         pf.s2_underruns,
                         pf.s2_late_skips,
                         pf.s2_dropped_inputs);
#ifdef REVERB_SELFTEST
            usblog::Line(hw, "selftest: error L %d dB R %d dB | tail (after 4096) "
                         "out/IR energy %d / %d dB",
                         (int)(10.0 * log10(st_e[0] / st_r[0] + 1e-30)),
                         (int)(10.0 * log10(st_e[1] / st_r[1] + 1e-30)),
                         (int)(10.0 * log10(st_tail_out + 1e-30)),
                         (int)(10.0 * log10(st_tail_ir + 1e-30)));
#endif
            hw.SetLed(led = !led);
        }
    }
}
