// cabsim.cpp - guitar cabinet simulator on a bare Daisy Seed.
//
// Mono in (IN_L) -> two cabinet IRs (two mics on the same cab) -> blended or
// spread across the outputs. One engine with two outputs: both mics share
// the input delay lines and their FFTs, so the second mic costs only its
// MACs and IFFTs. 4096-sample IRs (85 ms) at block 32 run entirely in the
// audio callback: no main-loop stage, no SDRAM.
//
// IRs: cab_mic_a.h / cab_mic_b.h next to this file (a Marshall 4x12
// Greenback, SM57 and R1), generated with tools/wav2ir.py (see README.md).
//
// Controls (USB serial, one key each):
//   [ ]  mic blend towards A / B       w  mono blend <-> wide (L = A, R = B)
//   - +  output level                  b  bypass
//   D    reset into the bootloader     h  help
// Once per second it prints the audio callback load.

#include <cmath>

#include "daisy_seed.h"

#include "daisyconvol/daisyconvol.h"
#include "daisyconvol_daisy/seed.h"
#include "usb_log.h" // non-blocking log lines (see there)

// The two mics' IRs (tools/wav2ir.py --name CabMicA / CabMicB).
#include "cab_mic_a.h"
#include "cab_mic_b.h"

using namespace daisy;

#ifdef CABSIM_ITCM
// make ITCM=1: the engine's and the FFT's code is linked to the ITCM but
// stored in QSPI (see the Makefile). It is copied from .preinit_array, so
// before any C++ static initializer: the FFT backend's lives there too.
extern "C" uint32_t _sitcmram_text, _itcmram_text_start, _itcmram_text_end;

static void CopyCodeToItcm()
{
    const volatile uint32_t* src = &_sitcmram_text;
    volatile uint32_t*       dst = &_itcmram_text_start;
    while(dst < &_itcmram_text_end)
        *dst++ = *src++;
    __DSB();
    __ISB();
}
__attribute__((section(".preinit_array"), used)) static void (*const
    copy_code_to_itcm)() = CopyCodeToItcm;
#endif

namespace
{

constexpr size_t kBlockSize  = 32;
constexpr size_t kMaxIrLen   = 4096; // 85 ms at 48 kHz
constexpr float  kSampleRate = 48000.0f;
constexpr size_t kMics       = 2;

using CE = daisyconvol::ConvEngine;

constexpr CE::Config kConfig = [] {
    CE::Config c;
    c.block_size  = kBlockSize;
    c.max_ir_len  = kMaxIrLen;
    c.num_outputs = kMics;
    return c;
}();
static_assert(CE::SlowFloats(kConfig) == 0, "cabinets need no SDRAM stage");

constexpr size_t kFastFloats   = CE::FastFloats(kConfig);
constexpr size_t kMediumFloats = CE::MediumFloats(kConfig);

// Hot per-callback state and the FFT tables in DTCM, the stage-1 spectra and
// delay line in AXI SRAM. The FFT tables would otherwise be read from QSPI.
DAISYCONVOL_DTCM alignas(32) float   fast_mem[kFastFloats];
alignas(32) float                    medium_mem[kMediumFloats];
DAISYCONVOL_DTCM alignas(32) uint8_t fft_tables[CE::FftTableBytes(kConfig)];


DaisySeed hw;
CE        engine;

// Controls, written by the main loop, read by the callback.
volatile float blend    = 0.5f; // 0 = mic A only, 1 = mic B only
volatile float level    = 1.0f;
volatile bool  wide     = false;
volatile bool  bypassed = false;

// Callback load (DWT cycles), for the once-per-second log.
volatile uint32_t cb_cycles_peak = 0;
volatile uint64_t cb_cycles_sum  = 0;
volatile uint32_t cb_count       = 0;

float mic_buf[kMics][kBlockSize];

void AudioCallback(AudioHandle::InputBuffer  in,
                   AudioHandle::OutputBuffer out,
                   size_t                    size)
{
    const uint32_t t0 = DWT->CYCCNT;

    float* mics[kMics] = {mic_buf[0], mic_buf[1]};
    engine.ProcessBlock(in[0], mics, size); // silences on a size mismatch

    const float g  = level;
    const float gb = g * blend, ga = g - gb;
    if(bypassed)
    {
        for(size_t i = 0; i < size; i++)
            out[0][i] = out[1][i] = in[0][i];
    }
    else if(wide)
    {
        for(size_t i = 0; i < size; i++)
        {
            out[0][i] = g * mic_buf[0][i];
            out[1][i] = g * mic_buf[1][i];
        }
    }
    else
    {
        for(size_t i = 0; i < size; i++)
            out[0][i] = out[1][i] = ga * mic_buf[0][i] + gb * mic_buf[1][i];
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
    usblog::Line(hw, "keys: [ ] blend A/B | - + level | w wide | b bypass | h help");
}

void PrintState()
{
    usblog::Line(hw, "blend %d%% B | level %d%% | %s%s",
                 (int)(blend * 100.0f + 0.5f),
                 (int)(level * 100.0f + 0.5f),
                 wide ? "wide (L = A, R = B)" : "mono blend",
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
                             {nullptr, 0, 0});
    const float* irs[kMics]  = {kCabMicA, kCabMicB};
    size_t       lens[kMics] = {kCabMicALen, kCabMicBLen};
    for(size_t m = 0; m < kMics; m++)
        if(lens[m] > kMaxIrLen)
            lens[m] = kMaxIrLen; // longer IRs: truncated
    ok = ok && engine.SetIR(irs, lens)
         && daisyconvol::FftRelocateTables(fft_tables, sizeof(fft_tables));

    hw.StartLog(false);
    hw.usb_handle.SetReceiveCallback(OnUsbReceive, UsbHandle::FS_INTERNAL);
    if(!ok)
        for(;;)
        {
            usblog::Line(hw, "cabsim: engine init failed");
            System::Delay(1000);
        }

    hw.StartAudio(AudioCallback);

    constexpr uint32_t kBudget = kBlockSize * 10000; // cycles per block
    uint32_t           last    = System::GetNow();
    bool               led     = false;
    System::Delay(500); // let the USB serial port come up
    usblog::Line(hw, "cabsim: %u-sample IRs, 2 mics, block %u",
                 (unsigned)engine.ir_len(),
                 (unsigned)kBlockSize);
    PrintHelp();
    PrintState();

    for(;;)
    {
        const char k = pending_key;
        if(k)
        {
            pending_key = 0;
            switch(k)
            {
                case '[': blend = fmaxf(0.0f, blend - 0.1f); break;
                case ']': blend = fminf(1.0f, blend + 0.1f); break;
                case '-': level = fmaxf(0.0f, level - 0.1f); break;
                case '+':
                case '=': level = fminf(2.0f, level + 0.1f); break;
                case 'w': wide = !wide; break;
                case 'b': bypassed = !bypassed; break;
                case 'D':
                    System::ResetToBootloader(
                        System::BootloaderMode::DAISY_INFINITE_TIMEOUT);
                    break;
                default: PrintHelp(); break;
            }
            PrintState();
        }

        const uint32_t now = System::GetNow();
        if(now - last >= 1000)
        {
            last = now;
            __disable_irq();
            const uint32_t n    = cb_count;
            const uint64_t sum  = cb_cycles_sum;
            const uint32_t peak = cb_cycles_peak;
            cb_count = 0;
            cb_cycles_sum  = 0;
            cb_cycles_peak = 0;
            __enable_irq();
            const uint32_t avg = n ? (uint32_t)(sum / n) : 0;
            // The engine's own averages over the same second: stage 0 and
            // the time-sliced stage 1, per block.
            const auto pf = engine.perf();
            engine.ResetPerf();
            usblog::Line(hw, "callback avg %lu (%lu.%lu%%) peak %lu (%lu%%) cyc | "
                         "engine avg %lu: s0 %lu sliced %lu",
                         avg,
                         avg * 100 / kBudget,
                         avg * 1000 / kBudget % 10,
                         peak,
                         peak * 100 / kBudget,
                         pf.callback.Avg(),
                         pf.stage0.Avg(),
                         pf.stage1_slice.Avg());
            hw.SetLed(led = !led);
        }
    }
}
