// daisyConvol on a bare Daisy Seed: a guitar cabinet. The input (IN_L) is
// convolved with a 4096-sample cabinet IR (85 ms: a Marshall 4x12 with
// Greenbacks, SM57) and sent to both outputs. At this length everything
// runs in the audio callback, with zero added latency, no SDRAM and no
// main-loop stage. The app fits the internal flash: no bootloader needed.
//
// Build with `make` in this folder (LIBDAISY_DIR points at a built libDaisy
// checkout, see the Makefile) and flash build/daisyconvol_seed.bin with the
// Daisy Web Programmer or `make program-dfu`. The LED blinks once per
// second while it runs, fast if the engine could not start.

#include "daisy_seed.h"

#include "daisyconvol/daisyconvol.h"
#include "daisyconvol_daisy/seed.h"

// The IR: kCabMicA[kCabMicALen], from tools/wav2ir.py (examples/cabsim).
#include "cab_mic_a.h"

using daisyconvol::ConvEngine;

namespace
{

// The audio block size the engine is built for (libDaisy allows up to 256).
constexpr size_t kBlock = 32;

// One Config sizes the memory and initializes the engine.
constexpr ConvEngine::Config MakeConfig()
{
    ConvEngine::Config c;
    c.block_size = kBlock;
    c.max_ir_len = 4096;
    return c;
}
constexpr ConvEngine::Config kConfig = MakeConfig();

// The engine's memory, placed by the caller: the state every callback
// touches in the DTCM, the larger partitions in AXI SRAM. A cabinet-length
// IR needs no SDRAM.
constexpr size_t kFastFloats   = ConvEngine::FastFloats(kConfig);
constexpr size_t kMediumFloats = ConvEngine::MediumFloats(kConfig);
static_assert(ConvEngine::SlowFloats(kConfig) == 0, "no SDRAM stage");

DAISYCONVOL_DTCM alignas(32) float fast_mem[kFastFloats];
alignas(32) float                  medium_mem[kMediumFloats];

daisy::DaisySeed hw;
ConvEngine       engine;

void AudioCallback(daisy::AudioHandle::InputBuffer  in,
                   daisy::AudioHandle::OutputBuffer out,
                   size_t                           size)
{
    engine.ProcessBlock(in[0], out[0], size); // silence if size != kBlock
    for(size_t i = 0; i < size; i++)
        out[1][i] = out[0][i];
}

} // namespace

int main()
{
    hw.Init(true); // 480 MHz
    hw.SetAudioBlockSize(kBlock);
    daisyconvol_daisy::EnableFlushToZero();

    // Not real-time safe: before StartAudio. SetIR computes the spectra of
    // every partition of the IR, which then never touches flash again.
    const bool ok
        = hw.AudioBlockSize() == kBlock
          && engine.Init(kConfig,
                         {fast_mem, kFastFloats, 0},
                         {medium_mem, kMediumFloats, 0},
                         {})
          && engine.SetIR(kCabMicA, kCabMicALen);
    if(ok)
        hw.StartAudio(AudioCallback);

    const uint32_t blink_ms = ok ? 500 : 100;
    uint32_t       last     = daisy::System::GetNow();
    bool           led      = false;
    for(;;)
    {
        // The main-loop stage of IRs longer than 4096 samples (a no-op
        // here). With one, never block this loop for long.
        engine.DoBackgroundWork();

        const uint32_t now = daisy::System::GetNow();
        if(now - last >= blink_ms)
        {
            last = now;
            led  = !led;
            hw.SetLed(led);
        }
    }
}
