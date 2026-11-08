# daisyConvol

Zero-latency partitioned convolution for the [Daisy Seed](https://daisy.audio/products/seed3)
(STM32H750): guitar cabinets, convolution reverbs and any FIR from a few
samples to seconds long, in real time.

[![Tests](https://github.com/VZtec/daisyConvol/actions/workflows/test.yml/badge.svg)](https://github.com/VZtec/daisyConvol/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform: Daisy Seed](https://img.shields.io/badge/platform-Daisy%20Seed%20%C2%B7%20STM32H750-8a2be2.svg)](https://daisy.audio/products/seed3)
[![Standard: C++14](https://img.shields.io/badge/C%2B%2B-14-00599c.svg)](#requirements)

## Highlights

- **No added latency.** Non-uniform partitioned convolution (Gardner /
  zita-convolver style): the head of the IR is computed in full inside every
  audio callback, and the longer partitions are scheduled against exact
  deadlines.
- **From cabinets to 5 s reverbs.** A 4096-sample cabinet runs entirely in
  the callback, from internal RAM. Longer IRs add a stage that runs in the
  main loop from SDRAM, which can also be done in bounded steps. Measured: a
  5 s mono reverb takes ~76 % of the CPU at 480 MHz, a two-mic cabinet ~11 %.
- **One input, up to 4 outputs**, each with its own IR. The outputs share the
  input FFTs and delay lines, so a stereo reverb from a mono guitar costs far
  less than two engines.
- **Seamless IR switching.** You can precompute any number of IRs and switch
  between them while audio runs. Switching is lock-free and leaves no gap,
  including for reverbs that have a main-loop stage.
- **Real-time safe.** No heap, no locks and no exceptions, with a bounded
  worst case per callback. All memory comes from arenas you place yourself
  (DTCM / AXI SRAM / SDRAM).
- **Verified against direct convolution.** On the target's own CMSIS-DSP
  float32 FFT built for the PC, the error is −132 to −139 dB across block
  sizes and IR shapes. On a Daisy, every reverb example's measured impulse
  response matches its IR to −135 dB.

## Quick start

### Requirements

- A [Daisy Seed](https://daisy.audio/products/seed3). IRs longer than 4096
  samples use its 64 MB SDRAM. All timings assume the 480 MHz clock
  (`hw.Init(true)`).
- A built [libDaisy](https://github.com/electro-smith/libDaisy), developed
  against `85172e2b5c9abea1bf56dcd5d2c7b3f5717eb66b` (Jan 2024). Every
  example also builds against v9.0.0, the latest release, and CI builds
  against both. With libDaisy v7.0.0 or later, fetch its submodules
  (`git submodule update --init --recursive`), because CMSIS-DSP is one of
  them.
- The [Arm GNU toolchain](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
  and GNU make. The
  [DaisyToolchain](https://github.com/electro-smith/DaisyToolchain) has both.
  Built with GCC 10.2.1 (DaisyToolchain); CI uses 10.3.
- **DaisySP is not needed.** The FFT is CMSIS-DSP's, compiled from libDaisy's
  tree.
- For the host tests: CMake 3.16+ and any C++14 compiler (GCC, Clang, MSVC).

### Add the library

As a git submodule next to libDaisy:

```sh
git submodule add https://github.com/VZtec/daisyConvol lib/daisyconvol
git -C lib/daisyconvol checkout v0.2.0
```

Your Makefile names its sources and libDaisy, sets any [build
options](#build-options), and includes `daisyconvol.mk` *instead of*
libDaisy's core Makefile (the `.mk` includes the core Makefile itself):

```make
TARGET = mycab
CPP_SOURCES = main.cpp
LIBDAISY_DIR = lib/libDaisy
DAISYCONVOL_MAX_IR = 4096
include lib/daisyconvol/daisyconvol.mk
```

### Write the firmware

Convert your IR to a C header (mono, resampled to 48 kHz, unit-energy
normalized). 0.085 s keeps it within 4096 samples:

```sh
python lib/daisyconvol/tools/wav2ir.py my_cab.wav -o my_ir.h --name MyIr --max-seconds 0.085
```

Then write `main.cpp`. [`examples/seed/main.cpp`](examples/seed/main.cpp) is
the complete version, with error handling and an LED:

```cpp
#include "daisy_seed.h"
#include "daisyconvol/daisyconvol.h"
#include "daisyconvol_daisy/seed.h"
#include "my_ir.h" // kMyIr[kMyIrLen]

using daisyconvol::ConvEngine;

constexpr size_t kBlock = 32;

constexpr ConvEngine::Config MakeConfig()
{
    ConvEngine::Config c;
    c.block_size = kBlock;
    c.max_ir_len = 4096; // a cabinet: no SDRAM, no main-loop stage
    return c;
}
constexpr ConvEngine::Config kConfig = MakeConfig();

// The engine's memory, sized at compile time from the same Config.
DAISYCONVOL_DTCM alignas(32) float fast_mem[ConvEngine::FastFloats(kConfig)];
alignas(32) float medium_mem[ConvEngine::MediumFloats(kConfig)];

daisy::DaisySeed hw;
ConvEngine       engine;

void AudioCallback(daisy::AudioHandle::InputBuffer  in,
                   daisy::AudioHandle::OutputBuffer out,
                   size_t                           size)
{
    engine.ProcessBlock(in[0], out[0], size); // size must be kBlock
    for(size_t i = 0; i < size; i++)
        out[1][i] = out[0][i];
}

int main()
{
    hw.Init(true); // 480 MHz
    hw.SetAudioBlockSize(kBlock);
    daisyconvol_daisy::EnableFlushToZero();

    engine.Init(kConfig,
                {fast_mem, ConvEngine::FastFloats(kConfig), 0},
                {medium_mem, ConvEngine::MediumFloats(kConfig), 0},
                {}); // SlowFloats(kConfig) == 0: no SDRAM arena
    engine.SetIR(kMyIr, kMyIrLen); // not real-time safe: before StartAudio
    hw.StartAudio(AudioCallback);

    for(;;)
        engine.DoBackgroundWork(); // the main-loop stage of long IRs
}
```

For a reverb, raise `max_ir_len` (e.g. `2 * 48000`), add the SDRAM arena,
and keep the main loop responsive (see [The main loop](#the-main-loop)):

```cpp
DAISYCONVOL_SDRAM alignas(32) float slow_mem[ConvEngine::SlowFloats(kConfig)];
// ... engine.Init(kConfig, fast, medium, {slow_mem, ConvEngine::SlowFloats(kConfig), 0});
```

### Build and flash

```sh
make
make program-dfu
```

Before `make program-dfu`, put the Seed in DFU mode: hold BOOT, press RESET,
release BOOT. Alternatively, flash `build/mycab.bin` with the [Daisy Web
Programmer](https://electro-smith.github.io/Programmer/).

A cabinet app fits the 128 KB internal flash: `examples/seed` takes 103 KB.
Long IRs embedded as headers do not fit (a 5 s IR is 940 KB). Such apps run
from QSPI under the Daisy bootloader: set `APP_TYPE = BOOT_QSPI` in the
Makefile, run `make program-boot` once, then `make program-dfu` as usual.

To try the library first, build the minimal example:

```sh
make -C examples/seed LIBDAISY_DIR=/path/to/libDaisy
make -C examples/seed program-dfu
```

## Examples

The examples run on a bare Daisy Seed: mono input on IN_L, both outputs. The
IRs they use are included. Each Makefile expects libDaisy next to this
repository; you can also pass `LIBDAISY_DIR=...`. The examples other than
`seed` are controlled with single keys over the USB serial port and run from
QSPI (`make program-boot` once).

| Example | What it is | CPU (480 MHz, block 32) |
|---------|------------|-------------------------|
| [seed](examples/seed/main.cpp) | The minimal firmware: one 4096-sample cabinet IR, internal flash, C++14 | not yet measured |
| [cabsim](examples/cabsim/README.md) | Guitar cabinet: Marshall 4x12 Greenback, SM57 + R1 mics, blend or wide stereo; all in the audio callback | ~11 % |
| [hall_reverb](examples/hall_reverb/README.md) | 5 s hall reverb (Elveden Hall, Marble Hall), dry/wet; `MONO=0` for stereo | ~76 % |
| [church_reverb](examples/church_reverb/README.md) | 5 s church reverb (Iglesia San Pedro, Chivilcoy), dry/wet | ~76 % |
| [irswitch](examples/irswitch/README.md) | Switching IRs while audio runs, regenerating the free one in bounded steps | ~6 % (cab) |
| [convolver](examples/convolver/README.md) | Benchmark: any block size / IR length / outputs, per-stage cycle log | see [Performance](#performance) |

## How it works

**Partitioning.** The IR is split into stages whose partitions grow in size:

- **Stage 0** uses partitions of one block (B samples). It is computed in
  full in every callback, which is what makes the engine zero-latency.
- **Each further stage** uses partitions of P samples and starts at IR offset
  2P. That offset gives the stage one full period of P/B blocks to compute
  each frame. The work is either time-sliced inside the callback (FFT, MACs
  and IFFT spread over the period) or, for the last stage of a long IR, done
  in the main loop. The main loop and the callback exchange data through a
  lock-free single-producer / single-consumer handoff.

The convolution is overlap-save with 2P-point real FFTs. They run on the
complex FFT of CMSIS-DSP (the butterflies of `arm_rfft_fast_f32`), without
its bit-reversal pass: the real-spectrum split reads the bins where the
butterflies leave them, and the inverse is the same forward transform
(IDFT(Z)[m] = DFT(Z)[-m] / M), computing only the half of the output that
overlap-save keeps.

**Choosing the geometry.** `ConvEngine::MakeGeometry` derives it from the
block size and the maximum IR length, or you can pass your own in
`Config::geometry`. The rules:

- Partitions are powers of two up to 2048 (the 4096-point CMSIS FFT limit)
  and grow ×8 per stage.
- A stage runs in the callback when its FFT costs ≤ 15 % of the block budget
  and its period is ≥ 4 blocks.
- Otherwise it becomes the last stage, which runs in the main loop with
  2048-sample partitions. That stage is bound by SDRAM bandwidth, and the
  largest partitions use it best.

With `allow_background = false`, everything stays in the callback.

**Memory.** You supply three arenas, sized at compile time by
`FastFloats / MediumFloats / SlowFloats(cfg)`:

| Arena | Holds | Typically in |
|-------|-------|--------------|
| fast | stage 0 (spectra, delay line), windows, accumulators, scratch | DTCM (`DAISYCONVOL_DTCM`) |
| medium | spectra and delay lines of the in-callback stages; the main-loop stage's input ring and output buffers | AXI SRAM (plain `.bss`) |
| slow | spectra and delay line of the main-loop stage | SDRAM (`DAISYCONVOL_SDRAM`) |

An arena of size 0 is not needed: pass an empty `{}`. The default geometries
for a 2 s reverb (96000 samples), one output:

| B | Stages (partitions x size) | Fast + medium | Slow (SDRAM) |
|---|----------------------------|---------------|--------------|
| 16 | 16 x 16, 30 x 128, 45 x 2048 | 40 + 108 KB | 1.4 MB |
| 32 | 16 x 32, 14 x 256, 45 x 2048 | 48 + 104 KB | 1.4 MB |
| 64 | 16 x 64, 6 x 512, 45 x 2048 | 65 + 96 KB | 1.4 MB |
| 128 | 16 x 128, 2 x 1024, 45 x 2048 | 98 + 80 KB | 1.4 MB |
| 256 | 16 x 256, 45 x 2048 | 102 + 48 KB | 1.4 MB |

For a 4096-sample cabinet nothing runs in the main loop and no SDRAM is used.
At B = 32 the geometry is 16 x 32 + 14 x 256, with 16 + 56 KB of arenas. Each
extra output adds its IR spectra, accumulators and output buffers, but not
the delay lines. The fast arena grows with the block size: past B = 64, put
it in AXI SRAM rather than DTCM.

**Limits.**

- Block sizes: powers of two from 16 to 2048 in the engine. libDaisy's
  `AudioHandle` caps the block at 256 (`SetAudioBlockSize` clamps silently),
  so check `hw.AudioBlockSize()`. A block of the wrong size is silenced and
  counted in `perf().block_mismatches`; it never overruns a buffer.
- One input and up to 4 outputs. True stereo (2 in × 2 out) needs two
  engines with two outputs each.
- At 480 MHz the main loop keeps up with about 5 s of IR in mono and 3 s in
  stereo, a limit set by SDRAM bandwidth.

## Repository layout

| Path | Contents |
|------|----------|
| [`src/daisyconvol/`](src/daisyconvol/) | The DSP kernel, platform independent (no libDaisy, C++14, builds on any host): `daisyconvol.h`, `conv_engine.h/.cpp`, the FFT interface `fft_backend.h` and its portable double-precision implementation `fft_backend_ref.cpp` |
| [`src/daisyconvol_daisy/`](src/daisyconvol_daisy/) | The Daisy Seed layer: `seed.h` (memory sections, flush-to-zero, cycle counter), the CMSIS-DSP FFT backend `fft_backend_cmsis.cpp` and `cmsis_bitreversal.c` (Apache-2.0) |
| [`daisyconvol.mk`](daisyconvol.mk) | Make integration for libDaisy projects |
| [`examples/`](examples/) | `seed` (minimal), `cabsim`, `hall_reverb`, `church_reverb`, `irswitch`, `convolver`; `common/usb_log.h` is a non-blocking USB serial logger |
| [`test/`](test/) | Host test suites (CMake) and `all.sh`, which runs everything |
| [`tools/wav2ir.py`](tools/wav2ir.py) | WAV to C header: mono or `--stereo`, resampling, truncation with fade-out, unit-energy normalization, `--name` for several IRs |
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | CI: host suites with sanitizers, CMSIS suites and firmware builds against two libDaisy versions |

## API

Everything is in namespace `daisyconvol` (`daisyconvol/daisyconvol.h`),
except the Seed helpers in `daisyconvol_daisy` (`daisyconvol_daisy/seed.h`).
`conv_engine.h` documents every call. In the table, *callback* means the
audio callback (an interrupt) and *main* means `main()` / the main loop.

| Call | Context | What it does |
|------|---------|--------------|
| `ConvEngine::FastFloats(cfg)`, `MediumFloats(cfg)`, `SlowFloats(cfg)` | compile time | Arena sizes in floats for a `Config` (0 = arena not needed) |
| `ConvEngine::SpectraFloats(cfg)`, `PrepareScratchFloats(cfg)` | compile time | Storage for one precomputed IR, and the scratch `PrepareIR` needs (external IR mode) |
| `ConvEngine::FftTableBytes(cfg)` | compile time | Buffer size for `FftRelocateTables` |
| `ConvEngine::MakeGeometry(b, max_ir, ...)`, `ConfigGeometry(cfg)` | compile time | The default partition geometry / the one `Init(cfg)` will use |
| `Init(cfg, fast, medium, slow)` | main, audio stopped | Lays out the arenas (and zeroes them), creates the FFT plans; false if the config is invalid or an arena is too small |
| `SetIR(ir, len)`, `SetIR(irs, lens)` | main, audio stopped | Computes the partition spectra of one IR per output; false if too long, NaN / inf, or in external IR mode |
| `PrepareIR(irs, lens, storage, n, scratch, n, &spectra)` | main, any time | External IR mode: computes an IR's spectra into your storage, in one call (tens of ms for long IRs) |
| `BeginPrepareIR(&job, ...)`, `PrepareIRStep(&job)` | main, while audio runs | The same in steps of one partition FFT (≤ ~0.5 ms each); bit-identical result |
| `SwapIR(&spectra)` | main | External IR mode: switches to a prepared IR at the next period boundary, lock-free (`nullptr` = silence) |
| `SwapPending()`, `ActiveIR()` | main | Whether a switch is still in progress / the IR in effect; once not pending, any other IR's storage may be reused |
| `ProcessBlock(in, out, n)` | callback | Convolves one block (n must be the block size); `in == out` allowed |
| `ProcessBlock(in, outs, n)` | callback | Same, all outputs (`outs[c]`, e.g. libDaisy's `OutputBuffer`) |
| `Reset()` | callback (or audio stopped) | Clears the delay lines, keeps the IR; a main-loop stage acknowledges it at its next step |
| `DoBackgroundWork()` | main loop | Runs all pending main-loop stage work (no-op without that stage) |
| `DoBackgroundStep()` | main loop | One bounded piece of it (~0.3-0.7 ms at 480 MHz); returns true if more may be pending |
| `perf()`, `ResetPerf()` | main | Consistent snapshot of cycle counts and counters (underruns, late frames, non-finite inputs, block mismatches) / clears them |
| `geometry()`, `block_size()`, `num_outputs()`, `max_ir_len()`, `ir_len()` | any | Current configuration |
| `FftRelocateTables(buf, bytes)` | main, after `Init`, audio stopped | Copies the FFT twiddle tables from flash (QSPI) to RAM |
| `DAISYCONVOL_DTCM`, `DAISYCONVOL_SDRAM` | declarations | Place an arena in the DTCM / SDRAM (NOLOAD sections; `Init` zeroes the arenas) |
| `daisyconvol_daisy::EnableFlushToZero()` | main, after `hw.Init` | Flush denormals to zero, in the main loop and in interrupts |
| `daisyconvol_daisy::EnableCycleCounter()`, `CycleCount()` | main / `Config::get_cycles` | DWT cycle counter, so that `perf()` reports cycles |

`Config` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `block_size` | 32 | B: power of two, 16..2048 (≤ 256 with libDaisy) |
| `max_ir_len` | 144000 | Longest IR in samples; sizes the geometry and the arenas |
| `num_outputs` | 1 | Outputs (1..4), each with its own IR |
| `allow_background` | true | Allow a main-loop stage for long IRs |
| `external_ir` | false | IRs come from `PrepareIR` + `SwapIR` instead of `SetIR` |
| `external_background` | false | External IR mode: let long IRs use a main-loop stage too |
| `geometry` | nullptr | A custom `Geometry` (must be `Valid()`); enable its FFT sizes with `DAISYCONVOL_FFT_SIZES` |
| `sanitize_input` | true | NaN / infinite input samples become 0 (~2 cycles per sample) |
| `cycles_per_sample` | 10000 | CPU budget the default geometry assumes (480 MHz / 48 kHz) |
| `get_cycles` | nullptr | Cycle counter for `perf()`, e.g. `daisyconvol_daisy::CycleCount` |

### The main loop

`DoBackgroundWork()` must finish each main-loop frame (2048 samples, 42.7 ms
at 48 kHz) within one such period after its input is complete. A frame takes
~12 ms at 480 MHz for a 2 s mono IR and ~30 ms for 5 s. The audio callbacks
take ~7 % of the time on top of that, so the rest of each main-loop
iteration has to fit in what remains.

If the loop has other work (a display, controls), call `DoBackgroundStep()`
instead and interleave it with that work. If the main loop stalls, part of
the reverb tail drops out for a moment (`perf().s2_underruns`), but the
output never glitches and it recovers by itself.

A common way to stall the loop is libDaisy's `hw.PrintLine`. It becomes
blocking once a serial terminal has connected and hangs if the terminal is
then closed. The examples log through `examples/common/usb_log.h`, which
drops the line instead.

### Several outputs

```cpp
c.num_outputs = 2; // in MakeConfig()
const float* irs[2]  = {ir_left, ir_right};
const size_t lens[2] = {len_left, len_right};
engine.SetIR(irs, lens);
// in the callback:
engine.ProcessBlock(in[0], out, size); // writes out[0] and out[1]
```

### Switching IRs while audio runs

```cpp
c.external_ir = true; // in MakeConfig(); IR spectra live in your storage

float spectra[kNumIrs][ConvEngine::SpectraFloats(kConfig)]; // 32 KB each, cabinet at B = 32
float scratch[ConvEngine::PrepareScratchFloats(kConfig)];
ConvEngine::IrSpectra ir[kNumIrs];
for(size_t i = 0; i < kNumIrs; i++)
    engine.PrepareIR(&irs[i], &lens[i], spectra[i], ConvEngine::SpectraFloats(kConfig),
                     scratch, ConvEngine::PrepareScratchFloats(kConfig), &ir[i]);

engine.SwapIR(&ir[1]); // from the main loop, any time
```

The switch keeps the input history. The output goes exactly from the old
IR's convolution to the new one's: a hard cut with no gap, within 16 blocks
with the default geometries. For long IRs, set `external_background = true`.
The switch then completes within ~130 ms, and the IRs should be prepared in
steps with `BeginPrepareIR` / `PrepareIRStep` (see
[`examples/irswitch`](examples/irswitch/irswitch.cpp)).

## Build options

Set these in your Makefile before including `daisyconvol.mk`, or on the make
command line. The `.mk` records the -D options, `OPT` and the C/C++ standard
in `build/daisyconvol.flags`, so changing any of them rebuilds every object.

| Option | Default | Effect |
|--------|---------|--------|
| `LIBDAISY_DIR` | (required) | Built libDaisy checkout: the core Makefile and the CMSIS-DSP sources come from it |
| `DAISYCONVOL_BLOCKS` | `32` | Block size(s) the engine runs at (powers of two, 16..2048). They fix the partition sizes of the default geometries, and only the FFT sizes those need are linked (each costs its CMSIS tables in flash) |
| `DAISYCONVOL_MAX_IR` | unset (any length) | Longest IR in samples. ≤ 4096 leaves out the 4096-point FFT tables (~40 KB of flash) |
| `DAISYCONVOL_FFT_SIZES` | derived from the two above | Explicit FFT sizes (32..4096), for a custom `Geometry` or `cycles_per_sample` |

libDaisy's own variables work as usual (`APP_TYPE`, `OPT`, `CPP_STANDARD`,
...). The library builds with libDaisy's default `-std=gnu++14`. Its hot
kernels are compiled at `-O2` even when `OPT = -Os`.

## Memory and flash

These figures come from the linker (`--print-memory-usage`) for clean builds
of the examples at their defaults (block 32, GCC 10.2.1 from the
DaisyToolchain). The figures are for the whole firmware, libDaisy included.

`examples/seed` (4096-sample cabinet, internal flash):

| Region | libDaisy 85172e2b | libDaisy v9.0.0 | Capacity |
|--------|------------------:|----------------:|---------:|
| Flash (internal) | 102,552 B | 109,240 B | 131,072 B |
| DTCM | 16,896 B | 16,896 B | 131,072 B |
| AXI SRAM | 71,588 B | 72,160 B | 524,288 B |
| D2 SRAM (libDaisy's audio DMA buffers) | 16,384 B | 16,384 B | 294,912 B |
| SDRAM | 0 | 0 | 64 MB |

`arm-none-eabi-size` (85172e2b): text 99,548, data 3,004, bss 102,576. Of
these, the engine's arenas are 16 KB (DTCM) + 56 KB (AXI SRAM). The IR is
16 KB of flash, and the FFT tables for block 32 are 5.5 KB.

The QSPI examples (libDaisy 85172e2b):

| Example | QSPI flash | DTCM | AXI SRAM | SDRAM |
|---------|-----------:|-----:|---------:|------:|
| cabsim (two 4096-sample IRs) | 131,100 B | 30,272 B | 112,856 B | 0 |
| hall_reverb (5 s IR) | 1,100,736 B | 49,664 B | 179,472 B | 3,712 KB |
| church_reverb (5 s IR) | 1,100,740 B | 49,664 B | 179,472 B | 3,712 KB |
| irswitch (4096-sample IRs) | 102,008 B | 16,896 B | 161,472 B | 16,388 B |

## Performance

All numbers in this section were measured on a Daisy Seed at 480 MHz and
48 kHz, block 32 unless noted, from the examples' USB serial logs (DWT
cycle counter). The firmware was built from code with the same engine as
this release, in runs of 20-30 s. They are observed values, not worst-case
bounds.

| Example / configuration | Callback avg / peak | Main-loop frame | Total CPU |
|-------------------------|---------------------|-----------------|-----------|
| `examples/seed`: 4096-sample cabinet | not yet measured | - | not yet measured |
| `cabsim`: two mics, 4096 samples each | 34.3 k cycles / 22 % | - | ~11 % |
| `cabsim ITCM=1`: same, engine and FFT code in ITCM | 30.2 k / 16 % | - | ~9 % |
| `convolver IR_LEN=4096`: cabinet | 19.8 k / 20 % | - | ~6 % |
| `convolver`: 2 s IR (96000 samples) | 20.9 k / 18 % | 5.78 M cycles | ~35 % |
| `convolver STEREO=1`: 2 s stereo IR | 34.3 k / 21 % | 9.30 M | ~56 % |
| `convolver TABLES_IN_RAM=0`: FFT tables left in QSPI | 29.6 k / 31 % | 6.37 M | ~40 % |
| `hall_reverb`: 5 s mono IR, dry/wet | 22.6 k / 15 % | 14.2 M | ~76 % |
| `church_reverb`: 5 s mono IR, dry/wet | 22.5 k / 15 % | 14.2 M | ~76 % |
| `hall_reverb MONO=0`: 2 s stereo IR | 34.0 k / 20 % | 9.34 M | ~56 % |

How the columns are computed:

- *Callback peak* is a percentage of the block budget (32 samples × 10,000
  cycles).
- *Total CPU* = callback average / budget + frame cycles × frames per second
  / 480 MHz (one 2048-sample frame every 42.7 ms).

The 2 s IR at other block sizes (`convolver BLOCK=...`):

| B | Callback avg | Callback peak | Main-loop frame | Total CPU |
|---|--------------|---------------|-----------------|-----------|
| 16 | 13.6 k | 28 % | 5.93 M | ~38 % |
| 32 | 20.9 k | 18 % | 5.78 M | ~35 % |
| 64 | 35.9 k | 14 % | 5.72 M | ~34 % |
| 128 | 81.9 k | 14 % | 6.72 M | ~39 % |
| 256 | 101 k | 5 % | 6.67 M | ~37 % |

Reverb length vs. CPU (mono, block 32, `hall_reverb MONO=1 IR_LEN=...`):

| IR | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s |
|----|-----|-----|-----|-----|-----|-----|
| Total CPU | 22 % | 35 % | 49 % | 62 % | 76 % | ~90 % (estimated) |

The callback share stays at ~7 %. Each second of IR adds ~13.5 % in the
main loop, which is the SDRAM bandwidth. The bundled IR is 5 s long, so 6 s
is extrapolated: the frames and the callbacks would fill ~93 % of the frame
period at their peak, which makes 5 s the practical maximum in mono.

Next to other heavy DSP, code placement matters. Running the engine from
QSPI through the 16 KB I-cache doubled its stage-1 cost next to a NAM amp
model; placing the engine and the CMSIS FFT in ITCM avoids that. If your app
runs from QSPI, copy the FFT tables to RAM after `Init()`:
`daisyconvol::FftRelocateTables(buf, sizeof buf)`, with a `buf` of
`ConvEngine::FftTableBytes(cfg)` bytes.

### Measuring your own build

libDaisy's `CpuLoadMeter` measures the audio callback:

```cpp
daisy::CpuLoadMeter load;
// after hw.Init():
load.Init(hw.AudioSampleRate(), hw.AudioBlockSize());
// in the audio callback:
load.OnBlockStart();
engine.ProcessBlock(in[0], out[0], size);
load.OnBlockEnd();
// in the main loop, e.g. once per second:
hw.PrintLine("callback: avg %d %%, max %d %%",
             int(load.GetAvgCpuLoad() * 100.0f), int(load.GetMaxCpuLoad() * 100.0f));
```

With a main-loop stage (IRs > 4096 samples), add its share. Set
`Config::get_cycles = daisyconvol_daisy::CycleCount` and call
`daisyconvol_daisy::EnableCycleCounter()`. Then `perf().stage2_frame.Avg()`
gives the cycles per main-loop frame, and
`stage2_frame.Avg() * (48000.0f / 2048) / 480e6f` is the main-loop share. `perf().s2_underruns` must stay at 0. The `convolver`
example prints all of these once per second.

## Tests

[`test/all.sh`](test/all.sh) runs everything:

```sh
LIBDAISY_DIR=/path/to/libDaisy test/all.sh
```

The [workflow](.github/workflows/test.yml) runs the same suites on every
push.

| Suite | What it checks | Needs |
|-------|----------------|-------|
| `test_direct` | The engine against a direct time-domain convolution (double), for 10 IRs × 5 block sizes (16..256), reported as an error table in dB. Cases: unit impulse, IRs shorter than one partition, one partition, 1000 samples, cabinet length, first main-loop sample, an impulse in the main-loop stage, and the maximum configured length. Pass: < −100 dB | CMake, C++14 compiler |
| `test_conv` | The full suite: every stage boundary for B = 16..1024 and IRs up to 3 s, 1-4 outputs, custom geometries, main-loop stalls and late frames, IR switching (with and without a main-loop stage), `Reset()` inside a main-loop frame, 32-bit counter wraps, bounded main-loop steps, input sanitizing, wrong block sizes, in-place processing, re-`Init`, relocated FFT tables, incremental `PrepareIR`, a randomized mix of all of it, and the FFT sizes `daisyconvol.mk` enables | CMake, C++14 compiler |
| `test_direct_cmsis`, `test_conv_cmsis` | Both suites again on the target's CMSIS-DSP float32 FFT, compiled for the host | `LIBDAISY_DIR` (either layout), MSVC or GCC |
| firmware | Every example builds with `daisyconvol.mk` | `arm-none-eabi-gcc`, a built libDaisy |

`test_direct` on the CMSIS FFT (error in dB):

```
IR                                       B=16     B=32     B=64    B=128    B=256
unit impulse (1 sample)                -138.5   -137.8   -137.3   -136.6   -136.0
3 samples (< one partition)            -136.1   -135.7   -135.4   -135.2   -134.8
15 samples (< one partition)           -135.8   -134.8   -135.2   -134.9   -134.5
16 samples (one partition at B=16)     -137.2   -136.5   -135.7   -135.1   -134.6
1000 samples, decaying                 -133.5   -133.3   -133.1   -133.4   -133.6
4096 samples (cabinet), decaying       -133.0   -133.7   -133.1   -132.7   -132.5
4097 samples (main-loop stage)         -132.7   -133.1   -133.0   -132.9   -133.0
impulse at 9000 (main-loop stage)      -133.9   -133.9   -133.9   -133.9   -133.9
12000 samples (maximum), decaying      -132.4   -133.2   -133.2   -132.9   -132.5
12000 samples (maximum), noise         -132.4   -132.6   -132.6   -132.6   -132.6
```

On the Windows DaisyToolchain, `g++` on PATH is the Arm cross compiler.
CMake picks MSVC there, and `test/all.sh` works from Git Bash.

## References

- W. G. Gardner, "Efficient Convolution without Input-Output Delay",
  *J. Audio Eng. Soc.* 43(3), 1995: the non-uniform partitioning.
- G. García, "Optimal Filter Partition for Efficient Convolution with Short
  Input/Output Delay", 113th AES Convention, 2002, paper 5660.
- F. Wefers, *Partitioned Convolution Algorithms for Real-Time
  Auralization*, PhD thesis, RWTH Aachen, 2015.
- F. Adriaensen, [zita-convolver](https://kokkinizita.linuxaudio.org/linuxaudio/):
  the scheduling style (stages with exact deadlines, a lower-priority thread
  for the long partitions).
- [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP): `arm_rfft_fast_f32`.
- [libDaisy](https://github.com/electro-smith/libDaisy) and the
  [Daisy documentation](https://docs.daisy.audio/).

## License and credits

The code is MIT, see [LICENSE](LICENSE). Third-party work is listed in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md):

| Component | Attribution |
|-----------|-------------|
| `src/daisyconvol_daisy/cmsis_bitreversal.c` | Adapted from CMSIS-DSP's `arm_bitreversal2.c`, © 2019 ARM Limited, [Apache-2.0](src/daisyconvol_daisy/LICENSE-Apache-2.0.txt) |
| CMSIS-DSP FFT (compiled from your libDaisy checkout, not included) | © ARM Limited, Apache-2.0 |
| `examples/cabsim/cab_mic_a.h`, `cab_mic_b.h` | Marshall 4x12 with Celestion Greenbacks, SM57 and R1 mics, captured by [VZtec](https://www.vztecfx.com/), MIT like the code |
| `examples/hall_reverb/reverb_hall_ir.h` | Elveden Hall (Suffolk, England), Marble Hall, by Matt Rogalsky, from the [OpenAIR](https://www.openair.hosted.york.ac.uk/?page_id=459) library, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `examples/church_reverb/reverb_church_ir.h` | Iglesia San Pedro, Chivilcoy (Argentina), far position, captured by [VZtec](https://www.vztecfx.com/) (Nicolás Benaghi, Beto Dorsa, Ignacio Viano), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Both reverb IRs were modified: averaged to mono, cut at 5 s with a 500 ms
fade-out, and normalized to unit energy.
