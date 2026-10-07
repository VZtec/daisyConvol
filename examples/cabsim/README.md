# cabsim example

Two-mic guitar cabinet simulator on a bare Daisy Seed: mono in (IN_L), two
cabinet IRs of the same cab (e.g. an SM57 and a ribbon), blended into both
outputs or spread left / right. 48 kHz, block 32.

One engine with two outputs (`Config::num_outputs = 2`): both mics share the
input delay lines and their FFTs, so the second mic costs only its MACs and
IFFTs. 4096-sample IRs (85 ms) run entirely in the audio callback: no
main-loop work, no SDRAM. The FFT tables are copied to DTCM at boot
(`FftRelocateTables`).

## Your IRs

    python ../../tools/wav2ir.py "cab SM57.wav" -o cab_mic_a.h --name CabMicA --max-seconds 0.0853334
    python ../../tools/wav2ir.py "cab R1.wav"   -o cab_mic_b.h --name CabMicB --max-seconds 0.0853334

(`--max-seconds 0.0853334` = 4096 samples; each IR is normalized to unit
energy). The included ones are a Marshall 4x12 with Celestion Greenbacks,
mic A an SM57 and mic B an R1 ribbon (VZtec captures).

## Build and flash

Runs from QSPI under the Daisy bootloader. Board in DFU (hold BOOT, tap
RESET, release BOOT, or send `D` over the USB serial port):

    make program-boot       # once
    make
    make program-dfu

## Controls (USB serial, 115200)

| Key | |
|-----|---|
| `[` `]` | blend towards mic A / mic B (10 % steps) |
| `w` | mono blend <-> wide (L = mic A, R = mic B) |
| `-` `+` | output level |
| `b` | bypass |
| `D` | reset into the bootloader |

Once per second it prints the audio callback load.

## Measured

Daisy Seed, 480 MHz, code in QSPI, two 4096-sample IRs: callback 52.5 k
cycles average (16.4 % of the 320 k per block), 78 k peak (24 %).
