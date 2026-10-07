# church_reverb example

Convolution church reverb on a bare Daisy Seed: mono in (IN_L), dry signal
plus a 5 s IR of the Iglesia San Pedro in Chivilcoy, Argentina, on both
outputs. 48 kHz, block 32.

Same engine setup as [hall_reverb](../hall_reverb/README.md): the first 4096
samples of the IR run in the audio callback, the rest (2048-sample
partitions, spectra and delay line in SDRAM) in the main loop in bounded
`DoBackgroundStep()` pieces. The church decays in about 5.5 s (RT60): the IR
is cut at 5 s, the practical maximum for a mono reverb (~79 % CPU), with a
500 ms fade-out.

## The IR

`reverb_church_ir.h` is the far microphone position of the Iglesia San
Pedro, Chivilcoy, Argentina, captured by [VZtec](https://www.vztecfx.com/)
(2026-05-09, AKG 414 spaced pair, balloon burst; measurement team Nicolás
Benaghi, Beto Dorsa and Ignacio Viano), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modified:
channels averaged to mono, truncated to 5 s with a 500 ms fade-out,
normalized to unit energy. Generated with:

    python ../../tools/wav2ir.py "Iglesia San Pedro Chivilcoy - Far.wav" -o reverb_church_ir.h --name ReverbChurchIr --max-seconds 5 --fade-ms 500

For a stereo version add `--stereo`, keep it to ~3 s (`--max-seconds 3`)
and build with `make MONO=0 IR_LEN=144000`.

## Build and flash

Runs from QSPI under the Daisy bootloader. Board in DFU (hold BOOT, tap
RESET, release BOOT, or send `D` over the USB serial port):

    make program-boot       # once
    make
    make program-dfu

`make SELFTEST=1` runs an impulse through the engine at boot and prints the
error against the IR with the load log.

## Controls (USB serial, 115200)

| Key | |
|-----|---|
| `[` `]` | wet less / more (5 % steps; dry is 1 - wet) |
| `d` | dry on / off |
| `-` `+` | output level |
| `b` | bypass |
| `D` | reset into the bootloader |

Once per second it prints the callback load, the main-loop frame cost and
the total CPU share of each, plus underruns. See the cost table for 1-6 s
IRs in the [hall_reverb README](../hall_reverb/README.md).
