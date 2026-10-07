# hall_reverb example

Convolution hall reverb on a bare Daisy Seed: mono in (IN_L), dry signal
plus a 5 s hall IR on both outputs. 48 kHz, block 32.

The first 4096 samples of the IR run in the audio callback; the rest
(2048-sample partitions, spectra and delay line in SDRAM) in the main loop,
in bounded `DoBackgroundStep()` pieces so the loop stays responsive. The
main loop must not block for more than a few milliseconds at 5 s (one
period of that stage is 42.7 ms and the frame itself takes ~30 ms), or
parts of the tail drop out for a moment (counted as underruns). The
logging goes through `../common/usb_log.h`, which never blocks.

## Your IR

    python ../../tools/wav2ir.py "hall.wav" -o reverb_hall_ir.h --name ReverbHallIr --max-seconds 5 --fade-ms 500

(channels are averaged to mono). For a stereo IR add `--stereo`, build with
`make MONO=0` and keep it to ~3 s (`--max-seconds 3`, `IR_LEN=144000`).
The included `reverb_hall_ir.h` is the Marble Hall of Elveden Hall
(Suffolk, England), recorded by Matt Rogalsky, from the
[OpenAIR](https://www.openair.hosted.york.ac.uk/?page_id=459) library,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Modified: channels averaged to mono, truncated to 5 s with a 500 ms
fade-out, normalized to unit energy.

## Build and flash

Runs from QSPI under the Daisy bootloader. Board in DFU (hold BOOT, tap
RESET, release BOOT, or send `D` over the USB serial port):

    make program-boot       # once
    make                    # 5 s mono; e.g. make MONO=0 IR_LEN=144000 for 3 s stereo
    make program-dfu

`make SELFTEST=1` runs an impulse through the engine at boot and prints the
error against the IR (head and tail separately) with the load log.

## Controls (USB serial, 115200)

| Key | |
|-----|---|
| `[` `]` | wet less / more (5 % steps; dry is 1 - wet) |
| `d` | dry on / off |
| `-` `+` | output level |
| `b` | bypass |
| `D` | reset into the bootloader |

Once per second it prints the callback load, the main-loop frame cost and
the total CPU share of each, plus underruns.

## Measured

Daisy Seed, 480 MHz, code in QSPI, block 32, 30 s per run, no underruns in
any. Mono (`MONO=1 IR_LEN=<samples>`):

| IR | Callback avg / peak | Main-loop frame avg / peak | CPU callback + main loop | SDRAM |
|----|---------------------|----------------------------|--------------------------|-------|
| 1 s | 33.8 k / 14 % | 3.07 M / 3.26 M | 10.5 + 14.7 = **25 %** | 1.1 MB |
| 2 s | 34.1 k / 14 % | 5.80 M / 6.05 M | 10.6 + 27.7 = **38 %** | 2.2 MB |
| 3 s | 34.2 k / 14 % | 8.67 M / 9.03 M | 10.7 + 41.5 = **52 %** | 3.3 MB |
| 4 s | 34.4 k / 14 % | 11.4 M / 11.8 M | 10.7 + 54.4 = **65 %** | 4.4 MB |
| 5 s | 34.7 k / 14 % | 14.3 M / 14.8 M | 10.8 + 68.3 = **79 %** | 5.6 MB |
| 6 s | 34.8 k / 14 % | 16.9 M / 17.5 M | 10.8 + 81.1 = **92 %** | 6.7 MB |

Stereo (`MONO=0`): 2 s 17.0 + 45 = ~62 %, 3 s 17.1 + 67 = ~84 %.

The main loop has one frame period (2048 samples = 20.48 M cycles) for each
frame, and the audio callbacks take ~11 % of that time: at 6 s the peak
frame (17.5 M) plus the callbacks (~2.2 M) is 97 % of the window, with no
headroom left for anything else in the loop. 5 s is the practical maximum
for a mono reverb with the default settings; each extra second costs ~13.5 %
CPU (SDRAM bandwidth).
