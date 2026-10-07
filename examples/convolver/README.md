# convolver example

Mono in (IN_L) -> IR (up to 3 s) -> both outputs at 48 kHz. The early
stages run in the audio callback, the last one in the main loop
(`DoBackgroundWork()`).

It runs from QSPI under the Daisy bootloader (the FFT tables, the USB
logger and an IR do not fit the 128 KB internal flash). Install the
bootloader once, then flash the app, each time with the board in DFU (hold
BOOT, tap RESET, release BOOT):

    make program-boot       # once
    make                    # or: make BLOCK=64 IR_LEN=96000  (16 .. 1024; IR in samples)
    make program-dfu

Without `ir_data.h` it convolves with a synthetic reverb (IR_LEN, default 3 s) generated at
boot. At boot it prints the geometry it uses; then, once per second, the
worst callback, stage-0 and time-sliced cycles and the main-loop frame cost.

## Using a real impulse response

The IR is embedded in the firmware as a `const float[]`:

    python ../../tools/wav2ir.py path/to/your_ir.wav -o ir_data.h
    make && make program-dfu

`SetIR()` reads the array once at startup; the audio path never touches
QSPI afterwards.

`wav2ir.py` handles 16/24/32-bit PCM and float32 WAV, mono or multichannel
(averaged), resamples to 48 kHz if needed, truncates to `--max-seconds`
(default 3 s), and unit-energy normalizes (`--gain-db X`, `--no-normalize`
to disable). A synthetic `tools/test_ir.wav` (0.8 s) is included to try the
flow. Delete `ir_data.h` to go back to the synthetic reverb.

Sending the byte `D` over the USB serial port resets the board into the
Daisy bootloader, so it can be reflashed without touching BOOT/RESET.
