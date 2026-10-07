# irswitch example

External IR mode: three IRs prepared with `PrepareIR()` into the app's own
storage, switched with `SwapIR()` while audio runs. Mono in (IN_L), both
outputs, 48 kHz.

- Every 2 s (or when `n` arrives over USB serial) the main loop switches to
  the next IR. The output goes from the old IR's convolution to the new one's
  exactly at a period boundary (a hard cut, no gap).
- It prints how long each switch took until `SwapPending()` went false, and
  `ActiveIR()`.
- Once the switch is done, the IR it left is unused: the main loop
  regenerates it (a new variant) while audio keeps running, in bounded
  steps (the IR generation in 2048-sample chunks, then `BeginPrepareIR` /
  `PrepareIRStep`, one partition FFT per step), so the main-loop stage
  never misses its deadline. It prints how long each regeneration took.
- Main-loop stage work runs as bounded `DoBackgroundStep()` pieces next to the
  rest of the loop (an LED heartbeat stands in for UI work).

Build and flash (board in DFU: hold BOOT, tap RESET, release BOOT):

    make program-boot       # once: the Daisy bootloader
    make                    # cabinet-like IRs (4096 samples), all in the callback
    make IR_LEN=48000       # 1 s rooms: adds a main-loop stage, which switches too
    make program-dfu

With `IR_LEN` above 4096 the switch is announced two main-loop stage periods
ahead (85 ms at B = 32), so it takes up to ~130 ms to complete; cabinets
switch within 16 blocks (~11 ms). Sending `D` resets into the bootloader.
