# Third-party notices

daisyConvol is MIT licensed (see [LICENSE](LICENSE)). It contains, or builds
against, the following work by others.

## Code

| Component | Where | Origin | License |
|-----------|-------|--------|---------|
| `arm_bitreversal_32` in C | [`src/daisyconvol_daisy/cmsis_bitreversal.c`](src/daisyconvol_daisy/cmsis_bitreversal.c) | Adapted from CMSIS-DSP `arm_bitreversal2.c` V1.9.0, © 2019 ARM Limited or its affiliates (reduced to the 32-bit function, reformatted) | Apache-2.0, full text in [`src/daisyconvol_daisy/LICENSE-Apache-2.0.txt`](src/daisyconvol_daisy/LICENSE-Apache-2.0.txt) |
| CMSIS-DSP real FFT (`arm_rfft_fast_f32.c`, `arm_cfft_f32.c`, `arm_cfft_radix8_f32.c`, `arm_common_tables.c`) | Not in this repository: `daisyconvol.mk` compiles them from your libDaisy checkout | © ARM Limited or its affiliates | Apache-2.0 |
| libDaisy | Not in this repository | © Electrosmith | MIT |

## Impulse responses (examples only)

| File | Origin | License |
|------|--------|---------|
| `examples/cabsim/cab_mic_a.h`, `cab_mic_b.h` (also used by `examples/seed`) | Marshall 4x12 with Celestion Greenbacks, SM57 and R1 mics, captured by [VZtec](https://www.vztecfx.com/) | MIT, like the code ([LICENSE](LICENSE)) |
| `examples/hall_reverb/reverb_hall_ir.h` | Elveden Hall (Suffolk, England), Marble Hall, by Matt Rogalsky, from the [OpenAIR](https://www.openair.hosted.york.ac.uk/?page_id=459) library | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `examples/church_reverb/reverb_church_ir.h` | Iglesia San Pedro, Chivilcoy (Argentina), far position, captured by [VZtec](https://www.vztecfx.com/) (Nicolás Benaghi, Beto Dorsa, Ignacio Viano) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Both reverb IRs were modified: averaged to mono, cut at 5 s with a 500 ms
fade-out and normalized to unit energy (`tools/wav2ir.py`).
