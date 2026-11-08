# daisyConvol

Zero-latency partitioned convolution for the Daisy Seed. Work happens on
`develop`; `main` only receives releases.

## Before merging into main

Every merge into `main` is a release. Before merging, always:

1. **Update the version.**
   - Pick the next version (semver; while 0.x, a change to a public header or
     to `src/daisyconvol/fft_backend.h` bumps the minor version).
   - Update the install line in `README.md`
     (`git -C lib/daisyconvol checkout vX.Y.Z`) and any other mention of the
     version.
   - After the merge: an annotated tag `vX.Y.Z` on the merge commit in
     `main`, pushed to origin. Never move a tag that was already pushed.

2. **Update every measurement in the READMEs** (`README.md` and
   `examples/*/README.md`) so it matches the code being merged. Do not merge
   with stale figures; if one cannot be measured again (e.g. no board at
   hand), tell the user which ones before merging.
   - CPU and timing (Daisy Seed, 480 MHz, 48 kHz, block 32 unless noted):
     the feature bullets, the examples table, the main-loop timing and IR
     switch times, the Performance tables (per configuration, block size,
     reverb length) and the per-example figures in `cabsim`, `hall_reverb`
     and `church_reverb`. Runs of 20-25 s from the USB serial log, skipping
     the first 2 lines, with no underruns, late frames or dropped inputs.
     The convolver rows use its synthetic IR: never touch
     `examples/convolver/ir_data.h` (the user's, git-ignored).
   - Memory and flash ("Memory and flash" in `README.md`): clean builds of
     the examples at their defaults with GCC 10.2.1 (DaisyToolchain),
     figures from the linker's `--print-memory-usage`, `examples/seed`
     against libDaisy 85172e2b and v9.0.0, the QSPI examples against
     85172e2b, plus `arm-none-eabi-size` of `examples/seed` and the flash
     size quoted in "Build and flash".

3. **Merge through a PR `develop` -> `main` with CI green.** The workflow
   only runs on pull requests and on pushes to `main`, never on `develop`.

## Checks

- Host suites and firmware builds: `LIBDAISY_DIR=<built libDaisy> test/all.sh`
  (MSVC, GCC or Clang; ASan with `CMAKE_ARGS=-DDAISYCONVOL_SANITIZE=ON`).
- Kernel changes are compared with code in ITCM (`make ITCM=1` in
  `examples/cabsim`): from QSPI, I-cache layout alone moves the figures.
