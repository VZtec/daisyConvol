# daisyConvol on a bare Daisy Seed: a 4096-sample guitar cabinet, all in the
# audio callback, from the internal flash. LIBDAISY_DIR points at a built
# libDaisy checkout (`make` in it first); the default expects it next to
# this repository:
#
#   make LIBDAISY_DIR=/path/to/libDaisy
#   make program-dfu
#
TARGET = daisyconvol_seed
CPP_SOURCES = main.cpp
LIBDAISY_DIR ?= ../../../libDaisy

# The IR is the first mic of examples/cabsim.
C_INCLUDES += -I../cabsim

# Block size 32, IRs up to 4096 samples: only those FFT sizes are linked.
DAISYCONVOL_BLOCKS = 32
DAISYCONVOL_MAX_IR = 4096
include ../../daisyconvol.mk
