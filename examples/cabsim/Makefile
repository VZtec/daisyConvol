# cabsim - two-mic guitar cabinet simulator (see README.md).
#
#   make LIBDAISY_DIR=/path/to/libDaisy     (default: next to this repository)
#   make program-dfu
TARGET = cabsim
CPP_SOURCES = cabsim.cpp
LIBDAISY_DIR ?= ../../../libDaisy
C_INCLUDES += -I../common
CPP_STANDARD = -std=gnu++17

# Runs from QSPI under the Daisy bootloader (make program-boot once).
APP_TYPE = BOOT_QSPI
OPT = -O2

# daisyConvol: block 32, IRs up to 4096 samples (only those FFT sizes).
DAISYCONVOL_BLOCKS = 32
DAISYCONVOL_MAX_IR = 4096
include ../../daisyconvol.mk
