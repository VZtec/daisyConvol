# hall_reverb - convolution hall reverb (see README.md).
#
#   make LIBDAISY_DIR=/path/to/libDaisy     (default: next to this repository)
#   make program-dfu
TARGET = hall_reverb
CPP_SOURCES = hall_reverb.cpp
LIBDAISY_DIR ?= ../../../libDaisy
C_INCLUDES += -I../common
CPP_STANDARD = -std=gnu++17

# Audio block size (power of two, 16..256) and IR length in samples.
BLOCK  ?= 32
IR_LEN ?= 240000
C_DEFS += -DREVERB_BLOCK=$(BLOCK) -DREVERB_IR_LEN=$(IR_LEN)
# MONO=1 (default): one IR for both sides; MONO=0: stereo IR (left /
# right), ~1.6x the main-loop work (use IR_LEN=144000 or less).
MONO ?= 1
C_DEFS += -DREVERB_MONO=$(MONO)
# SELFTEST=1: impulse response check at boot (printed with the help).
ifeq ($(SELFTEST),1)
C_DEFS += -DREVERB_SELFTEST
endif

# Runs from QSPI under the Daisy bootloader (make program-boot once).
APP_TYPE = BOOT_QSPI
OPT = -O2

# daisyConvol: only the FFT sizes this block size and IR length need.
DAISYCONVOL_BLOCKS = $(BLOCK)
DAISYCONVOL_MAX_IR = $(IR_LEN)
include ../../daisyconvol.mk
