# convolver - benchmark: any block size, IR length and output count, with a
# per-stage cycle log over USB serial (see README.md).
#
#   make LIBDAISY_DIR=/path/to/libDaisy     (default: next to this repository)
#   make program-dfu
TARGET = convolver
CPP_SOURCES = convolver.cpp
LIBDAISY_DIR ?= ../../../libDaisy
C_INCLUDES += -I../common
CPP_STANDARD = -std=gnu++17

# Audio block size (power of two, 16..256: libDaisy's AudioHandle limit).
BLOCK ?= 32
C_DEFS += -DCONV_BLOCK=$(BLOCK)
# Maximum (and synthetic) IR length in samples.
ifdef IR_LEN
C_DEFS += -DCONV_IR_LEN=$(IR_LEN)
endif
# STEREO=1: two outputs, each with its own IR (shared input delay lines).
ifeq ($(STEREO),1)
C_DEFS += -DCONV_STEREO=1
endif
# TABLES_IN_RAM=0: leave the FFT tables in (QSPI) flash.
ifeq ($(TABLES_IN_RAM),0)
C_DEFS += -DCONV_TABLES_IN_RAM=0
endif
# With an ir_data.h (generate it with ../../tools/wav2ir.py) the app
# convolves with that IR instead of the synthetic reverb.
ifneq ($(wildcard ir_data.h),)
C_DEFS += -DCONV_USE_EMBEDDED_IR
endif

# The FFT twiddle tables, the USB logger and an embedded IR do not fit the
# 128 KB internal flash: the app runs from QSPI under the Daisy bootloader
# (install it once with `make program-boot`, then `make program-dfu`).
APP_TYPE = BOOT_QSPI
OPT = -O2

# daisyConvol: only the FFT sizes this block size and IR length need.
DAISYCONVOL_BLOCKS = $(BLOCK)
DAISYCONVOL_MAX_IR = $(IR_LEN)
include ../../daisyconvol.mk
