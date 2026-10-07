# irswitch - switching IRs while audio runs (see README.md).
#
#   make LIBDAISY_DIR=/path/to/libDaisy     (default: next to this repository)
#   make program-dfu
TARGET = irswitch
CPP_SOURCES = irswitch.cpp
LIBDAISY_DIR ?= ../../../libDaisy
C_INCLUDES += -I../common
CPP_STANDARD = -std=gnu++17

# Audio block size (power of two, 16..256) and IR length in samples
# (> 4096: the engine adds a main-loop stage).
BLOCK  ?= 32
IR_LEN ?= 4096
C_DEFS += -DCONV_BLOCK=$(BLOCK) -DCONV_IR_LEN=$(IR_LEN)

# USB logger + engine + tables: run from QSPI under the Daisy bootloader
# (make program-boot once, then make program-dfu).
APP_TYPE = BOOT_QSPI
OPT = -O2

DAISYCONVOL_BLOCKS = $(BLOCK)
DAISYCONVOL_MAX_IR = $(IR_LEN)
include ../../daisyconvol.mk
