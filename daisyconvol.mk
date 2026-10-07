# daisyconvol.mk - make integration of daisyConvol for libDaisy projects.
#
# A project's Makefile sets TARGET, its own CPP_SOURCES and LIBDAISY_DIR (a
# built libDaisy checkout), whatever other libDaisy variables and options
# below it wants to change, then includes this file *instead of* libDaisy's
# core Makefile. This file adds the library's sources, include path and
# defines, then includes the core Makefile itself:
#
#   TARGET = mycab
#   CPP_SOURCES = main.cpp
#   LIBDAISY_DIR = lib/libDaisy
#   DAISYCONVOL_MAX_IR = 4096
#   include lib/daisyconvol/daisyconvol.mk
#
# Code then includes "daisyconvol/daisyconvol.h" (the engine) and
# "daisyconvol_daisy/seed.h" (memory sections and Seed helpers).
#
# Options (set before the include or on the make command line):
#   DAISYCONVOL_BLOCKS     audio block size(s) the engine runs at (powers of
#                          two, 16..2048; default 32). The block size fixes
#                          the partition sizes of the default geometry, and
#                          only the FFT sizes those need are linked: each one
#                          costs its CMSIS twiddle / bit-reversal tables in
#                          flash.
#   DAISYCONVOL_MAX_IR     longest IR in samples (default: unset, any
#                          length). Up to 4096 (cabinets) the 4096-point FFT
#                          tables (~40 KB of flash) are left out, except at
#                          block size 2048, whose stage 0 needs them.
#   DAISYCONVOL_FFT_SIZES  explicit FFT sizes instead (32..4096), e.g. for a
#                          custom Geometry or Config::cycles_per_sample.
#
# libDaisy's dependency files do not track -D options or OPT, so this file
# records them (build/daisyconvol.flags): changing an option, on the command
# line too, rebuilds every object.

# Where this file is (evaluated now: MAKEFILE_LIST grows with each include).
ifndef DAISYCONVOL_DIR
DAISYCONVOL_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
endif

ifndef LIBDAISY_DIR
$(error daisyConvol: set LIBDAISY_DIR to a built libDaisy checkout before including daisyconvol.mk)
endif
ifeq ($(wildcard $(LIBDAISY_DIR)/core/Makefile),)
$(error daisyConvol: no libDaisy at LIBDAISY_DIR = '$(LIBDAISY_DIR)' ($(LIBDAISY_DIR)/core/Makefile not found))
endif

# --- Options ----------------------------------------------------------------

DAISYCONVOL_BLOCKS ?= 32
DAISYCONVOL_MAX_IR ?=

# FFT sizes of ConvEngine::MakeGeometry(B, any length) - 2 x each partition.
DAISYCONVOL_FFTS_16   = 32 256 4096
DAISYCONVOL_FFTS_32   = 64 512 4096
DAISYCONVOL_FFTS_64   = 128 1024 4096
DAISYCONVOL_FFTS_128  = 256 2048 4096
DAISYCONVOL_FFTS_256  = 512 4096
DAISYCONVOL_FFTS_512  = 1024 4096
DAISYCONVOL_FFTS_1024 = 2048 4096
DAISYCONVOL_FFTS_2048 = 4096
# ...and of MakeGeometry(B, <= 4096) (test/conv.cpp checks both).
DAISYCONVOL_FFTS_CAB_16   = 32 256
DAISYCONVOL_FFTS_CAB_32   = 64 512
DAISYCONVOL_FFTS_CAB_64   = 128 1024
DAISYCONVOL_FFTS_CAB_128  = 256 2048
DAISYCONVOL_FFTS_CAB_256  = 512
DAISYCONVOL_FFTS_CAB_512  = 1024
DAISYCONVOL_FFTS_CAB_1024 = 2048
DAISYCONVOL_FFTS_CAB_2048 = 4096

# $(call daisyconvol_le4096,N): 1 if the decimal number N is <= 4096. Pure
# make (no $(shell), which the Windows toolchain may not have): numbers of
# up to 3 digits are, 4-digit ones compare as strings, longer ones are not.
daisyconvol_digits = $(strip $(subst 0,0 ,$(subst 1,1 ,$(subst 2,2 ,$(subst 3,3 ,$(subst 4,4 ,$(subst 5,5 ,$(subst 6,6 ,$(subst 7,7 ,$(subst 8,8 ,$(subst 9,9 ,$(1))))))))))))
daisyconvol_le4096 = $(if $(word 5,$(call daisyconvol_digits,$(1))),,$(if $(word 4,$(call daisyconvol_digits,$(1))),$(if $(filter $(1),$(firstword $(sort $(1) 4096))),1),1))

ifneq ($(and $(strip $(DAISYCONVOL_MAX_IR)),$(call daisyconvol_le4096,$(strip $(DAISYCONVOL_MAX_IR)))),)
DAISYCONVOL_FFT_SET = CAB_
endif
DAISYCONVOL_FFT_SIZES ?= $(sort $(foreach b,$(DAISYCONVOL_BLOCKS),$(DAISYCONVOL_FFTS_$(DAISYCONVOL_FFT_SET)$(b))))
ifeq ($(strip $(DAISYCONVOL_FFT_SIZES)),)
$(error daisyConvol: unsupported DAISYCONVOL_BLOCKS '$(DAISYCONVOL_BLOCKS)' (powers of two, 16..2048))
endif
C_DEFS += $(foreach n,$(DAISYCONVOL_FFT_SIZES),-DDAISYCONVOL_FFT_$(n)=1)

# --- Sources ----------------------------------------------------------------

CPP_SOURCES += \
$(DAISYCONVOL_DIR)/src/daisyconvol/conv_engine.cpp \
$(DAISYCONVOL_DIR)/src/daisyconvol_daisy/fft_backend_cmsis.cpp

# CMSIS-DSP real FFT. libdaisy.a does not include the DSP sources, so the FFT
# kernels are compiled here; cmsis_bitreversal.c is arm_bitreversal_32 in C
# (the CMSIS bundled with libDaisy before v7 only ships it as a .S the core
# Makefile does not assemble). Unused twiddle tables are dropped by
# --gc-sections. libDaisy v7.0.0 and later have CMSIS-DSP as a submodule in
# Drivers/CMSIS-DSP; earlier versions bundle it in Drivers/CMSIS/DSP.
ifneq ($(wildcard $(LIBDAISY_DIR)/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_rfft_fast_f32.c),)
DAISYCONVOL_CMSIS_DIR = $(LIBDAISY_DIR)/Drivers/CMSIS-DSP/Source
else ifneq ($(wildcard $(LIBDAISY_DIR)/Drivers/CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c),)
DAISYCONVOL_CMSIS_DIR = $(LIBDAISY_DIR)/Drivers/CMSIS/DSP/Source
else
$(error daisyConvol: CMSIS-DSP not found under $(LIBDAISY_DIR)/Drivers (libDaisy v7+: git submodule update --init --recursive))
endif
C_SOURCES += \
$(DAISYCONVOL_CMSIS_DIR)/TransformFunctions/arm_cfft_f32.c \
$(DAISYCONVOL_CMSIS_DIR)/TransformFunctions/arm_cfft_radix8_f32.c \
$(DAISYCONVOL_CMSIS_DIR)/TransformFunctions/arm_rfft_fast_f32.c \
$(DAISYCONVOL_CMSIS_DIR)/CommonTables/arm_common_tables.c \
$(DAISYCONVOL_DIR)/src/daisyconvol_daisy/cmsis_bitreversal.c

C_INCLUDES += -I$(DAISYCONVOL_DIR)/src

# --- libDaisy ---------------------------------------------------------------

SYSTEM_FILES_DIR ?= $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# --- Rebuild on option changes ---------------------------------------------

DAISYCONVOL_STAMP = $(BUILD_DIR)/daisyconvol.flags
DAISYCONVOL_FLAGS = $(strip $(C_DEFS) $(OPT) $(CPP_STANDARD) $(C_STANDARD))
ifneq ($(file < $(DAISYCONVOL_STAMP)),$(DAISYCONVOL_FLAGS))
.PHONY: $(DAISYCONVOL_STAMP)
endif
$(DAISYCONVOL_STAMP): | $(BUILD_DIR)
	$(file > $@,$(DAISYCONVOL_FLAGS))

$(OBJECTS): $(DAISYCONVOL_STAMP) $(DAISYCONVOL_DIR)/daisyconvol.mk
