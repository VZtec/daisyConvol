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

# make ITCM=1: the engine and the CMSIS FFT run from the ITCM (64 KB, zero
# wait states) instead of from QSPI through the 16 KB I-cache, as an app
# with other heavy DSP would place them; code layout then does not change
# their cost. The linker script is libDaisy's with an .itcmram_text section
# added, and cabsim.cpp copies the code there at boot.
ifeq ($(ITCM),1)
C_DEFS += -DCABSIM_ITCM
LDSCRIPT = $(BUILD_DIR)/cabsim_itcm.lds
endif

include ../../daisyconvol.mk

ifeq ($(ITCM),1)
CABSIM_ITCM_OBJS = conv_engine fft_backend_cmsis arm_cfft_f32 \
                   arm_cfft_radix8_f32 arm_rfft_fast_f32 cmsis_bitreversal
$(LDSCRIPT): $(SYSTEM_FILES_DIR)/STM32H750IB_qspi.lds Makefile | $(BUILD_DIR)
	awk -v objs='$(CABSIM_ITCM_OBJS)' \
	    'BEGIN { n = split(objs, o, " "); \
	             s = "\t.itcmram_text :\n\t{\n\t\t. = ALIGN(8);\n\t\t. = . + 8;\n\t\t_itcmram_text_start = .;\n"; \
	             for (i = 1; i <= n; i++) s = s "\t\t*" o[i] ".o(.text .text*)\n"; \
	             s = s "\t\t. = ALIGN(4);\n\t\t_itcmram_text_end = .;\n\t} > ITCMRAM AT > QSPIFLASH\n"; \
	             s = s "\t_sitcmram_text = LOADADDR(.itcmram_text) + (_itcmram_text_start - ADDR(.itcmram_text));\n\n" } \
	     !done && /^[ \t]*\.text[ \t]*:/ { printf "%s", s; done = 1 } \
	     { print } \
	     END { if (!done) { print "no .text section found" > "/dev/stderr"; exit 1 } }' \
	    $< > $@
$(BUILD_DIR)/$(TARGET).elf: $(LDSCRIPT)
endif
