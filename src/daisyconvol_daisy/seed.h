// seed.h - Daisy Seed glue for daisyConvol: where the engine's memory goes,
// and the CPU settings it is measured with. Needs libDaisy's include paths
// (the STM32H7 CMSIS device header); daisyconvol.mk sets them up.
//
//   #include "daisyconvol/daisyconvol.h"
//   #include "daisyconvol_daisy/seed.h"
//
//   DAISYCONVOL_DTCM  alignas(32) float fast_mem[CE::FastFloats(cfg)];
//   alignas(32)                   float medium_mem[CE::MediumFloats(cfg)];
//   DAISYCONVOL_SDRAM alignas(32) float slow_mem[CE::SlowFloats(cfg)];

#pragma once

#include <cstddef>
#include <cstdint>

#include "stm32h7xx.h"

// Static storage in the DTCM (128 KB, zero wait states, never cached): the
// fast arena, i.e. the state every callback touches. Past B = 64 the fast
// arena outgrows it next to the stack: put it in AXI SRAM (plain .bss).
#define DAISYCONVOL_DTCM __attribute__((section(".dtcmram_bss")))
// Static storage in the SDRAM (64 MB): the slow arena, i.e. the main-loop
// stage of long IRs. Same section as libDaisy's DSY_SDRAM_BSS.
#define DAISYCONVOL_SDRAM __attribute__((section(".sdram_bss")))
// Both sections are NOLOAD in libDaisy's linker scripts (the startup code
// does not clear them); ConvEngine::Init() zeroes the arenas it uses.
// Plain .bss (the medium arena) is AXI SRAM (512 KB).

namespace daisyconvol_daisy
{

// Size of an arena's backing array: never 0, since zero-length arrays are
// not C++. Pass the engine the real size (FastFloats() etc., 0 = unused).
constexpr size_t StorageFloats(size_t n)
{
    return n ? n : 1;
}

// Flush denormals to zero, in the main loop and in interrupt handlers: a
// handler (the audio callback) starts with FPSCR loaded from FPDSCR, so
// both are set. Reverb tails decay into denormals, which the Cortex-M7 FPU
// processes far more slowly. Call once after DaisySeed::Init().
inline void EnableFlushToZero()
{
    FPU->FPDSCR |= FPU_FPDSCR_FZ_Msk;
    __set_FPSCR(__get_FPSCR() | FPU_FPDSCR_FZ_Msk);
    __DSB();
}

// Starts the DWT cycle counter, for ConvEngine::Config::get_cycles
// (= CycleCount) and the cycle counts in ConvEngine::perf().
inline void EnableCycleCounter()
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR    = 0xC5ACCE55; // unlock the DWT on the Cortex-M7
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

inline uint32_t CycleCount()
{
    return DWT->CYCCNT;
}

} // namespace daisyconvol_daisy
