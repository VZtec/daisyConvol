/* core_cm0.h - host-build shim for the desktop tests (not for the target).
 *
 * Lets CMSIS-DSP's arm_math.h, and with it the target FFT sources
 * (arm_cfft_f32.c, arm_rfft_fast_f32.c, ...), compile on a PC: build them
 * with ARM_MATH_CM0 (portable C paths only) and this directory first on the
 * include path, instead of the real CMSIS core headers. With MSVC also
 * define __CSMC__, which selects plain C in arm_math.h's compiler switch.
 */
#ifndef DAISYCONVOL_HOST_CORE_CM0_H
#define DAISYCONVOL_HOST_CORE_CM0_H

#include <stdint.h>

#if defined(_MSC_VER)
#define __STATIC_INLINE static __inline
#define __INLINE __inline
#else
#define __STATIC_INLINE static inline
#define __INLINE inline
#endif

/* Count leading zeros (used by some q31/q15 helpers in arm_math.h). */
__STATIC_INLINE uint32_t __CLZ(uint32_t x)
{
    uint32_t n = 0;
    if(x == 0)
        return 32;
    while(!(x & 0x80000000u))
    {
        x <<= 1;
        n++;
    }
    return n;
}

#endif
