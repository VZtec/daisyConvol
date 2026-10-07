/* cmsis_bitreversal.c - arm_bitreversal_32 in C, required by
 * arm_cfft_f32.c.
 *
 * The CMSIS-DSP bundled with libDaisy before v7.0.0 only ships this routine
 * as ARM assembly (arm_bitreversal2.S), which libDaisy's core Makefile does
 * not assemble (.S vs .s). This is the C version from later CMSIS-DSP
 * releases (Source/TransformFunctions/arm_bitreversal2.c), reduced to the
 * 32-bit function and reformatted; the algorithm is unchanged. Also built
 * with libDaisy v7.0.0 and later, whose arm_bitreversal2.c is not compiled.
 *
 * Original work: CMSIS DSP Library, arm_bitreversal2.c, V1.9.0.
 * Copyright (C) 2019 ARM Limited or its affiliates. All rights reserved.
 * Modifications: copyright (c) 2026 VZtec.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0 (also LICENSE-Apache-2.0.txt here)
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>

void arm_bitreversal_32(uint32_t*       pSrc,
                        const uint16_t  bitRevLen,
                        const uint16_t* pBitRevTab)
{
    uint32_t a, b, i, tmp;

    for(i = 0; i < bitRevLen; i += 2)
    {
        a = pBitRevTab[i] >> 2;
        b = pBitRevTab[i + 1] >> 2;

        /* real */
        tmp      = pSrc[a];
        pSrc[a]  = pSrc[b];
        pSrc[b]  = tmp;

        /* complex */
        tmp         = pSrc[a + 1];
        pSrc[a + 1] = pSrc[b + 1];
        pSrc[b + 1] = tmp;
    }
}
