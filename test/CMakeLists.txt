# Host test suites for daisyConvol (any C++14 compiler: MSVC, GCC, Clang).
# test/all.sh runs them (and the firmware builds); by hand:
#
#   cmake -S test -B build/test
#   cmake --build build/test --config Release
#   ctest --test-dir build/test -C Release --output-on-failure
#
# test_direct compares the engine with a direct time-domain convolution;
# test_conv is the full suite (stage boundaries, outputs, IR switching,
# resets, main-loop stalls, counter wraps...). Both run on the portable
# double-precision reference FFT. With -DLIBDAISY_DIR=path/to/libDaisy,
# test_direct_cmsis and test_conv_cmsis run them again on the target's own
# CMSIS-DSP float32 FFT, compiled for the host (MSVC or GCC: arm_math.h has
# pointer-to-int32 casts that Clang rejects in C++).
# -DDAISYCONVOL_SANITIZE=ON adds AddressSanitizer (+ UBSan outside MSVC).

cmake_minimum_required(VERSION 3.16)
project(daisyconvol_tests C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(DAISYCONVOL_SANITIZE "Build with AddressSanitizer (+UBSan)" OFF)
set(LIBDAISY_DIR "" CACHE PATH
    "libDaisy checkout: also test the CMSIS-DSP FFT backend")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)

function(daisyconvol_test_flags target)
    target_include_directories(${target} PRIVATE ${SRC})
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /EHsc)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion)
    endif()
    if(DAISYCONVOL_SANITIZE)
        if(MSVC)
            target_compile_options(${target} PRIVATE /fsanitize=address)
        else()
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined -fno-omit-frame-pointer
                -fno-sanitize-recover=all)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

enable_testing()

foreach(suite direct conv)
    add_executable(test_${suite}
        ${suite}.cpp ${SRC}/daisyconvol/conv_engine.cpp
        ${SRC}/daisyconvol/fft_backend_ref.cpp)
    daisyconvol_test_flags(test_${suite})
    add_test(NAME test_${suite} COMMAND test_${suite})
endforeach()

if(LIBDAISY_DIR)
    # libDaisy v7.0.0 and later: CMSIS-DSP submodule in Drivers/CMSIS-DSP;
    # earlier: bundled in Drivers/CMSIS/DSP (see daisyconvol.mk).
    if(EXISTS ${LIBDAISY_DIR}/Drivers/CMSIS-DSP/Include/arm_math.h)
        set(DSP ${LIBDAISY_DIR}/Drivers/CMSIS-DSP)
        set(DSP_SUBMODULE ON)
    elseif(EXISTS ${LIBDAISY_DIR}/Drivers/CMSIS/DSP/Include/arm_math.h)
        set(DSP ${LIBDAISY_DIR}/Drivers/CMSIS/DSP)
        set(DSP_SUBMODULE OFF)
    else()
        message(FATAL_ERROR "LIBDAISY_DIR: no CMSIS-DSP (arm_math.h) under "
                            "${LIBDAISY_DIR}/Drivers")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
        message(WARNING "test_conv_cmsis needs MSVC or GCC (skipped)")
    else()
        # The FFT sources as the target builds them, on their portable C
        # paths: the bundled CMSIS needs a stand-in for its core header
        # (host_cmsis/core_cm0.h); the submodule one supports MSVC as is and
        # GCC through its desktop (Python wrapper) switch.
        add_library(cmsis_fft_host STATIC
            ${DSP}/Source/TransformFunctions/arm_cfft_f32.c
            ${DSP}/Source/TransformFunctions/arm_cfft_radix8_f32.c
            ${DSP}/Source/TransformFunctions/arm_rfft_fast_f32.c
            ${DSP}/Source/CommonTables/arm_common_tables.c
            ${SRC}/daisyconvol_daisy/cmsis_bitreversal.c)
        if(DSP_SUBMODULE)
            set(defs "")
            if(NOT MSVC)
                list(APPEND defs __GNUC_PYTHON__)
            endif()
        else()
            set(defs ARM_MATH_CM0)
            if(MSVC)
                list(APPEND defs __CSMC__) # plain C in arm_math.h's compiler list
            endif()
        endif()
        foreach(n 32 64 128 256 512 1024 2048 4096)
            list(APPEND defs DAISYCONVOL_FFT_${n}=1)
        endforeach()
        target_compile_definitions(cmsis_fft_host PUBLIC ${defs})
        target_include_directories(cmsis_fft_host SYSTEM PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/host_cmsis ${DSP}/Include)
        if(MSVC)
            target_compile_options(cmsis_fft_host PRIVATE /W0)
        else()
            target_compile_options(cmsis_fft_host PRIVATE -w)
        endif()

        foreach(suite direct conv)
            add_executable(test_${suite}_cmsis
                ${suite}.cpp ${SRC}/daisyconvol/conv_engine.cpp
                ${SRC}/daisyconvol_daisy/fft_backend_cmsis.cpp)
            daisyconvol_test_flags(test_${suite}_cmsis)
            target_link_libraries(test_${suite}_cmsis PRIVATE cmsis_fft_host)
            # arm_math.h casts pointers to int32_t (fine on the 32-bit
            # target).
            if(MSVC)
                target_compile_options(test_${suite}_cmsis PRIVATE
                    /wd4311 /wd4302 /wd4312)
            else()
                target_compile_options(test_${suite}_cmsis PRIVATE -fpermissive)
            endif()
            add_test(NAME test_${suite}_cmsis COMMAND test_${suite}_cmsis)
        endforeach()
    endif()
endif()
