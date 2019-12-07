#pragma once

// This file is very important to setup penguinV library to have the best performance
// Please be very carefull in setting these parameters!!!

#if defined (_MSC_VER) // Visual Studio

#include "cpu_id_windows.h"

    #ifdef _M_ARM
        #define PENGUINV_NEON_SET
    #else
        #define PENGUINV_SSE_SET
        #define PENGUINV_SSSE3_SET

        #ifdef _MSC_VER
            #if _MSC_VER >= 1700
                #define PENGUINV_AVX_SET
            #elif _MSC_VER >= 1911
                #define PENGUINV_AVX_SET
            #endif
        #endif
    #endif
#elif defined(__APPLE__) || defined(__linux__) || defined (__MINGW32__) // MacOS, Linux or MinGW

#include "cpu_id_unix.h"

    #if defined(__arm__) || defined(__aarch64__)
        #define PENGUINV_NEON_SET
    #elif __SSE2__
        #define PENGUINV_SSE_SET

        #ifdef __SSSE3__
            #define PENGUINV_SSSE3_SET
        #endif

        #ifdef __AVX2__
            #define PENGUINV_AVX_SET
        #endif

        #if defined(__AVX512BW__) && defined(__AVX512CD__) && defined(__AVX512DQ__) && defined(__AVX512F__) && defined(__AVX512VL__)
            #define PENGUINV_AVX512BW_SET
        #endif
    #endif

#else
    #error "Unknown platform. Is your OS Windows or OSX or Linux?"
#endif

// We verify SIMD instruction set definition
#ifdef PENGUINV_NEON_SET

    #ifdef PENGUINV_SSE_SET
        #error "NEON and SSE cannot be supported on a single CPU. Please check SIMD instruction set verification code"
    #endif

    #ifdef PENGUINV_AVX_SET
        #error "NEON and AVX cannot be supported on a single CPU. Please check SIMD instruction set verification code"
    #endif

#endif

#ifdef PENGUINV_AVX_SET
    #ifndef PENGUINV_SSE_SET
        #error "None of existing processors can support AVX but not SSE. Please check SIMD instruction set verification code"
    #endif
#endif

#ifdef PENGUINV_AVX512BW_SET
    #ifndef PENGUINV_AVX_SET
        #error "None of existing processors can support AVX512 but not AVX. Please check SIMD instruction set verification code"
    #endif
#endif

#ifdef PENGUINV_SSSE3_SET
    #ifndef PENGUINV_SSE_SET
        #error "SSSE3 technology cannot exist without SSE. Please check SIMD instruction set verification code"
    #endif
#endif

// Identify available technologies during runtime
struct SimdInfo
{
    static bool isSseAvailable()
    {
#ifdef PENGUINV_SSE_SET
        static const bool isAvailable = CpuInformation::isSseSupported();
        return isAvailable;
#else
        return false;
#endif
    }

    static bool isAvxAvailable()
    {
#ifdef PENGUINV_AVX_SET
        static const bool isAvailable = CpuInformation::isAvxSupported();
        return isAvailable;
#else
        return false;
#endif
    }

    static bool isNeonAvailable()
    {
#ifdef PENGUINV_NEON_SET
        static const bool isAvailable = CpuInformation::isNeonSupported();
        return isAvailable;
#else
        return false;
#endif
    }

    static bool isAVX512BWAvailable()
    {
#ifdef PENGUINV_AVX512BW_SET
        static const bool isAvailable = CpuInformation::isAvx512BWSupported();
        return isAvailable;
#else
        return false;
#endif
    }
};
