#pragma once

// This file is very important to setup penguinV library to have the best performance
// Please be very carefull in setting these parameters!!!

#if defined (_MSC_VER) // Visual Studio

#include "cpu_id_windows.h"

    #ifdef _M_ARM
        #define PENGUINV_NEON_SET
    #else
        #define PENGUINV_SSE_SET

        #ifdef _MSC_VER
            #if _MSC_VER >= 1700
                #define PENGUINV_AVX_SET
            #endif
        #endif
    #endif
#elif defined(__APPLE__) || defined(__linux__) || defined (__MINGW32__) // MacOS, Linux or MinGW

#include "cpu_id_unix.h"

    #ifdef __arm__
        #define PENGUINV_NEON_SET
    #elif __SSE2__
        #define PENGUINV_SSE_SET

        #ifdef __AVX2__
            #define PENGUINV_AVX_SET
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

// Identify available technologies during runtime
#ifdef PENGUINV_SSE_SET
static const bool isSseAvailable  = CpuInformation::isSseSupported();
#else
static const bool isSseAvailable  = false;
#endif

#ifdef PENGUINV_AVX_SET
static const bool isAvxAvailable  = CpuInformation::isAvxSupported();
#else
static const bool isAvxAvailable  = false;
#endif

#ifdef PENGUINV_NEON_SET
static const bool isNeonAvailable = CpuInformation::isNeonSupported();
#else
static const bool isNeonAvailable = false;
#endif
