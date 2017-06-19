#pragma once

// This file is very important to setup penguinV library to have the best performance
// Please be very carefull in setting these parameters!!!

#ifdef _WIN32 // Windows

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
#elif __APPLE__ // MacOS

#include "cpu_id_macos.h"

    #ifdef __arm__
        #define PENGUINV_NEON_SET
    #else

    #endif

#elif linux // Linux

    #ifdef __arm__
        #define PENGUINV_NEON_SET
    #else

    #endif

#include "cpu_id_linux.h"

#else
	#error "Unknown platform"
#endif

// Identify available technologies during runtime
#ifdef PENGUINV_AVX_SET
static const bool isAvxAvailable  = isAvxSupported();
#endif
#ifdef PENGUINV_SSE_SET
static const bool isSseAvailable  = isSseSupported();
#endif
#ifdef PENGUINV_NEON_SET
static const bool isNeonAvailable = isNeonSupported();
#endif
