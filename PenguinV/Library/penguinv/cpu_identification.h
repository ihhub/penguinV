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

    #include "TargetConditionals.h"
    #if TARGET_IPHONE_SIMULATOR
         // iOS Simulator
    #elif TARGET_OS_IPHONE
        // iOS device
    #elif TARGET_OS_MAC
        // Other kinds of Mac OS
    #else
    #   error "Unknown Apple platform"
    #endif
#else // Linux or something else?

#include "cpu_id_linux.h"

#endif

// Identify available technologies during runtime
static const bool isAvxAvailable  = isAvxSupported();
static const bool isSseAvailable  = isSseSupported();
static const bool isNeonAvailable = isNeonSupported();
