#pragma once

// This file is very important to setup penguinV library to have the best performance
// Please be very carefull in setting these parameters!!!

// 1) First of all we need to setup proper CPU architecture what you are compiling on
//    If you do not know what type of processor you are compiling for then just comment below lines
//    Otherwise comment all lines except one what you are compiling for

//#define PENGUINV_INTEL_AMD // Mainly for desktops, laptops, servers
//#define PENGUINV_ARM       // Mainly for mobiles, IoT devices


// 2) Enable possible supported instruction sets on diffirent architectures
//    This part of code should not be changed expect some specific situations in below comments
#if defined(PENGUINV_INTEL_AMD)
	#define PENGUINV_AVX_SET // old Visual Studio (2010 and older) does not have such instruction so we recommend to comment this line
	#define PENGUINV_SSE_SET
#elif defined(PENGUINV_ARM)
	#define PENGUINV_NEON_SET
#else
	
#endif


// 3) Here actually should be your code for identification of supported instruction set
//    You can use third-party libraries, setup manually or something else
//    So we are giving an identification step in your hands
static const bool isAvxAvailable  = false;
static const bool isSseAvailable  = false;
static const bool isNeonAvailable = false;
