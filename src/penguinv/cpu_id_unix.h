#pragma once


#ifndef __arm__
#include <cpuid.h>
#endif

struct CpuInformation
{
    static bool isSseSupported()
    {
#ifndef __arm__
        int info[4];
        __cpuid_count( 0, 0, info[0], info[1], info[2], info[3] );
        const int id = info[0];

        if( id >= 0x00000001 ) {
            __cpuid_count( 0x00000001, 0, info[0], info[1], info[2], info[3] );
            return (info[3] & ((int)1 << 26)) != 0;
        }
#endif
        return false;
    }

    static bool isAvxSupported()
    {
#ifndef __arm__
        int info[4];
        __cpuid_count( 0, 0, info[0], info[1], info[2], info[3] );
        const int id = info[0];

        if( id >= 0x00000007 ) {
            __cpuid_count( 0x00000007, 0, info[0], info[1], info[2], info[3] );
            return (info[1] & ((int)1 << 5)) != 0;
        }
#endif
        return false;
    }

    static bool isNeonSupported()
    {
#ifdef __arm__
        return true;
#else
        return false;
#endif
    }
};
