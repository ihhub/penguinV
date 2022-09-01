#pragma once

#if !defined( __arm__ ) && !defined( __aarch64__ )
#include <cpuid.h>
#endif

struct CpuInformation
{
    static bool isSseSupported()
    {
#if !defined( __arm__ ) && !defined( __aarch64__ )
        int info[4];
        __cpuid_count( 0, 0, info[0], info[1], info[2], info[3] );
        const int id = info[0];

        if ( id >= 0x00000001 ) {
            __cpuid_count( 0x00000001, 0, info[0], info[1], info[2], info[3] );
            return ( info[3] & ( (int)1 << 26 ) ) != 0;
        }
#endif
        return false;
    }

    static bool isAvxSupported()
    {
#if !defined( __arm__ ) && !defined( __aarch64__ )
        int info[4];
        __cpuid_count( 0, 0, info[0], info[1], info[2], info[3] );
        const int id = info[0];

        if ( id >= 0x00000007 ) {
            __cpuid_count( 0x00000007, 0, info[0], info[1], info[2], info[3] );
            return ( info[1] & ( (int)1 << 5 ) ) != 0;
        }
#endif
        return false;
    }

    static bool isAvx512SKLSupported()
    {
#if !defined( __arm__ ) && !defined( __aarch64__ )
        int info[4];
        __cpuid_count( 0, 0, info[0], info[1], info[2], info[3] );
        const int id = info[0];

        if ( id >= 0x00000007 ) {
            __cpuid_count( 0x00000007, 0, info[0], info[1], info[2], info[3] );
            return ( info[1] & ( ( (int)1 << 31 ) | ( (int)1 << 30 ) | ( (int)1 << 28 ) | ( (int)1 << 17 ) | ( (int)1 << 16 ) ) ) != 0;
        }
#endif
        return false;
    }

    static bool isNeonSupported()
    {
#if !defined( __arm__ ) && !defined( __aarch64__ )
        return false;
#else
        return true;
#endif
    }
};
