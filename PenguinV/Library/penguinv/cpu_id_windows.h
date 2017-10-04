#pragma once

#include <intrin.h>

struct CpuInformation
{
    static bool isSseSupported()
    {
        int info[4];
        __cpuidex( info, 0, 0 );
        int nIds = info[0];
    
        if( nIds >= 0x00000001 ) {
            __cpuidex( info, 0x00000001, 0 );
            return (info[3] & ((int)1 << 26)) != 0;
        }
    
        return false;
    }
    
    static bool isAvxSupported()
    {
        int info[4];
        __cpuidex( info, 0, 0 );
        int nIds = info[0];
    
        if( nIds >= 0x00000007 ) {
            __cpuidex( info, 0x00000007, 0 );
            return (info[1] & ((int)1 <<  5)) != 0;
        }
    
        return false;
    }
    
    static bool isNeonSupported()
    {
        #ifdef _M_ARM
            return true;
        #else
            return false;
        #endif
    }
};
