#pragma once

struct CpuInformation
{
    static bool isSseSupported()
    {
        return false;
    }
    
    static bool isAvxSupported()
    {
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
