#pragma once

struct CpuInformation
{
    bool isSseSupported()
    {
        return false;
    }
    
    bool isAvxSupported()
    {
        return false;
    }
    
    bool isNeonSupported()
    {
        #ifdef __arm__
            return true;
        #else
            return false;
        #endif
    }
};
