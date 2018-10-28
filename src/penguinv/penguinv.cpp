#include "penguinv.h"

#include <map>
#include <mutex>

namespace
{
    std::map< uint8_t, penguinV::FunctionTable > functionTableMap;

    template <typename _Function>
    void setFunction( _Function F1, _Function F2, bool forceSetup )
    {
        if ( (F1 == nullptr) || (forceSetup && (F2 != nullptr)) )
        {
            F1 = F2;
        }
    }

#define SET_FUNCTION( functionName ) \
    setFunction( oldTable.functionName, newTable.functionName, forceSetup );

    void setupTable( penguinV::FunctionTable & oldTable, const penguinV::FunctionTable & newTable, bool forceSetup )
    {
        SET_FUNCTION(AbsoluteDifference)
        SET_FUNCTION(Accumulate)
        SET_FUNCTION(BitwiseAnd)
        SET_FUNCTION(BitwiseOr)
        SET_FUNCTION(BitwiseXor)
        SET_FUNCTION(ConvertToGrayScale)
        SET_FUNCTION(ConvertToRgb)
        SET_FUNCTION(Copy)
        SET_FUNCTION(ExtractChannel)
        SET_FUNCTION(Fill)
        SET_FUNCTION(Flip)
        SET_FUNCTION(GammaCorrection)
        SET_FUNCTION(GetPixel)
        SET_FUNCTION(Histogram)
        SET_FUNCTION(Invert)
        SET_FUNCTION(IsEqual)
        SET_FUNCTION(LookupTable)
        SET_FUNCTION(Maximum)
        SET_FUNCTION(Merge)
        SET_FUNCTION(Minimum)
        SET_FUNCTION(Normalize)
        SET_FUNCTION(ProjectionProfile)
        SET_FUNCTION(Resize)
        SET_FUNCTION(RgbToBgr)
        SET_FUNCTION(SetPixel)
        SET_FUNCTION(SetPixel2)
        SET_FUNCTION(Split)
        SET_FUNCTION(Subtract)
        SET_FUNCTION(Sum)
        SET_FUNCTION(Threshold)
        SET_FUNCTION(Threshold2)
        SET_FUNCTION(Transpose)
    }
}

namespace penguinV
{
    void registerFunctionTable( const Image & image, const FunctionTable & table, bool forceSetup )
    {
        static std::mutex mapGuard;

        mapGuard.lock();
        std::map< uint8_t, penguinV::FunctionTable >::iterator oldTable = functionTableMap.find( image.type() );
        if (oldTable != functionTableMap.end())
            setupTable( oldTable->second, table, forceSetup );
        else
            functionTableMap[image.type()] = table;
        mapGuard.unlock();
    }

    const FunctionTable & functionTable( const Image & image )
    {
        std::map< uint8_t, penguinV::FunctionTable >::const_iterator table = functionTableMap.find( image.type() );
        if (table == functionTableMap.end())
            throw imageException( "Function table is not initialised" );

        return table->second;
    }
}
