#include "cpu_identification.h"
#include "../image_function.h"

#ifdef PENGUINV_AVX_SET
#include "../image_function_avx.h"
#endif

#ifdef PENGUINV_SSE_SET
#include "../image_function_sse.h"
#endif

#ifdef PENGUINV_NEON_SET
#include "../image_function_neon.h"
#endif

#include "penguinv.h"

// We directly make first call to initialize function table
// to prevent multithreading issues
static const penguinV::FunctionTable& functionTable = penguinV::functionTable();

namespace
{
    penguinV::FunctionTable initialize()
    {
        penguinV::FunctionTable table;

        // A list of basic functions
        table.AbsoluteDifference = &Image_Function::AbsoluteDifference;
        table.Accumulate         = &Image_Function::Accumulate;
        table.BitwiseAnd         = &Image_Function::BitwiseAnd;
        table.BitwiseOr          = &Image_Function::BitwiseOr;
        table.BitwiseXor         = &Image_Function::BitwiseXor;
        table.ConvertToGrayScale = &Image_Function::ConvertToGrayScale;
        table.ConvertToRgb       = &Image_Function::ConvertToRgb;
        table.Copy               = &Image_Function::Copy;
        table.ExtractChannel     = &Image_Function::ExtractChannel;
        table.Fill               = &Image_Function::Fill;
        table.Flip               = &Image_Function::Flip;
        table.GammaCorrection    = &Image_Function::GammaCorrection;
        table.GetPixel           = &Image_Function::GetPixel;
        table.GetThreshold       = &Image_Function::GetThreshold;
        table.Histogram          = &Image_Function::Histogram;
        table.Invert             = &Image_Function::Invert;
        table.IsEqual            = &Image_Function::IsEqual;
        table.LookupTable        = &Image_Function::LookupTable;
        table.Maximum            = &Image_Function::Maximum;
        table.Merge              = &Image_Function::Merge;
        table.Minimum            = &Image_Function::Minimum;
        table.Normalize          = &Image_Function::Normalize;
        table.ProjectionProfile  = &Image_Function::ProjectionProfile;
        table.Resize             = &Image_Function::Resize;
        table.RgbToBgr           = &Image_Function::RgbToBgr;
        table.SetPixel           = &Image_Function::SetPixel;
        table.SetPixel2          = &Image_Function::SetPixel;
        table.Split              = &Image_Function::Split;
        table.Subtract           = &Image_Function::Subtract;
        table.Sum                = &Image_Function::Sum;
        table.Threshold          = &Image_Function::Threshold;
        table.Threshold2         = &Image_Function::Threshold;
        table.Transpose          = &Image_Function::Transpose;

#ifdef PENGUINV_SSE_SET
        if( isSseAvailable ) {
            table.AbsoluteDifference = &Image_Function_Sse::AbsoluteDifference;
            table.BitwiseAnd         = &Image_Function_Sse::BitwiseAnd;
            table.BitwiseOr          = &Image_Function_Sse::BitwiseOr;
            table.BitwiseXor         = &Image_Function_Sse::BitwiseXor;
            table.Invert             = &Image_Function_Sse::Invert;
            table.Maximum            = &Image_Function_Sse::Maximum;
            table.Minimum            = &Image_Function_Sse::Minimum;
            table.Subtract           = &Image_Function_Sse::Subtract;
            table.Sum                = &Image_Function_Sse::Sum;
            table.Threshold          = &Image_Function_Sse::Threshold;
            table.Threshold2         = &Image_Function_Sse::Threshold;
        }
#endif

#ifdef PENGUINV_AVX_SET
        if( isAvxAvailable ) {
            table.AbsoluteDifference = &Image_Function_Avx::AbsoluteDifference;
            table.BitwiseAnd         = &Image_Function_Avx::BitwiseAnd;
            table.BitwiseOr          = &Image_Function_Avx::BitwiseOr;
            table.BitwiseXor         = &Image_Function_Avx::BitwiseXor;
            table.Invert             = &Image_Function_Avx::Invert;
            table.Maximum            = &Image_Function_Avx::Maximum;
            table.Minimum            = &Image_Function_Avx::Minimum;
            table.Subtract           = &Image_Function_Avx::Subtract;
            table.Sum                = &Image_Function_Avx::Sum;
            table.Threshold          = &Image_Function_Avx::Threshold;
            table.Threshold2         = &Image_Function_Avx::Threshold;
        }
#endif

#ifdef PENGUINV_NEON_SET
        if( isNeonAvailable ) {
            table.AbsoluteDifference = &Image_Function_Neon::AbsoluteDifference;
            table.BitwiseAnd         = &Image_Function_Neon::BitwiseAnd;
            table.BitwiseOr          = &Image_Function_Neon::BitwiseOr;
            table.BitwiseXor         = &Image_Function_Neon::BitwiseXor;
            table.Invert             = &Image_Function_Neon::Invert;
            table.Maximum            = &Image_Function_Neon::Maximum;
            table.Minimum            = &Image_Function_Neon::Minimum;
            table.Subtract           = &Image_Function_Neon::Subtract;
            table.Threshold          = &Image_Function_Neon::Threshold;
            table.Threshold2         = &Image_Function_Neon::Threshold;
        }
#endif
        return table;
    }
};

namespace penguinV
{
    const FunctionTable & functionTable()
    {
        static FunctionTable table = initialize();
        return table;
    }
};
