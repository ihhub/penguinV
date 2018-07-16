#include "penguinv.h"

#include "cpu_identification.h"
#include "../image_function.h"
#include "../image_function_simd.h"

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

        // SIMD
        table.AbsoluteDifference = &Image_Function_Simd::AbsoluteDifference;
        table.BitwiseAnd         = &Image_Function_Simd::BitwiseAnd;
        table.BitwiseOr          = &Image_Function_Simd::BitwiseOr;
        table.BitwiseXor         = &Image_Function_Simd::BitwiseXor;
        table.Invert             = &Image_Function_Simd::Invert;
        table.Maximum            = &Image_Function_Simd::Maximum;
        table.Minimum            = &Image_Function_Simd::Minimum;
        table.Subtract           = &Image_Function_Simd::Subtract;
        table.Sum                = &Image_Function_Simd::Sum;
        table.Threshold          = &Image_Function_Simd::Threshold;
        table.Threshold2         = &Image_Function_Simd::Threshold;

        return table;
    }
}

namespace penguinV
{
    const FunctionTable & functionTable()
    {
        static FunctionTable table = initialize();
        return table;
    }
}
