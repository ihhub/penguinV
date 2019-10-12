#include "../../src/function_pool.h"
#include "../../src/image_function.h"
#include "../../src/image_function_helper.h"
#include "../../src/image_function_simd.h"
#include "../../src/thread_pool.h"
#include "../../src/penguinv/cpu_identification.h"
#include "performance_test_image_function.h"
#include "performance_test_helper.h"

namespace
{
    class FunctionRegistrator
    {
    public:
        static FunctionRegistrator& instance()
        {
            static FunctionRegistrator registrator;
            return registrator;
        }

        void add( const PerformanceTestFramework::testFunction test, const std::string & name )
        {
            _function[test] = name;
        }

        void set( PerformanceTestFramework & framework )
        {
            for (std::map < PerformanceTestFramework::testFunction, std::string >::const_iterator func = _function.cbegin(); func != _function.cend(); ++func)
                framework.add( func->first, func->second );

            _function.clear();
        }

    private:
        std::map < PerformanceTestFramework::testFunction, std::string > _function; // container with pointer to functions and their names
    };
}

namespace Function_Template
{
    using namespace PenguinV_Image;
    using namespace Image_Function_Helper::FunctionTable;

    void SetupFunction( const std::string & namespaceName )
    {
        if ( namespaceName == "function_pool" ) {
            simd::EnableSimd( true );
            ThreadPoolMonoid::instance().resize( 4 );
        }
        else if ( namespaceName == "image_function_avx512" ) {
            simd::EnableSimd( false );
            simd::EnableAVX512( true );
        }
        else if ( namespaceName == "image_function_avx" ) {
            simd::EnableSimd( false );
            simd::EnableAvx( true );
        }
        else if ( namespaceName == "image_function_sse" ) {
            simd::EnableSimd( false );
            simd::EnableSse( true );
        }
        else if ( namespaceName == "image_function_neon" ) {
            simd::EnableSimd( false );
            simd::EnableNeon( true );
        }
    }

    void CleanupFunction(const std::string& namespaceName)
    {
        if ( (namespaceName == "image_function_avx512") || (namespaceName == "image_function_avx") || (namespaceName == "image_function_sse") || (namespaceName == "image_function_neon") )
            simd::EnableSimd( true );
    }

    #define TEST_FUNCTION_LOOP( testFunction, namespaceName )          \
        SetupFunction( namespaceName );                                \
        Performance_Test::TimerContainer timer;                        \
        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) { \
            timer.start();                                             \
            testFunction;                                              \
            timer.stop();                                              \
        }                                                              \
        CleanupFunction( namespaceName );                              \
        return timer.mean();

    std::pair < double, double > template_AbsoluteDifference( AbsoluteDifferenceForm2 AbsoluteDifference, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( AbsoluteDifference( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_Accumulate( AccumulateForm1 Accumulate , const std::string & namespaceName, uint32_t size )
    {
        const PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );
        std::vector < uint32_t > result( size * size * image.colorCount(), 0u );

        TEST_FUNCTION_LOOP( Accumulate( image, result ), namespaceName )
    }

    std::pair < double, double > template_BitwiseAnd( BitwiseAndForm2 BitwiseAnd, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( BitwiseAnd( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_BitwiseOr( BitwiseOrForm2 BitwiseOr, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( BitwiseOr( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_BitwiseXor( BitwiseXorForm2 BitwiseXor, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( BitwiseXor( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_ConvertToGrayScale( ConvertToGrayScaleForm2 ConvertToGrayScale, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformRGBImage( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformImage   ( size, size );

        TEST_FUNCTION_LOOP( ConvertToGrayScale( input, output ), namespaceName )
    }

    std::pair < double, double > template_ConvertToRgb( ConvertToRgbForm2 ConvertToRgb, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformImage   ( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformRGBImage( size, size );

        TEST_FUNCTION_LOOP( ConvertToRgb( input, output ), namespaceName )
    }

    std::pair < double, double > template_Fill( FillForm1 Fill, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );
        uint8_t value = Performance_Test::randomValue<uint8_t>( 256 );

        TEST_FUNCTION_LOOP( Fill( image, value ), namespaceName )
    }

    std::pair < double, double > template_Flip( FlipForm2 Flip, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Flip( image[0], image[1], true, true), namespaceName )
    }

    std::pair < double, double > template_GammaCorrection( GammaCorrectionForm2 GammaCorrection, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        double a     = Performance_Test::randomValue <uint32_t>( 100 ) / 100.0;
        double gamma = Performance_Test::randomValue <uint32_t>( 300 ) / 100.0;

        TEST_FUNCTION_LOOP( GammaCorrection( image[0], image[1], a, gamma ), namespaceName )
    }

    std::pair < double, double > template_Histogram( HistogramForm2 Histogram, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );
        std::vector < uint32_t > histogramTable;

        TEST_FUNCTION_LOOP( Histogram( image, histogramTable ), namespaceName )
    }

    std::pair < double, double > template_Invert( InvertForm2 Invert, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Invert( image[0], image[1] ), namespaceName )
    }

    std::pair < double, double > template_LookupTable( LookupTableForm2 LookupTable, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        std::vector<uint8_t> table(256, 0);

        TEST_FUNCTION_LOOP( LookupTable( image[0], image[1], table ), namespaceName )
    }

    std::pair < double, double > template_Maximum( MaximumForm2 Maximum, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Maximum( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_Minimum( MinimumForm2 Minimum, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Minimum( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_ProjectionProfile( ProjectionProfileForm2 ProjectionProfile , const std::string & namespaceName, uint32_t size )
    {
        const PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );
        std::vector < uint32_t > projection;

        TEST_FUNCTION_LOOP( ProjectionProfile( image, false, projection ), namespaceName )
    }

    std::pair < double, double > template_RgbToBgr( RgbToBgrForm2 RgbToBgr, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformRGBImages( 2, size, size );

        TEST_FUNCTION_LOOP( RgbToBgr( image[0], image[1] ), namespaceName )
    }

    std::pair < double, double > template_ResizeDown( ResizeForm2 Resize, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformImage( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformImage( size / 2, size / 2 );

        TEST_FUNCTION_LOOP( Resize( input, output ), namespaceName )
    }

    std::pair < double, double > template_ResizeUp( ResizeForm2 Resize, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformImage( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformImage( size * 2, size * 2 );

        TEST_FUNCTION_LOOP( Resize( input, output ), namespaceName )
    }

    std::pair < double, double > template_Subtract( SubtractForm2 Subtract, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Subtract( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_Sum( SumForm1 Sum, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );

        TEST_FUNCTION_LOOP( Sum( image ), namespaceName )
    }

    std::pair < double, double > template_Threshold( ThresholdForm2 Threshold, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t threshold = Performance_Test::randomValue<uint8_t>( 256 );

        TEST_FUNCTION_LOOP( Threshold( image[0], image[1], threshold ), namespaceName )
    }

    std::pair < double, double > template_ThresholdDouble( ThresholdDoubleForm2 Threshold, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t minThreshold = Performance_Test::randomValue<uint8_t>( 256 );
        uint8_t maxThreshold = Performance_Test::randomValue<uint8_t>( minThreshold, 256 );

        TEST_FUNCTION_LOOP( Threshold( image[0], image[1], minThreshold, maxThreshold ), namespaceName )
    }

    std::pair < double, double > template_Transpose( TransposeForm2 Transpose, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Transpose( image[0], image[1] ), namespaceName )
    }
}

#define FUNCTION_REGISTRATION( function, functionWrapper, size )                                                                                   \
struct Register_##functionWrapper                                                                                                                  \
{                                                                                                                                                  \
    explicit Register_##functionWrapper( bool makeRegistration )                                                                                   \
    {                                                                                                                                              \
        if( makeRegistration )                                                                                                                     \
            FunctionRegistrator::instance().add( functionWrapper, namespaceName + std::string("::") + std::string(#function) + std::string(" (") + \
                                                 std::string(#size) + std::string("x") + std::string(#size) + std::string(")") );                  \
    }                                                                                                                                              \
};                                                                                                                                                 \
const Register_##functionWrapper registrator_##functionWrapper( isSupported );

#define REGISTER_FUNCTION( functionName, functionPointer )                                                                                             \
    std::pair < double, double > type1_##functionName() { return Function_Template::template_##functionName( functionPointer, namespaceName, 256  ); } \
    std::pair < double, double > type2_##functionName() { return Function_Template::template_##functionName( functionPointer, namespaceName, 512  ); } \
    std::pair < double, double > type3_##functionName() { return Function_Template::template_##functionName( functionPointer, namespaceName, 1024 ); } \
    std::pair < double, double > type4_##functionName() { return Function_Template::template_##functionName( functionPointer, namespaceName, 2048 ); } \
    FUNCTION_REGISTRATION( functionName, type1_##functionName, 256  )                                                                                  \
    FUNCTION_REGISTRATION( functionName, type2_##functionName, 512  )                                                                                  \
    FUNCTION_REGISTRATION( functionName, type3_##functionName, 1024 )                                                                                  \
    FUNCTION_REGISTRATION( functionName, type4_##functionName, 2048 )

#define SET_FUNCTION( function ) REGISTER_FUNCTION( function, function )

namespace image_function
{
    using namespace Image_Function;

    const bool isSupported = true;
    const std::string namespaceName = "image_function";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( Accumulate         )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( ConvertToRgb       )
    SET_FUNCTION( ConvertToGrayScale )
    SET_FUNCTION( Flip               )
    SET_FUNCTION( Fill               )
    SET_FUNCTION( GammaCorrection    )
    SET_FUNCTION( Histogram          )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( LookupTable        )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( ProjectionProfile  )
    SET_FUNCTION( RgbToBgr           )
    REGISTER_FUNCTION( ResizeDown, Resize )
    REGISTER_FUNCTION( ResizeUp, Resize   )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    REGISTER_FUNCTION( ThresholdDouble, Threshold )
    SET_FUNCTION( Transpose          )
}

namespace function_pool
{
    using namespace Function_Pool;

    const bool isSupported = true;
    const std::string namespaceName = "function_pool";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( ConvertToRgb       )
    SET_FUNCTION( ConvertToGrayScale )
    SET_FUNCTION( GammaCorrection    )
    SET_FUNCTION( Histogram          )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( LookupTable        )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( ProjectionProfile  )
    SET_FUNCTION( RgbToBgr           )
    REGISTER_FUNCTION( ResizeDown, Resize )
    REGISTER_FUNCTION( ResizeUp, Resize   )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    REGISTER_FUNCTION( ThresholdDouble, Threshold )
}

#ifdef PENGUIV_AV512BW_SET
namespace image_function_avx512
{
    using namespace Image_Function_Simd;

    const bool isSupported = SimdInfo::isAvx512Available();
    const std::string namespaceName = "image_function_avx512";

    SET_FUNCTION( AbsoluteDifference )
}
#endif

#ifdef PENGUINV_AVX_SET
namespace image_function_avx
{
    using namespace Image_Function_Simd;

    const bool isSupported = SimdInfo::isAvxAvailable();
    const std::string namespaceName = "image_function_avx";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( Accumulate         )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( ProjectionProfile  )
    SET_FUNCTION( RgbToBgr           )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    REGISTER_FUNCTION( ThresholdDouble, Threshold )
}
#endif

#ifdef PENGUINV_NEON_SET
namespace image_function_neon
{
    using namespace Image_Function_Simd;

    const bool isSupported = SimdInfo::isNeonAvailable();
    const std::string namespaceName = "image_function_neon";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( Accumulate         )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( ConvertToRgb       )
    SET_FUNCTION( Flip               )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( ProjectionProfile  )
    SET_FUNCTION( RgbToBgr           )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    REGISTER_FUNCTION( ThresholdDouble, Threshold )
}
#endif

#ifdef PENGUINV_SSE_SET
namespace image_function_sse
{
    using namespace Image_Function_Simd;

    const bool isSupported = SimdInfo::isSseAvailable();
    const std::string namespaceName = "image_function_sse";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( Accumulate         )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( ConvertToRgb       )
    SET_FUNCTION( Flip               )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( ProjectionProfile  )
    SET_FUNCTION( RgbToBgr           )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    REGISTER_FUNCTION( ThresholdDouble, Threshold )
}
#endif

void addTests_Image_Function( PerformanceTestFramework & framework )
{
    FunctionRegistrator::instance().set( framework );
}
