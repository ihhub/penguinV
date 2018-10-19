#include "../../src/function_pool.h"
#include "../../src/image_function.h"
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

    // Function pointer definitions
    typedef void     (*AbsoluteDifferenceFunction)( const Image & in1, const Image & in2, Image & out );
    typedef void     (*AccumulateFunction)( const Image & image, std::vector < uint32_t > & result );
    typedef void     (*BinaryDilateFunction)( Image & image, uint32_t dilationX, uint32_t dilationY );
    typedef void     (*BinaryErodeFunction)( Image & image, uint32_t erosionX, uint32_t erosionY );
    typedef void     (*BitwiseAndFunction)( const Image & in1, const Image & in2, Image & out );
    typedef void     (*BitwiseOrFunction)( const Image & in1, const Image & in2, Image & out );
    typedef void     (*BitwiseXorFunction)( const Image & in1, const Image & in2, Image & out );
    typedef void     (*ConvertToGrayScaleFunction)( const Image & in, Image & out );
    typedef void     (*ConvertToRgbFunction)( const Image & in, Image & out );
    typedef void     (*CopyFunction)( const Image & in, Image & out );
    typedef void     (*ExtractChannelFunction)( const Image & in, Image & out, uint8_t channelId );
    typedef void     (*FillFunction)( Image & image, uint8_t value );
    typedef void     (*FlipFunction)( const Image & in, Image & out, bool horizontal, bool vertical );
    typedef void     (*GammaCorrectionFunction)( const Image & in, Image & out, double a, double gamma );
    typedef uint8_t  (*GetPixelFunction)( const Image & image, uint32_t x, uint32_t y );
    typedef uint8_t  (*GetThresholdFunction)( const std::vector < uint32_t > & histogram );
    typedef void     (*HistogramFunction)( const Image & image, std::vector < uint32_t > & histogram );
    typedef void     (*InvertFunction)( const Image & in, Image & out );
    typedef bool     (*IsBinaryFunction)( const Image & image );
    typedef bool     (*IsEqualFunction)( const Image & in1, const Image & in2 );
    typedef void     (*LookupTableFunction)( const Image & in, Image & out, const std::vector < uint8_t > & table );
    typedef void     (*MaximumFunction)( const Image & in1, const Image & in2, Image & out );
    typedef void     (*MergeFunction)( const Image & in1, const Image & in2, const Image & in3, Image & out );
    typedef void     (*MinimumFunction)( const Image & in1, const Image & in2, Image & out );
    typedef void     (*NormalizeFunction)( const Image & in, Image & out );
    typedef void     (*ProjectionProfileFunction)( const Image & image, bool horizontal, std::vector < uint32_t > & projection );
    typedef void     (*ResizeFunction)( const Image & in, Image & out );
    typedef void     (*RgbToBgrFunction)( const Image & in, Image & out );
    typedef void     (*RotateFunction)( const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle );
    typedef void     (*SetPixelFunction)( Image & image, uint32_t x, uint32_t y, uint8_t value );
    typedef void     (*SplitFunction)( const Image & in, Image & out1, Image & out2, Image & out3 );
    typedef void     (*SubtractFunction)( const Image & in1, const Image & in2, Image & out );
    typedef uint32_t (*SumFunction)( const Image & image );
    typedef void     (*ThresholdFunction)( const Image & in, Image & out, uint8_t threshold );
    typedef void     (*ThresholdDoubleFunction)( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold );
    typedef void     (*TransposeFunction)( const Image & in, Image & out );

    void SetupFunction( const std::string & namespaceName )
    {
        if ( namespaceName == "function_pool" ) {
            Image_Function_Simd::Simd_Activation::EnableSimd( true );
            Thread_Pool::ThreadPoolMonoid::instance().resize( 4 );
        }
        else if ( namespaceName == "image_function_avx" ) {
            Image_Function_Simd::Simd_Activation::EnableSimd( false );
            Image_Function_Simd::Simd_Activation::EnableAvx( true );
        }
        else if ( namespaceName == "image_function_sse" ) {
            Image_Function_Simd::Simd_Activation::EnableSimd( false );
            Image_Function_Simd::Simd_Activation::EnableSse( true );
        }
        else if ( namespaceName == "image_function_neon" ) {
            Image_Function_Simd::Simd_Activation::EnableSimd( false );
            Image_Function_Simd::Simd_Activation::EnableNeon( true );
        }
    }

    void CleanupFunction(const std::string& namespaceName)
    {
        if ( (namespaceName == "image_function_avx") || (namespaceName == "image_function_sse") || (namespaceName == "image_function_neon") )
            Image_Function_Simd::Simd_Activation::EnableSimd( true );
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

    std::pair < double, double > template_AbsoluteDifference( AbsoluteDifferenceFunction AbsoluteDifference, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( AbsoluteDifference( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_BitwiseAnd( BitwiseAndFunction BitwiseAnd, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( BitwiseAnd( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_BitwiseOr( BitwiseOrFunction BitwiseOr, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( BitwiseOr( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_BitwiseXor( BitwiseXorFunction BitwiseXor, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( BitwiseXor( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_ConvertToGrayScale( ConvertToGrayScaleFunction ConvertToGrayScale, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformRGBImage( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformImage   ( size, size );

        TEST_FUNCTION_LOOP( ConvertToGrayScale( input, output ), namespaceName )
    }

    std::pair < double, double > template_ConvertToRgb( ConvertToRgbFunction ConvertToRgb, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformImage   ( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformRGBImage( size, size );

        TEST_FUNCTION_LOOP( ConvertToRgb( input, output ), namespaceName )
    }

    std::pair < double, double > template_Fill( FillFunction Fill, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );
        uint8_t value = Performance_Test::randomValue<uint8_t>( 256 );

        TEST_FUNCTION_LOOP( Fill( image, value ), namespaceName )
    }

    std::pair < double, double > template_GammaCorrection( GammaCorrectionFunction GammaCorrection, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        double a     = Performance_Test::randomValue <uint32_t>( 100 ) / 100.0;
        double gamma = Performance_Test::randomValue <uint32_t>( 300 ) / 100.0;

        TEST_FUNCTION_LOOP( GammaCorrection( image[0], image[1], a, gamma ), namespaceName )
    }

    std::pair < double, double > template_Histogram( HistogramFunction Histogram, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );
        std::vector < uint32_t > histogramTable;

        TEST_FUNCTION_LOOP( Histogram( image, histogramTable ), namespaceName )
    }

    std::pair < double, double > template_Invert( InvertFunction Invert, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Invert( image[0], image[1] ), namespaceName )
    }

    std::pair < double, double > template_LookupTable( LookupTableFunction LookupTable, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        std::vector<uint8_t> table(256, 0);

        TEST_FUNCTION_LOOP( LookupTable( image[0], image[1], table ), namespaceName )
    }

    std::pair < double, double > template_Maximum( MaximumFunction Maximum, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Maximum( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_Minimum( MinimumFunction Minimum, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Minimum( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_RgbToBgr( RgbToBgrFunction RgbToBgr, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformRGBImages( 2, size, size );

        TEST_FUNCTION_LOOP( RgbToBgr( image[0], image[1] ), namespaceName )
    }

    std::pair < double, double > template_ResizeDown( ResizeFunction Resize, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformImage( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformImage( size / 2, size / 2 );

        TEST_FUNCTION_LOOP( Resize( input, output ), namespaceName )
    }

    std::pair < double, double > template_ResizeUp( ResizeFunction Resize, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image input  = Performance_Test::uniformImage( size, size );
        PenguinV_Image::Image output = Performance_Test::uniformImage( size * 2, size * 2 );

        TEST_FUNCTION_LOOP( Resize( input, output ), namespaceName )
    }

    std::pair < double, double > template_Subtract( SubtractFunction Subtract, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Subtract( image[0], image[1], image[2] ), namespaceName )
    }

    std::pair < double, double > template_Sum( SumFunction Sum, const std::string & namespaceName, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );

        TEST_FUNCTION_LOOP( Sum( image ), namespaceName )
    }

    std::pair < double, double > template_Threshold( ThresholdFunction Threshold, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t threshold = Performance_Test::randomValue<uint8_t>( 256 );

        TEST_FUNCTION_LOOP( Threshold( image[0], image[1], threshold ), namespaceName )
    }

    std::pair < double, double > template_ThresholdDouble( ThresholdDoubleFunction Threshold, const std::string & namespaceName, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t minThreshold = Performance_Test::randomValue<uint8_t>( 256 );
        uint8_t maxThreshold = Performance_Test::randomValue<uint8_t>( minThreshold, 256 );

        TEST_FUNCTION_LOOP( Threshold( image[0], image[1], minThreshold, maxThreshold ), namespaceName )
    }

    std::pair < double, double > template_Transpose( TransposeFunction Transpose, const std::string & namespaceName, uint32_t size )
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
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( ConvertToRgb     )
    SET_FUNCTION( ConvertToGrayScale )
    SET_FUNCTION( Fill               )
    SET_FUNCTION( GammaCorrection    )
    SET_FUNCTION( Histogram          )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( LookupTable        )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
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
    SET_FUNCTION( ConvertToRgb     )
    SET_FUNCTION( ConvertToGrayScale )
    SET_FUNCTION( GammaCorrection    )
    SET_FUNCTION( Histogram          )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( LookupTable        )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( RgbToBgr           )
    REGISTER_FUNCTION( ResizeDown, Resize )
    REGISTER_FUNCTION( ResizeUp, Resize   )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    REGISTER_FUNCTION( ThresholdDouble, Threshold )
}

#ifdef PENGUINV_AVX_SET
namespace image_function_avx
{
    using namespace Image_Function_Simd;

    const bool isSupported = isAvxAvailable;
    const std::string namespaceName = "image_function_avx";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
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

    const bool isSupported = isNeonAvailable;
    const std::string namespaceName = "image_function_neon";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
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

    const bool isSupported = isSseAvailable;
    const std::string namespaceName = "image_function_sse";

    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
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
