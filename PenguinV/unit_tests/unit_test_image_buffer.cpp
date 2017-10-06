#include "unit_test_image_buffer.h"
#include "unit_test_helper.h"
#include "../Library/image_function.h"

namespace Unit_Test
{
    void addTests_Image_Buffer( UnitTestFramework & framework )
    {
        ADD_TEST( framework, Template_Image_Test::EmptyConstructorTest );
        ADD_TEST( framework, Template_Image_Test::Constructor2ParametersTest );
        ADD_TEST( framework, Template_Image_Test::Constructor3ParametersTest );
        ADD_TEST( framework, Template_Image_Test::Constructor4ParametersTest );

        ADD_TEST( framework, Template_Image_Test::CopyConstructorU8Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorU16Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorU32Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorU64Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorS8Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorS16Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorS32Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorS64Test );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorFTest );
        ADD_TEST( framework, Template_Image_Test::CopyConstructorDTest );

        ADD_TEST( framework, Template_Image_Test::NullAssignmentTest );

        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorU8Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorU16Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorU32Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorU64Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorS8Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorS16Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorS32Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorS64Test );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorFTest );
        ADD_TEST( framework, Template_Image_Test::AssignmentOperatorDTest );
    }

    namespace Template_Image_Test
    {
        bool EmptyConstructorTest()
        {
            Template_Image::ImageTemplate < uint8_t > image;

            return isEmpty( image );
        }

        bool Constructor2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 2048 );
                uint32_t height     = randomValue<uint32_t>( 2048 );

                if( !equalSize( Template_Image::ImageTemplate < uint8_t >( width, height ), width, height, rowSize( width ), 1, 1 ) )
                    return false;
            }

            return true;
        }

        bool Constructor3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 2048 );
                uint32_t height     = randomValue<uint32_t>( 2048 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );

                if( !equalSize( Template_Image::ImageTemplate < uint8_t >( width, height, colorCount ), width, height,
                                rowSize( width, colorCount ), colorCount, 1 ) )
                    return false;
            }

            return true;
        }

        bool Constructor4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 2048 );
                uint32_t height     = randomValue<uint32_t>( 2048 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                if( !equalSize( Template_Image::ImageTemplate < uint8_t >( width, height, colorCount, alignment ), width, height,
                                rowSize( width, colorCount, alignment ), colorCount, alignment ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorU8Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint8_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint8_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorU16Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint16_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint16_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorU32Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint32_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint32_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorU64Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint64_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint64_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorS8Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int8_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int8_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorS16Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int16_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int16_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorS32Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int32_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int32_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorS64Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int64_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int64_t > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorFTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < float > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < float > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool CopyConstructorDTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < double > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < double > image_copy( image );

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool NullAssignmentTest()
        {
            try {
                Template_Image::ImageTemplate < uint8_t > image;

                uint8_t fakeArray[1];
                uint8_t fakeValue = static_cast<uint8_t>(rand() % 2);
                if( fakeValue == 1 )
                    fakeValue = 0;

                image.assign( fakeArray, fakeValue, fakeValue, fakeValue, fakeValue );
            }
            catch( imageException & ) {
                return true;
            }

            return false;
        }

        bool AssignmentOperatorU8Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint8_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint8_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorU16Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint16_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint16_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorU32Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint32_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint32_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorU64Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < uint64_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < uint64_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorS8Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int8_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int8_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorS16Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int16_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int16_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorS32Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int32_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int32_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorS64Test()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < int64_t > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < int64_t > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorFTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < float > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < float > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }

        bool AssignmentOperatorDTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint32_t width      = randomValue<uint32_t>( 1024 );
                uint32_t height     = randomValue<uint32_t>( 1024 );
                uint8_t  colorCount = randomValue<uint8_t >( 1, 4 );
                uint8_t  alignment  = randomValue<uint8_t >( 1, 32 );

                Template_Image::ImageTemplate < double > image( width, height, colorCount, alignment );
                Template_Image::ImageTemplate < double > image_copy;

                image_copy = image;

                if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
                    return false;
            }

            return true;
        }
    }
}
