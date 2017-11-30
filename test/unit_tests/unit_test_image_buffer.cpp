#include "unit_test_image_buffer.h"
#include "unit_test_helper.h"
#include "../../src/image_function.h"

namespace Unit_Test
{
    namespace template_image
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

    void addTests_Image_Buffer( UnitTestFramework & framework )
    {
        ADD_TEST( framework, template_image::EmptyConstructorTest );
        ADD_TEST( framework, template_image::Constructor2ParametersTest );
        ADD_TEST( framework, template_image::Constructor3ParametersTest );
        ADD_TEST( framework, template_image::Constructor4ParametersTest );

        ADD_TEST( framework, template_image::CopyConstructorU8Test );
        ADD_TEST( framework, template_image::CopyConstructorU16Test );
        ADD_TEST( framework, template_image::CopyConstructorU32Test );
        ADD_TEST( framework, template_image::CopyConstructorU64Test );
        ADD_TEST( framework, template_image::CopyConstructorS8Test );
        ADD_TEST( framework, template_image::CopyConstructorS16Test );
        ADD_TEST( framework, template_image::CopyConstructorS32Test );
        ADD_TEST( framework, template_image::CopyConstructorS64Test );
        ADD_TEST( framework, template_image::CopyConstructorFTest );
        ADD_TEST( framework, template_image::CopyConstructorDTest );

        ADD_TEST( framework, template_image::NullAssignmentTest );

        ADD_TEST( framework, template_image::AssignmentOperatorU8Test );
        ADD_TEST( framework, template_image::AssignmentOperatorU16Test );
        ADD_TEST( framework, template_image::AssignmentOperatorU32Test );
        ADD_TEST( framework, template_image::AssignmentOperatorU64Test );
        ADD_TEST( framework, template_image::AssignmentOperatorS8Test );
        ADD_TEST( framework, template_image::AssignmentOperatorS16Test );
        ADD_TEST( framework, template_image::AssignmentOperatorS32Test );
        ADD_TEST( framework, template_image::AssignmentOperatorS64Test );
        ADD_TEST( framework, template_image::AssignmentOperatorFTest );
        ADD_TEST( framework, template_image::AssignmentOperatorDTest );
    }
}
