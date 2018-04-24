#include "unit_test_helper.h"
#include "unit_test_image_buffer.h"
#include "../../src/image_function.h"

namespace template_image
{
    bool EmptyConstructorTest()
    {
        PenguinV_Image::ImageTemplate < uint8_t > image;

        return Unit_Test::isEmpty( image );
    }

    bool Constructor2ParametersTest()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 2048 );

            if( !Unit_Test::equalSize( PenguinV_Image::ImageTemplate < uint8_t >( width, height ), width, height, Unit_Test::rowSize( width ), 1, 1 ) )
                return false;
        }

        return true;
    }

    bool Constructor3ParametersTest()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );

            if( !Unit_Test::equalSize( PenguinV_Image::ImageTemplate < uint8_t >( width, height, colorCount ), width, height,
                                       Unit_Test::rowSize( width, colorCount ), colorCount, 1 ) )
                return false;
        }

        return true;
    }

    bool Constructor4ParametersTest()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );
            const uint8_t  alignment  = Unit_Test::randomValue<uint8_t >( 1, 32 );

            if( !Unit_Test::equalSize( PenguinV_Image::ImageTemplate < uint8_t >( width, height, colorCount, alignment ), width, height,
                                       Unit_Test::rowSize( width, colorCount, alignment ), colorCount, alignment ) )
                return false;
        }

        return true;
    }

    template <typename _Type>
    bool CopyConstructorTest()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );
            const uint8_t  alignment  = Unit_Test::randomValue<uint8_t >( 1, 32 );

            PenguinV_Image::ImageTemplate < _Type > image( width, height, colorCount, alignment );
            PenguinV_Image::ImageTemplate < _Type > image_copy( image );

            if( !Unit_Test::equalSize( image, image_copy ) || !Unit_Test::equalData( image, image_copy ) )
                return false;
        }

        return true;
    }

    bool CopyConstructorU8Test()
    {
        return CopyConstructorTest < uint8_t > ();
    }

    bool CopyConstructorU16Test()
    {
        return CopyConstructorTest < uint16_t > ();
    }

    bool CopyConstructorU32Test()
    {
        return CopyConstructorTest < uint32_t > ();
    }

    bool CopyConstructorU64Test()
    {
        return CopyConstructorTest < uint64_t > ();
    }

    bool CopyConstructorS8Test()
    {
        return CopyConstructorTest < int8_t > ();
    }

    bool CopyConstructorS16Test()
    {
        return CopyConstructorTest < int16_t > ();
    }

    bool CopyConstructorS32Test()
    {
        return CopyConstructorTest < int32_t > ();
    }

    bool CopyConstructorS64Test()
    {
        return CopyConstructorTest < int64_t > ();
    }

    bool CopyConstructorFTest()
    {
        return CopyConstructorTest < float > ();
    }

    bool CopyConstructorDTest()
    {
        return CopyConstructorTest < double > ();
    }

    bool NullAssignmentTest()
    {
        try {
            PenguinV_Image::ImageTemplate < uint8_t > image;

            uint8_t fakeArray[1];
            uint8_t fakeValue = Unit_Test::randomValue<uint8_t>( 2 );
            if( fakeValue == 1 )
                fakeValue = 0;

            image.assign( fakeArray, fakeValue, fakeValue, fakeValue, fakeValue );
        }
        catch( imageException & ) {
            return true;
        }

        return false;
    }

    template <typename _Type>
    bool AssignmentOperatorTest()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );
            const uint8_t  alignment  = Unit_Test::randomValue<uint8_t >( 1, 32 );

            PenguinV_Image::ImageTemplate < _Type > image( width, height, colorCount, alignment );
            PenguinV_Image::ImageTemplate < _Type > image_copy;

            image_copy = image;

            if( !Unit_Test::equalSize( image, image_copy ) || !Unit_Test::equalData( image, image_copy ) )
                return false;
        }

        return true;
    }

    bool AssignmentOperatorU8Test()
    {
        return AssignmentOperatorTest < uint8_t > ();
    }

    bool AssignmentOperatorU16Test()
    {
        return AssignmentOperatorTest < uint16_t > ();
    }

    bool AssignmentOperatorU32Test()
    {
        return AssignmentOperatorTest < uint32_t > ();
    }

    bool AssignmentOperatorU64Test()
    {
        return AssignmentOperatorTest < uint64_t > ();
    }

    bool AssignmentOperatorS8Test()
    {
        return AssignmentOperatorTest < int8_t > ();
    }

    bool AssignmentOperatorS16Test()
    {
        return AssignmentOperatorTest < int16_t > ();
    }

    bool AssignmentOperatorS32Test()
    {
        return AssignmentOperatorTest < int32_t > ();
    }

    bool AssignmentOperatorS64Test()
    {
        return AssignmentOperatorTest < int64_t > ();
    }

    bool AssignmentOperatorFTest()
    {
        return AssignmentOperatorTest < float > ();
    }

    bool AssignmentOperatorDTest()
    {
        return AssignmentOperatorTest < double > ();
    }
}

namespace Unit_Test
{
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
