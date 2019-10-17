#include "unit_test_helper.h"
#include "unit_test_image_buffer.h"
#include "../../src/image_function.h"

namespace template_image
{
    bool EmptyConstructor()
    {
        penguinV::ImageTemplate < uint8_t > image;
        return Unit_Test::isEmpty( image );
    }

    bool Constructor()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 2048 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );
            const uint8_t  alignment  = Unit_Test::randomValue<uint8_t >( 1, 32 );

            if( !Unit_Test::equalSize( penguinV::ImageTemplate < uint8_t >( width, height, colorCount, alignment ), width, height,
                                       Unit_Test::rowSize( width, colorCount, alignment ), colorCount, alignment ) )
                return false;
        }

        return true;
    }

    bool NullAssignment()
    {
        try {
            penguinV::ImageTemplate < uint8_t > image;

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
    bool _CopyConstructor()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );
            const uint8_t  alignment  = Unit_Test::randomValue<uint8_t >( 1, 32 );

            penguinV::ImageTemplate < _Type > image( width, height, colorCount, alignment );
            image.fill( static_cast<_Type>( Unit_Test::randomValue<uint8_t>( 256u ) ) );

            const penguinV::ImageTemplate < _Type > image_copy( image );

            if( !Unit_Test::equalSize( image, image_copy ) || !Unit_Test::equalData( image, image_copy ) )
                return false;
        }

        return true;
    }

    template <typename _Type>
    bool _AssignmentOperator()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width      = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint32_t height     = Unit_Test::randomValue<uint32_t>( 1024 );
            const uint8_t  colorCount = Unit_Test::randomValue<uint8_t >( 1, 4 );
            const uint8_t  alignment  = Unit_Test::randomValue<uint8_t >( 1, 32 );

            penguinV::ImageTemplate < _Type > image( width, height, colorCount, alignment );
            image.fill( static_cast<_Type>( Unit_Test::randomValue<uint8_t>( 256u ) ) );

            penguinV::ImageTemplate < _Type > image_copy;

            image_copy = image;

            if( !Unit_Test::equalSize( image, image_copy ) || !Unit_Test::equalData( image, image_copy ) )
                return false;
        }

        return true;
    }
}

#define ADD_TEMPLATE_FUNCTION( function, type )                                                                     \
    framework.add( template_image::_##function < type >, std::string("template_image::") + std::string(#function) + \
                   std::string(" (") + std::string(#type) + std::string(")") );

void addTests_Image_Buffer( UnitTestFramework & framework )
{
    ADD_TEST( framework, template_image::EmptyConstructor );
    ADD_TEST( framework, template_image::Constructor );
    ADD_TEST( framework, template_image::NullAssignment );

    ADD_TEMPLATE_FUNCTION( CopyConstructor, uint8_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, uint16_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, uint32_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, uint64_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, int8_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, int16_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, int32_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, int64_t );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, float );
    ADD_TEMPLATE_FUNCTION( CopyConstructor, double );

    ADD_TEMPLATE_FUNCTION( AssignmentOperator, uint8_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, uint16_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, uint32_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, uint64_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, int8_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, int16_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, int32_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, int64_t );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, float );
    ADD_TEMPLATE_FUNCTION( AssignmentOperator, double );
}
