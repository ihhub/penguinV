#pragma once

#include <vector>
#include "../image_buffer.h"

struct ReferenceOwner
{
    explicit ReferenceOwner( penguinV::Image & data_ )
        : data( data_ )
    {
    }
    penguinV::Image & data;
};

struct ConstReferenceOwner
{
    explicit ConstReferenceOwner( const penguinV::Image & data_ )
        : data( data_ )
    {
    }
    const penguinV::Image & data;
};

class ImageManager
{
public:
    typedef penguinV::Image ( *GenerateImage )( uint8_t imageType );
    typedef void ( *ConvertImage )( const penguinV::Image & in, penguinV::Image & out );

    explicit ImageManager( uint8_t requiredType, GenerateImage generateImage, ConvertImage convertImage )
        : _type( requiredType )
        , _generateImage( generateImage )
        , _convertImage( convertImage )
    {
    }

    ~ImageManager()
    {
        for ( std::vector<ConstReferenceOwner *>::iterator data = _input.begin(); data != _input.end(); ++data )
            delete *data;

        for ( size_t i = 0u; i < _output.size(); ++i ) {
            _restore( _outputClone[i], _output[i]->data );

            delete _output[i];
        }
    }

    const penguinV::Image & operator()( const penguinV::Image & image )
    {
        if ( image.type() != _type ) {
            _inputClone.push_back( _clone( image ) );
            return _inputClone.back();
        }
        else {
            return image;
        }
    }

    penguinV::Image & operator()( penguinV::Image & image )
    {
        if ( image.type() != _type ) {
            _output.push_back( new ReferenceOwner( image ) );
            _outputClone.push_back( _clone( image ) );
            return _outputClone.back();
        }
        else {
            return image;
        }
    }
private:
    uint8_t _type;
    GenerateImage _generateImage;
    ConvertImage _convertImage;
    std::vector< ConstReferenceOwner * > _input;
    std::vector< ReferenceOwner * > _output;
    std::vector<penguinV::Image> _inputClone;
    std::vector<penguinV::Image> _outputClone;

    penguinV::Image _clone( const penguinV::Image & in )
    {
        penguinV::Image temp = _generateImage( _type ).generate( in.width(), in.height(), in.colorCount() );
        _convertImage( in, temp );
        return temp;
    }

    void _restore( const penguinV::Image & in, penguinV::Image & out )
    {
        _convertImage( in, out );
    }
};
