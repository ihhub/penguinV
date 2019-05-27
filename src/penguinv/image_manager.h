#pragma once

#include <vector>
#include "../image_buffer.h"

template <typename _Type>
struct ReferenceOwner
{
    explicit ReferenceOwner( PenguinV_Image::ImageTemplate< _Type > & data_ )
        : data( data_ )
    {
    }
    PenguinV_Image::ImageTemplate< _Type > & data;
};

template <typename _Type>
struct ConstReferenceOwner
{
    explicit ConstReferenceOwner( const PenguinV_Image::ImageTemplate< _Type > & data_ )
        : data( data_ )
    {
    }
    const PenguinV_Image::ImageTemplate< _Type > & data;
};

template <typename _Type>
class ImageManager
{
public:
    typedef PenguinV_Image::ImageTemplate< _Type > (*GenerateImage)( uint8_t imageType );
    typedef void (*ConvertImage)( const PenguinV_Image::ImageTemplate< _Type > & in, PenguinV_Image::ImageTemplate< _Type > & out );

    explicit ImageManager( uint8_t requiredType, GenerateImage generateImage, ConvertImage convertImage )
        : _type( requiredType )
        , _generateImage( generateImage )
        , _convertImage( convertImage )
    {
    }

    ~ImageManager()
    {
        for ( std::vector<ConstReferenceOwner< _Type > *>::iterator data = _input.begin(); data != _input.end(); ++data )
            delete *data;

        for ( size_t i = 0u; i < _output.size(); ++i ) {
            _restore( _outputClone[i], _output[i]->data );

            delete _output[i];
        }
    }

    const PenguinV_Image::Image & operator ()( const PenguinV_Image::ImageTemplate< _Type > & image )
    {
        if ( image.type() != _type ) {
            _inputClone.push_back( _clone( image ) );
            return _inputClone.back();
        }
        else {
            return image;
        }
    }

    PenguinV_Image::ImageTemplate< _Type > & operator ()( PenguinV_Image::ImageTemplate< _Type > & image )
    {
        if ( image.type() != _type ) {
            _output.push_back( new ReferenceOwner< _Type >( image ) );
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
    std::vector< ConstReferenceOwner< _Type > * > _input;
    std::vector< ReferenceOwner< _Type > * > _output;
    std::vector< PenguinV_Image::ImageTemplate< _Type > > _inputClone;
    std::vector< PenguinV_Image::ImageTemplate< _Type > > _outputClone;

    PenguinV_Image::Image _clone( const PenguinV_Image::ImageTemplate< _Type > & in )
    {
        PenguinV_Image::ImageTemplate< _Type > temp = _generateImage( _type ).generate( in.width(), in.height(), in.colorCount() );
        _convertImage( in, temp );
        return temp;
    }

    void _restore( const PenguinV_Image::ImageTemplate< _Type > & in, PenguinV_Image::ImageTemplate< _Type > & out )
    {
        _convertImage( in, out );
    }
};
