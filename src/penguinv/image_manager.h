/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#pragma once

#include "../image_buffer.h"
#include <vector>

template <typename _Type>
struct ReferenceOwner
{
    explicit ReferenceOwner( penguinV::ImageTemplate<_Type> & data_ )
        : data( data_ )
    {}
    penguinV::ImageTemplate<_Type> & data;
};

template <typename _Type>
struct ConstReferenceOwner
{
    explicit ConstReferenceOwner( const penguinV::ImageTemplate<_Type> & data_ )
        : data( data_ )
    {}
    const penguinV::ImageTemplate<_Type> & data;
};

template <typename _Type>
class ImageManager
{
public:
    typedef penguinV::ImageTemplate<_Type> ( *GenerateImage )( uint8_t imageType );
    typedef void ( *ConvertImage )( const penguinV::ImageTemplate<_Type> & in, penguinV::ImageTemplate<_Type> & out );

    explicit ImageManager( uint8_t requiredType, GenerateImage generateImage, ConvertImage convertImage )
        : _type( requiredType )
        , _generateImage( generateImage )
        , _convertImage( convertImage )
    {}

    ~ImageManager()
    {
        for ( typename std::vector<ConstReferenceOwner<_Type> *>::iterator data = _input.begin(); data != _input.end(); ++data )
            delete *data;

        for ( size_t i = 0u; i < _output.size(); ++i ) {
            _restore( _outputClone[i], _output[i]->data );

            delete _output[i];
        }
    }

    const penguinV::Image & operator()( const penguinV::ImageTemplate<_Type> & image )
    {
        if ( image.type() != _type ) {
            _inputClone.push_back( _clone( image ) );
            return _inputClone.back();
        }
        else {
            return image;
        }
    }

    penguinV::ImageTemplate<_Type> & operator()( penguinV::ImageTemplate<_Type> & image )
    {
        if ( image.type() != _type ) {
            _output.push_back( new ReferenceOwner<_Type>( image ) );
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
    std::vector<ConstReferenceOwner<_Type> *> _input;
    std::vector<ReferenceOwner<_Type> *> _output;
    std::vector<penguinV::ImageTemplate<_Type>> _inputClone;
    std::vector<penguinV::ImageTemplate<_Type>> _outputClone;

    penguinV::Image _clone( const penguinV::ImageTemplate<_Type> & in )
    {
        penguinV::ImageTemplate<_Type> temp = _generateImage( _type ).generate( in.width(), in.height(), in.colorCount() );
        _convertImage( in, temp );
        return temp;
    }

    void _restore( const penguinV::ImageTemplate<_Type> & in, penguinV::ImageTemplate<_Type> & out )
    {
        _convertImage( in, out );
    }
};
