/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
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

#include "memory/cpu_memory.h"
#include "penguinv_exception.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace penguinV
{
    template <typename TColorDepth>
    class ImageTemplate
    {
    public:
        explicit ImageTemplate( const uint32_t width_ = 0u, const uint32_t height_ = 0u, const uint8_t colorCount_ = 1u, const uint8_t alignment_ = 1u )
        {
            _setType();

            setColorCount( colorCount_ );
            setAlignment( alignment_ );
            resize( width_, height_ );
        }

        ImageTemplate( const ImageTemplate & image )
            : _deviceType( image._deviceType )
        {
            copy( image );
        }

        ImageTemplate( ImageTemplate && image )
        {
            swap( image );
        }

        ImageTemplate & operator=( const ImageTemplate & image )
        {
            copy( image );

            return ( *this );
        }

        ImageTemplate & operator=( ImageTemplate && image )
        {
            swap( image );

            return ( *this );
        }

        bool operator==( const ImageTemplate & image ) const
        {
            return _data == image._data && _width == image._width && _height == image._height && _colorChannelCount == image._colorChannelCount
                   && _alignment == image._alignment && _deviceType == image._deviceType;
        }

        virtual ~ImageTemplate()
        {
            clear();
        }

        void resize( uint32_t width_, uint32_t height_ )
        {
            if ( width_ > 0 && height_ > 0 && ( width_ != _width || height_ != _height ) ) {
                clear();

                _width = width_;
                _height = height_;

                _rowSize = width() * colorCount();
                if ( _rowSize % alignment() != 0 ) {
                    _rowSize = ( _rowSize / alignment() + 1 ) * alignment();
                }

                _data = _allocate( static_cast<size_t>( _height ) * static_cast<size_t>( _rowSize ) );
            }
        }

        void clear()
        {
            if ( _data != nullptr ) {
                _deallocate( _data );
                _data = nullptr;
            }

            _width = 0;
            _height = 0;
            _rowSize = 0;
        }

        TColorDepth * data()
        {
            return _data;
        }

        const TColorDepth * data() const
        {
            return _data;
        }

        void assign( TColorDepth * data_, uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if ( data_ == nullptr || width_ == 0 || height_ == 0 || colorCount_ == 0 || alignment_ == 0 ) {
                throw penguinVException( "Invalid image assignment parameters" );
            }

            clear();

            _width = width_;
            _height = height_;

            _colorChannelCount = colorCount_;
            _alignment = alignment_;

            _data = data_;

            _rowSize = width() * colorCount();
            if ( _rowSize % alignment() != 0 ) {
                _rowSize = ( _rowSize / alignment() + 1 ) * alignment();
            }
        }

        bool empty() const
        {
            return _data == nullptr;
        }

        uint32_t width() const
        {
            return _width;
        }

        uint32_t height() const
        {
            return _height;
        }

        uint32_t rowSize() const
        {
            return _rowSize;
        }

        uint8_t colorCount() const
        {
            return _colorChannelCount;
        }

        void setColorCount( uint8_t colorCount_ )
        {
            if ( colorCount_ > 0 && _colorChannelCount != colorCount_ ) {
                clear();
                _colorChannelCount = colorCount_;
            }
        }

        uint8_t alignment() const
        {
            return _alignment;
        }

        void setAlignment( uint8_t alignment_ )
        {
            if ( alignment_ > 0 && alignment_ != _alignment ) {
                clear();
                _alignment = alignment_;
            }
        }

        void fill( TColorDepth value )
        {
            if ( empty() ) {
                return;
            }

            _set( data(), value, sizeof( TColorDepth ) * height() * rowSize() );
        }

        void swap( ImageTemplate & image )
        {
            std::swap( _width, image._width );
            std::swap( _height, image._height );

            std::swap( _colorChannelCount, image._colorChannelCount );
            std::swap( _rowSize, image._rowSize );
            std::swap( _alignment, image._alignment );

            std::swap( _data, image._data );
            std::swap( _deviceType, image._deviceType );
        }

        void copy( const ImageTemplate & image )
        {
            if ( _deviceType != image._deviceType ) {
                throw penguinVException( "Cannot copy image of one type to another type." );
            }

            clear();

            _width = image._width;
            _height = image._height;

            _colorChannelCount = image._colorChannelCount;
            _rowSize = image._rowSize;
            _alignment = image._alignment;

            if ( image._data != nullptr ) {
                _data = _allocate( static_cast<size_t>( _height ) * static_cast<size_t>( _rowSize ) );

                _copy( _data, image._data, sizeof( TColorDepth ) * _height * _rowSize );
            }
        }

        bool mutate( uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if ( colorCount_ > 0 && alignment_ > 0 ) {
                uint32_t rowSize_ = width_ * colorCount_;
                if ( rowSize_ % alignment_ != 0 ) {
                    rowSize_ = ( rowSize_ / alignment_ + 1 ) * alignment_;
                }

                if ( rowSize_ * height_ != _rowSize * _height ) {
                    return false;
                }

                _width = width_;
                _height = height_;
                _colorChannelCount = colorCount_;
                _alignment = alignment_;
                _rowSize = rowSize_;

                return true;
            }

            return false;
        }

        uint8_t type() const
        {
            return _deviceType;
        }

        ImageTemplate generate( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u ) const
        {
            ImageTemplate image;
            image._deviceType = _deviceType;

            image.setColorCount( colorCount_ );
            image.setAlignment( alignment_ );
            image.resize( width_, height_ );

            return image;
        }

    protected:
        using AllocateFunction = std::function<TColorDepth *( size_t size )>;
        using DeallocateFunction = std::function<void( TColorDepth * data )>;
        using CopyFunction = std::function<void( TColorDepth * out, TColorDepth * in, size_t size )>;
        using SetFunction = std::function<void( TColorDepth * data, TColorDepth value, size_t size )>;

        void _setType( uint8_t type = 0u, AllocateFunction allocateFunction = _allocateMemory, DeallocateFunction deallocateFunction = _deallocateMemory,
                       CopyFunction copyFunction = _copyMemory, SetFunction setFunction = _setMemory )
        {
            _deviceType = type;
            FunctionFacade::instance().initialize( _deviceType, allocateFunction, deallocateFunction, copyFunction, setFunction );
        }

    private:
        TColorDepth * _allocate( size_t size ) const
        {
            return FunctionFacade::instance().allocate( _deviceType )( size );
        }

        void _deallocate( TColorDepth * data ) const
        {
            FunctionFacade::instance().deallocate( _deviceType )( data );
        }

        void _copy( TColorDepth * out, TColorDepth * in, size_t size ) const
        {
            FunctionFacade::instance().copy( _deviceType )( out, in, size );
        }

        void _set( TColorDepth * data, TColorDepth value, size_t size ) const
        {
            FunctionFacade::instance().set( _deviceType )( data, value, size );
        }

        static TColorDepth * _allocateMemory( size_t size )
        {
            return cpu_Memory::MemoryAllocator::instance().allocate<TColorDepth>( size );
        }

        static void _deallocateMemory( TColorDepth * data )
        {
            cpu_Memory::MemoryAllocator::instance().free( data );
        }

        static void _copyMemory( TColorDepth * out, TColorDepth * in, size_t size )
        {
            memcpy( out, in, size );
        }

        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            std::fill( data, data + size / sizeof( TColorDepth ), value );
        }

        // Width of image.
        uint32_t _width{ 0 };

        // Height of image.
        uint32_t _height{ 0 };

        // Number of colors per pixel / number of color channels present.
        uint8_t _colorChannelCount{ 1 };

        // Some image formats require that row size must be a multiple of some value (alignment).
        // For example Bitmap's image alignment must be a multiple of 4.
        uint8_t _alignment{ 1 };

        // Size of single row on image, usually it is equal to width if alignment is 1.
        uint32_t _rowSize{ 0 };

        // Image data.
        TColorDepth * _data{ nullptr };

        // Special attribute to specify different types of images based on technology it is used for.
        // For example, CPU, CUDA, OpenCL and others.
        uint8_t _deviceType{ 0 };

        class FunctionFacade
        {
        public:
            static FunctionFacade & instance()
            {
                static FunctionFacade facade;
                return facade;
            }

            void initialize( uint8_t type, AllocateFunction allocateFunction, DeallocateFunction deallocateFunction, CopyFunction copyFunction, SetFunction setFunction )
            {
                if ( _allocate[type] == nullptr ) {
                    _allocate[type] = allocateFunction;
                    _deallocate[type] = deallocateFunction;
                    _copy[type] = copyFunction;
                    _set[type] = setFunction;
                }
            }

            AllocateFunction allocate( uint8_t type ) const
            {
                return _getFunction( _allocate, type );
            }

            DeallocateFunction deallocate( uint8_t type ) const
            {
                return _getFunction( _deallocate, type );
            }

            CopyFunction copy( uint8_t type ) const
            {
                return _getFunction( _copy, type );
            }

            SetFunction set( uint8_t type )
            {
                return _getFunction( _set, type );
            }

        private:
            FunctionFacade() = default;

            FunctionFacade & operator=( const FunctionFacade & )
            {
                return ( *this );
            }

            FunctionFacade( const FunctionFacade & ) {}

            template <typename TFunction>
            TFunction _getFunction( const std::array<TFunction, 256> & data, uint8_t index ) const
            {
                if ( data[index] == nullptr ) {
                    throw penguinVException( "A function is not defined for this type of image" );
                }

                return data[index];
            }

            std::array<AllocateFunction, 256> _allocate{ nullptr };
            std::array<DeallocateFunction, 256> _deallocate{ nullptr };
            std::array<CopyFunction, 256> _copy{ nullptr };
            std::array<SetFunction, 256> _set{ nullptr };
        };
    };

    using Image = ImageTemplate<uint8_t>;
    using Image16Bit = ImageTemplate<uint16_t>;

    static const uint8_t GRAY_SCALE{ 1u };
    static const uint8_t RGB{ 3u };
    static const uint8_t RGBA{ 4u };
    static const uint8_t RED_CHANNEL{ 0u} ;
    static const uint8_t GREEN_CHANNEL{ 1u };
    static const uint8_t BLUE_CHANNEL{ 2u };
    static const uint8_t ALPHA_CHANNEL{ 3u };
}
