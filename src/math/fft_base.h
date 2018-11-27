#pragma once

#include <vector>
#include "../image_exception.h"

namespace FFT 
{
    // This class template is the base for storing complex-valued data ([real, imaginary]) for
    // the purpose of using in a Fast Fourier Transform. The template parameter is the type of 
    // data to use. Instances of this class template are meant to be inherited by a sub-class.
    // The sub-class should define _allocateData(), _freeData(), and _copyData(). 
    //
    // The function template definitions are in fft_base.tpp, and this file is included at the 
    // end of this file (after the class template declarations).

    template <typename DataType>
    class BaseComplexData 
    {
    public:
        BaseComplexData();

        BaseComplexData<DataType> & operator=( const BaseComplexData<DataType> & data );
        BaseComplexData<DataType> & operator=( BaseComplexData<DataType> && data );

        void resize( uint32_t width_, uint32_t height_ );

        DataType * data(); // returns a pointer to data
        const DataType * data() const;
        bool empty() const; // returns true is array is empty (unallocated)
        uint32_t width() const
        {
            return _width;
        }
        uint32_t height() const
        {
            return _height;
        }
        bool dimensionsMatch( const BaseComplexData<DataType> & other) const;
        bool dimensionsMatch( uint32_t width, uint32_t height) const;

    protected:
        DataType * _data;
        uint32_t _width;
        uint32_t _height;

        virtual void _allocateData(size_t size) = 0; // Override to deal with actual memory allocation
        virtual void _freeData() = 0; // Override to deal with actual memory freeing
        virtual void _copyData(const BaseComplexData<DataType> & data) = 0; // Override to deal with actual data copying

        void _clean();
        void _copy( const BaseComplexData<DataType> & data);
        void _swap( BaseComplexData<DataType> & data );
    };

    // The base class fft execution. Sub-classes need to implement the direct and inverse
    // transformations. 
    // - conversion from original domain of data to frequency domain and vice versa
    // - complex multiplication in frequency domain (convolution)

    class BaseFFTExecutor 
    {
    public:
        BaseFFTExecutor();
        BaseFFTExecutor( uint32_t width_, uint32_t height_ );

        void initialize( uint32_t width_, uint32_t height_ ); // Calls virtual functions

        uint32_t width() const
        {
            return _width;
        }  
        uint32_t height() const
        {
            return _height;
        }
        bool dimensionsMatch(const BaseFFTExecutor & other) const;
        bool dimensionsMatch(uint32_t width, uint32_t height) const;

    protected:
        uint32_t _width;
        uint32_t _height;

        void _clean();

        virtual void _makePlans() = 0;
        virtual void _cleanPlans() = 0;
    }; 

    // Class Template Member Function Definitions 
    template <typename DataType>
    BaseComplexData<DataType>::BaseComplexData()
        : _data  ( nullptr )
        , _width (0)
        , _height(0)
    {
    }

    template <typename DataType>
    BaseComplexData<DataType> & BaseComplexData<DataType>::operator=(const BaseComplexData<DataType> & data)
    {
        _copy( data );

        return *this;
    }

    template <typename DataType>
    BaseComplexData<DataType> & BaseComplexData<DataType>::operator= (BaseComplexData<DataType> && data)
    {
        _swap( data );

        return *this;
    }

    template <typename DataType>
    void BaseComplexData<DataType>::resize(uint32_t width_, uint32_t height_ )
    {
        if( !dimensionsMatch(width_, height_) && width_ != 0 && height_ != 0 ) {
            _clean();

            const uint32_t size = width_ * height_;

            _allocateData(size * sizeof(DataType));

            _width = width_;
            _height = height_;
        }
    }

    template <typename DataType>
    DataType * BaseComplexData<DataType>::data()
    {
        return _data;
    }

    template <typename DataType>
    const DataType * BaseComplexData<DataType>::data() const
    {
        return _data;
    }

    template <typename DataType>
    bool BaseComplexData<DataType>::empty() const
    {
        return _data == nullptr;
    }

    template <typename DataType>
    bool BaseComplexData<DataType>::dimensionsMatch( const BaseComplexData<DataType> & other) const 
    {
        return dimensionsMatch(other.width(), other.height());
    }

    template <typename DataType>
    bool BaseComplexData<DataType>::dimensionsMatch( uint32_t width, uint32_t height) const 
    {
        return _width == width && _height == height;
    }

    template <typename DataType>
    void BaseComplexData<DataType>::_clean() 
    {
        if( _data != nullptr ) {
            _freeData();
            _data = nullptr;
        }

        _width = 0;
        _height = 0;
    }

    template <typename DataType>
    void BaseComplexData<DataType>::_copy( const BaseComplexData<DataType> & data)
    {
        _clean();

        resize(data._width, data._height);

        if( !empty() )
            _copyData(data);
    }
    
    template <typename DataType>
    void BaseComplexData<DataType>::_swap( BaseComplexData<DataType> & data )
    {
        std::swap( _data  , data._data );
        std::swap( _width, data._width);
        std::swap( _height, data._height); 
    }
}
