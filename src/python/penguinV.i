%module penguinV

%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "..\image_buffer.h"
#include "..\FileOperation\bitmap.h"
#include "..\image_function.h"
%}

%nodefaultctor PenguinV_Image::ImageTemplate;
%nodefaultctor Image;

namespace std {
    %template(vectorUInt32) vector<uint32_t>;
};

namespace PenguinV_Image {

    template<typename TColorDepth> class ImageTemplate {

        public:

        ImageTemplate(uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u );
        ImageTemplate( const ImageTemplate<TColorDepth> & image_ );
        ~ImageTemplate();
        void resize(uint32_t width_, uint32_t height_ );
        void clear();
        bool empty() const;
        uint32_t width() const;
        uint32_t height() const;
        uint32_t rowSize() const;
        uint8_t colorCount() const;
        void setColorCount( uint8_t colorCount_ );
        uint8_t alignment() const;
        void setAlignment( uint8_t alignment_ );
        void fill( TColorDepth value );
        void swap( ImageTemplate & image );
        void copy( const ImageTemplate & image );
        bool mutate( uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ );
        uint8_t type() const;
        ImageTemplate generate( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u ) const;

    };

    // Type definitions aren't passed to wrapper code. We have to tell swig to generate the
    // template instance.

    typedef ImageTemplate<uint8_t> Image; 
    %feature("autodoc", "1");
    %template(Image) ImageTemplate<uint8_t>; 

    const uint8_t GRAY_SCALE;
    const uint8_t RGB;
    const uint8_t RGBA;
}

namespace Bitmap_Operation {

    PenguinV_Image::Image Load ( const std::string & path);
    void Save( const std::string & path, const PenguinV_Image::Image & image );

}

namespace Image_Function {

   using namespace PenguinV_Image;

   Image ConvertToGrayScale( const Image & in); 

   uint8_t GetThreshold( const std::vector < uint32_t > & histogram );

   std::vector < uint32_t > Histogram( const Image & image );

   Image Threshold( const Image & in, uint8_t threshold );

}

// For custom exceptions in ..\image_exception.h, it is easier to just manually redeclare the custom exceptions in python.

%pythoncode %{

class PenguinV_Error(Exception):
    '''Base class for errors/exceptions in penguinV.'''
    pass

class ImageException(PenguinV_Error):
    ''' Exceptions raised by image operations. 
    Attributes:
    expression - input expression in which the error occurred.
    message - explanation of the error.
    '''
    def __init__(self, expression, error):
        self.expression = expression
        self.error = error 

%} 
