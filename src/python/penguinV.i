%module penguinV

%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "../image_buffer.h"
#include "../file/bmp_image.h"
#include "../image_function.h"
%}

%nodefaultctor penguinV::ImageTemplate;
%nodefaultctor Image;

%feature("autodoc", "1");

namespace std {
    %template(vectorUInt32) vector<uint32_t>;
};

namespace penguinV {

    template<typename TColorDepth> class ImageTemplate {

        public:

        ImageTemplate(uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u );
        ImageTemplate( const ImageTemplate<TColorDepth> & image_ );
        // Missing ImageTemplate( ImageTemplate && image) as moving constructors don't make sense in python.
        // Missing overloading of operator= as it doesn't make sense in python as everything in python is just references.
        ~ImageTemplate(); //originally virtual, but every member function is virtual in python.

        void resize(uint32_t width_, uint32_t height_ );
        void clear();

        TColorDepth * data(); // This doesn't wrap to return an array or list in python. The pointer is just wrapped as
                              // an object that can't be manipulated directly in python, but it can be used with penguinV
                              // functions such as ImageTemplate::assign().

        void assign( TColorDepth * data_, uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ );

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
    %template(Image) ImageTemplate<uint8_t>;

    const uint8_t GRAY_SCALE;
    const uint8_t RGB;
    const uint8_t RGBA;
}

namespace Bitmap_Operation {

    penguinV::Image Load ( const std::string & path);
    void            Load( const std::string & path, penguinV::Image & image );

    void Save( const std::string & path, const penguinV::Image & image );
    void Save( const std::string & path, const penguinV::Image & image, uint32_t startX, uint32_t startY,
               uint32_t width, uint32_t height );
}

namespace Image_Function {

   using namespace penguinV;

   Image ConvertToGrayScale( const Image & in );
   void  ConvertToGrayScale( const Image & in, Image & out );
   Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
   void  ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height );

   uint8_t GetThreshold( const std::vector < uint32_t > & histogram );

   std::vector < uint32_t > Histogram( const Image & image );
   void                     Histogram( const Image & image, std::vector < uint32_t > & histogram );
   std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );
   void                     Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                       std::vector < uint32_t > & histogram );
   Image Threshold( const Image & in, uint8_t threshold );
   void  Threshold( const Image & in, Image & out, uint8_t threshold );
   Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold );
   void  Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold );

}

// For custom exceptions in ../image_exception.h, it is easier to just manually redeclare the custom exceptions in python.

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
