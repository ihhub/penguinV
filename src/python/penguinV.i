%module penguinV

%include "stdint.i"
// %include "carrays.i"
%include "std_string.i"

%{
#include "..\image_buffer.h"
#include "..\FileOperation\bitmap.h"
%}

%nodefaultdtor PenguinV_Image::ImageTemplate;
%nodefaultctor PenguinV_Image::ImageTemplate;

namespace PenguinV_Image {

    template<typename T> class ImageTemplate {

        public:

        ImageTemplate(uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u );
        ImageTemplate( const ImageTemplate<T> & image_ );
        ~ImageTemplate();
        void resize(uint32_t width_, uint32_t height_ );
        void clear();
        bool empty() const;
        uint32_t width() const;
        uint32_t height() const;
    };

    // Type definitions aren't passed to wrapper code. We have to tell swig to generate the
    // template instance.

    typedef ImageTemplate<uint8_t> Image; 
    %template(Image) ImageTemplate<uint8_t>; 

}

namespace Bitmap_Operation {

    PenguinV_Image::Image Load ( const std::string & path);
    void Save( const std::string & path, const PenguinV_Image::Image & image );

}
