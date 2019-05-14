#include "file_image.h"
#include "bmp_image.h"
#include "jpeg_image.h"
#include "png_image.h"

namespace File_Operation
{
    PenguinV_Image::Image Load( const std::string & path )
    {
        PenguinV_Image::Image image;

        Load( path, image );
        return image;
    }

    void Load( const std::string & path, PenguinV_Image::Image & image )
    {
#ifdef PENGUINV_ENABLED_JPEG_SUPPORT
        const static std::string jpegFileType = ".jpg";
        if ( path.size() >= jpegFileType.size() && path.compare( path.size() - jpegFileType.size(), jpegFileType.size(), jpegFileType ) == 0 ) {
            Jpeg_Operation::Load( path, image );
            return;
        }
#endif

#ifdef PENGUINV_ENABLED_PNG_SUPPORT
        const static std::string pngFileType = ".png";
        if ( path.size() >= pngFileType.size() && path.compare( path.size() - pngFileType.size(), pngFileType.size(), pngFileType ) == 0 ) {
            Png_Operation::Load( path, image );
            return;
        }
#endif

        Bitmap_Operation::Load( path, image );
    }

    void Save( const std::string & path, const PenguinV_Image::Image & image )
    {
        Save( path, image, 0, 0, image.width(), image.height() );
    }

    void Save( const std::string & path, const PenguinV_Image::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
#ifdef PENGUINV_ENABLED_JPEG_SUPPORT
        const std::string jpegFileType = ".jpg";
        if ( path.size() >= jpegFileType.size() && path.compare( path.size() - jpegFileType.size(), jpegFileType.size(), jpegFileType ) == 0 ) {
            Jpeg_Operation::Save( path, image, startX, startY, width, height );
            return;
        }
#endif

#ifdef PENGUINV_ENABLED_PNG_SUPPORT
        const std::string pngFileType = ".png";
        if ( path.size() >= pngFileType.size() && path.compare( path.size() - pngFileType.size(), pngFileType.size(), pngFileType ) == 0 ) {
            Png_Operation::Save( path, image, startX, startY, width, height );
            return;
        }
#endif

        Bitmap_Operation::Save( path, image, startX, startY, width, height );
    }
}
