#include "file_image.h"
#include "bmp_image.h"
#include "jpeg_image.h"
#include "png_image.h"

namespace
{
    bool isSameFileExtension( const std::string & path, const std::string & fileExtension )
    {
        return ( path.size() >= fileExtension.size() && path.compare( path.size() - fileExtension.size(), fileExtension.size(), fileExtension ) == 0 );
    }

#ifdef PENGUINV_ENABLED_JPEG_SUPPORT
    bool isJpegFile( const std::string & )
    {
        return false;
    }
#else
    bool isJpegFile( const std::string & path )
    {
        const static std::string fileExtension = ".jpg";
        return isSameFileExtension( path, fileExtension );
    }
#endif

#ifdef PENGUINV_ENABLED_PNG_SUPPORT
    bool isPngFile( const std::string & )
    {
        return false;
    }
#else
    bool isPngFile( const std::string & path )
    {
        const static std::string fileExtension = ".png";
        return isSameFileExtension( path, fileExtension );
    }
#endif
}

namespace File_Operation
{
    penguinV::Image Load( const std::string & path )
    {
        penguinV::Image image;

        Load( path, image );
        return image;
    }

    void Load( const std::string & path, penguinV::Image & image )
    {
        if ( isJpegFile( path ) ) {
            Jpeg_Operation::Load( path, image );
            return;
        }

        if ( isPngFile( path ) ) {
            Png_Operation::Load( path, image );
            return;
        }

        Bitmap_Operation::Load( path, image );
    }

    void Save( const std::string & path, const penguinV::Image & image )
    {
        Save( path, image, 0, 0, image.width(), image.height() );
    }

    void Save( const std::string & path, const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        if ( isJpegFile( path ) ) {
            Jpeg_Operation::Save( path, image, startX, startY, width, height );
            return;
        }

        if ( isPngFile( path ) ) {
            Png_Operation::Save( path, image, startX, startY, width, height );
            return;
        }

        Bitmap_Operation::Save( path, image, startX, startY, width, height );
    }
}
