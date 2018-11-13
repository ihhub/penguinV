#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#else
#include <jpeglib.h>
#endif

#include <setjmp.h>
#include "jpeg_file.h"
#include "../image_exception.h"

namespace
{
    struct penguinv_err_mgr
    {
        struct jpeg_error_mgr pub; // "public" fields
        jmp_buf setjmp_buffer;     // for return to caller
    };
    typedef struct penguinv_err_mgr *err_ptr;

    void custom_jerr_exit(j_common_ptr info)
    {
        err_ptr err = (err_ptr)info->err;
        (*info->err->output_message)(info);
        longjmp(err->setjmp_buffer, 1);
    }
}

namespace Jpeg_Operation
{
    PenguinV_Image::Image Load(const std::string & path)
    {
        if (path.empty())
            throw imageException("Incorrect file path for image file loading");

        FILE * file = fopen(path.data(), "rb");
        if (!file)
            throw imageException("Cannot create file for reading");

        struct jpeg_decompress_struct cInfo;
        err_ptr jErr;

        /* Allocate and initialize JPEG decompression object */
        cInfo.err = jpeg_std_error(&jErr->pub);
        jErr->pub.error_exit = custom_jerr_exit; // override default error handler
        if (setjmp(jErr->setjmp_buffer)) {
            jpeg_destroy_decompress(&cInfo);
            fclose(file);
            return PenguinV_Image::Image();
        }

        jpeg_create_decompress(&cInfo);

        // Specify the source of the compressed data
        jpeg_stdio_src(&cInfo, file);

        // Call jpeg_read_header() to obtain image info
        jpeg_read_header(&cInfo, TRUE);

        // perform decompression
        jpeg_start_decompress(&cInfo);

        const uint32_t width     = static_cast<uint32_t>(cInfo.image_width);
        const uint32_t height    = static_cast<uint32_t>(cInfo.image_height);
        const uint8_t colorCount = static_cast<uint8_t >(cInfo.num_components);

        // TODO: check color space, currently the template supports grayscale and rgb only
        PenguinV_Image::Image image(width, height, colorCount);
        uint8_t * imageData = image.data();
        const uint8_t * endData = imageData + image.rowSize() * image.height();

        for (; imageData != endData; imageData += image.rowSize())
            jpeg_read_scanlines(&cInfo, (JSAMPARRAY)imageData, 1);

        // finish decompression
        jpeg_finish_decompress(&cInfo);

        // release the JPEG decompression object
        jpeg_destroy_decompress(&cInfo);
        fclose(file);
        return image;
    }

    void Load(const std::string & path, PenguinV_Image::Image & image)
    {
        image = Load(path);
    }

    void Save(const std::string & path, const PenguinV_Image::Image & image)
    {
        if (path.empty())
            throw imageException("Incorrect file path for image file saving");

        FILE * file = fopen(path.data(), "wb");
        if (!file)
            throw imageException("Cannot create file for writing");

        struct jpeg_compress_struct cInfo;
        err_ptr jErr;
        cInfo.err = jpeg_std_error(&jErr->pub);

        jpeg_create_compress(&cInfo);
        jpeg_stdio_dest(&cInfo, file);

        cInfo.image_width = image.width();
        cInfo.image_height = image.height();
        cInfo.input_components = image.colorCount();
        cInfo.in_color_space = JCS_RGB; // use RGB color space

        jpeg_set_defaults(&cInfo);
        jpeg_start_compress(&cInfo, TRUE);

        const uint8_t * imageData = image.data();
        const uint8_t * endData = imageData + image.rowSize() * image.height();

        for (; imageData != endData; imageData += image.rowSize())
            jpeg_write_scanlines(&cInfo, (JSAMPARRAY)imageData, 1);

        jpeg_finish_compress(&cInfo);  // finish decompression
        jpeg_destroy_compress(&cInfo); // release the JPEG decompression object
        fclose(file);
    }
}
