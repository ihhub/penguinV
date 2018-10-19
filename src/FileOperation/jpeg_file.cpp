#include <stdlib.h>

#ifdef _WIN32
#else
#include <jpeglib.h>
#endif

#include <setjmp>
#include "jpeg_file.h"
#include "../image_exception.h"

namespace Jpeg_Ops {
    PenguinV_Image::Image Load(const std::string & image) {
        if( path.empty() )
            throw imageException( "Incorrect file path for image file loading" );

        FILE * file = fopen( path.data(), "rb" );
        if( !file )
            throw imageException( "Cannot create file for reading" );

        struct jpeg_decompress_struct cInfo;
        struct err_ptr jErr;

        /* Allocate and initialize JPEG decompression object */
        cInfo.err = jpeg_std_error(&jErr.pub);
        jErr.pub.error_exit = custom_jerr_exit; // override default error handler
        if(setjmp(jErr.setjmp_err)){
            jpeg_destroy_decompress(&cInfo);
            fclose(file);
            return PenguinV_Image::Image()
        }

        jpeg_create_decompress(&cInfo);

        /* Specify the source of the compressed data */
        jpeg_stdio_src(&cInfo, file);

        /* Call jpeg_read_header() to obtain image info */
        jpeg_read_header(&cInfo, TRUE);

        /* perform decompression */
        jpeg_start_decompress(&cInfo);

        const uint32_t width = static_cast<uint32_t>(cInfo.image_width);
        const uint32_t height = static_cast<uint32_t>(cInfo.image_height);
        const uint32_t nComponents = static_cast<uint8_t>(cInfo.num_components);

        // TODO: check color space, currently the template supports grayscale and rgb only
        PenguinV_Image::Image image(width, height, PenguinV_Image::RGB);
        uint8_t * dataList = image.data();
        dataList = new uint8_t[cInfo.image_width*cInfo.image_height*cInfo.num_components]

        while(cInfo.output_scanline < cInfo.image_height)
        {
            uint8_t* pix = dataList + cInfo.output_scanline*cInfo.image_width*cInfo.num_components;
            jpeg_read_scanlines(&cInfo, &pix, 1);
        }

        /* finish decompression */
        jpeg_finish_decompress(&cInfo);

        /* release the JPEG decompression object */
        jpeg_destroy_decompress(&cInfo);
        fclose(file);
        return image;
    }

    void Load(const std::string &path, PenguinV_Image::Image &image) {
        image = Load(path);
    }

    void Save(const std::string & path, PenguinV_Image::Image & image) {
        if( path.empty() )
            throw imageException( "Incorrect file path for image file saving" );

        FILE * file = fopen( path.data(), "wb" );
        if( !file )
            throw imageException( "Cannot create file for writing" );

        struct jpeg_compress_struct cInfo;
        struct err_ptr jErr;
        cInfo.err = jpeg_std_error(&jErr);

        jpeg_create_compress(&cInfo);
        jpeg_stdio_dest(&cInfo, file);

        cInfo.image_width = image.width();
        cInfo.image_height = image.height();
        cInfo.input_components = 3;
        cInfo.in_color_space = JCS_RGB; // use RGB color space

        jpeg_set_defaults(&cInfo);
        jpeg_start_compress(&cInfo, TRUE);
        uint8_t * dataToWrite = image.data();

        while (cInfo.next_scanline < cInfo.image_height) {
            uint8_t * dt = dataToWrite[cInfo.next_scanline * cInfo.image_width * 3];
            jpeg_write_scanlines(&cInfo, dt, 1);
        }

        jpeg_finish_decompress(&cInfo); /* finish decompression */
        jpeg_destroy_decompress(&cInfo); /* release the JPEG decompression object */
        fclose(file);
        return;
    }

    static void custom_jerr_exit(j_common_ptr info) {
        err_ptr err = (err_ptr) info->err;
        (*info->err->output_message) (info);
        longjmp(err->setjmp_buffer, 1);
    }
}
