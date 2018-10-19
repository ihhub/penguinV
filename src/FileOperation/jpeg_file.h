#include "../image_buffer.h"

namespace Jpeg_Ops {
    // load a jpeg image
    PenguinV_Image::Image Load(const std::string & path);
    void Load(const std::string & path, PenguinV_Image::Image & image);

    // save a jpeg image
    void Save(const std::string & path, PenguinV_Image::Image & image);
    void Save(const std::string & path, PenguinV_Image::Image & image, uint32_t startX, uint32_t startY,
            uint32_t width, uint32_t height);

    // error helpers
    struct penguinv_err_mgr {
        struct jpeg_error_mgr pub;	/* "public" fields */
        jmp_buf setjmp_buffer;	/* for return to caller */
    };
    typedef struct penguinv_err_mgr * err_ptr;
    static void custom_jerr_exit(j_common_ptr)
}