add_executable(example_fft
    ${CMAKE_CURRENT_LIST_DIR}/example_fft.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/filtering.cpp
    ${LIB_DIR}/fft.cpp
    ${LIB_DIR}/thirdparty/kissfft/kiss_fftnd.c
    ${LIB_DIR}/thirdparty/kissfft/kiss_fft.c
    ${LIB_DIR}/file/file_image.cpp
    ${LIB_DIR}/file/bmp_image.cpp
    ${LIB_DIR}/file/jpeg_image.cpp
    ${LIB_DIR}/file/png_image.cpp
    ${LIB_DIR}/math/fft_base.cpp)

target_link_libraries(example_fft
    PRIVATE example_features_options example_features_warnings)
