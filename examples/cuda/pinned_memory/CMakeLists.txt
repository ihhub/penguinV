add_executable(example_image_function
    ${CMAKE_CURRENT_LIST_DIR}/example_image_function.cpp
    ${LIB_DIR}/cuda/image_function_cuda.cu
    ${LIB_DIR}/cuda/cuda_device.cu
    ${LIB_DIR}/cuda/cuda_helper.cu
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/file/bmp_image.cpp
target_link_libraries(example_image_function
    PRIVATE example_features_options example_features_warnings)

if (WIN32)
    # Quiet warnings related to fopen on MSVC
    target_compile_definitions(example_image_function PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
