add_executable(example_file_operation
    ${CMAKE_CURRENT_LIST_DIR}/example_file_operation.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/file/file_image.cpp
    ${LIB_DIR}/file/bmp_image.cpp
    ${LIB_DIR}/file/png_image.cpp
    ${LIB_DIR}/file/jpeg_image.cpp)
target_link_libraries(example_file_operation
    PRIVATE example_features_options example_features_warnings)

if (WIN32)
    # Quiet warnings related to fopen on MSVC
    target_compile_definitions(example_file_operation PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
