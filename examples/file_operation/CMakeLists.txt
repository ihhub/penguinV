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

if(${PENGUINV_ENABLE_PNG_SUPPORT})
    if(${PENGUINV_USE_EXTERNAL_PNG})
        target_link_libraries(example_file_operation PRIVATE ZLIB_EXTERNAL PNG_EXTERNAL)
    else()
        target_link_libraries(example_file_operation PRIVATE PNG::PNG)
    endif()
    target_compile_definitions(example_file_operation PRIVATE PENGUINV_ENABLED_PNG_SUPPORT)
endif()

if(${PENGUINV_ENABLE_JPEG_SUPPORT})
    if(${PENGUINV_USE_EXTERNAL_JPEG})
        target_link_libraries(example_file_operation PRIVATE JPEG_EXTERNAL)
    else()
        target_link_libraries(example_file_operation PRIVATE ${JPEG_LIBRARIES})
    endif()
    target_compile_definitions(example_file_operation PRIVATE PENGUINV_ENABLED_JPEG_SUPPORT)
endif()

if (WIN32)
    # Quiet warnings related to fopen on MSVC
    target_compile_definitions(example_file_operation PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
