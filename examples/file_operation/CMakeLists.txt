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

include(find_png)
if(${PENGUINV_ENABLE_PNG_SUPPORT})
    if(${PENGUINV_USE_EXTERNAL_PNG})
        target_link_libraries(unit_tests PRIVATE ZLIB_EXTERNAL PNG_EXTERNAL)
    else()
        target_link_libraries(unit_tests PRIVATE PNG::PNG)
    endif()
    target_compile_definitions(unit_tests PRIVATE PENGUINV_ENABLED_PNG_SUPPORT)
endif()

include(find_jpeg)
if(${PENGUINV_ENABLE_JPEG_SUPPORT})
    if(${PENGUINV_USE_EXTERNAL_JPEG})
        target_link_libraries(unit_tests PRIVATE JPEG_EXTERNAL)
    else()
        target_link_libraries(unit_tests PRIVATE ${JPEG_LIBRARIES})
    endif()
    target_compile_definitions(unit_tests PRIVATE PENGUINV_ENABLED_JPEG_SUPPORT)
endif()
