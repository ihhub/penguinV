add_executable(example_filters
    ${CMAKE_CURRENT_LIST_DIR}/example_filters.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/filtering.cpp
    ${LIB_DIR}/file/bmp_image.cpp)
target_link_libraries(example_filters
    PRIVATE example_features_options example_features_warnings)

#copy example data to binary directory
copy_data_files()