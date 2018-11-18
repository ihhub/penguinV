add_executable(example_bitmap_operation
    ${CMAKE_CURRENT_LIST_DIR}/example_bitmap_operation.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/FileOperation/bitmap.cpp)
target_link_libraries(example_bitmap_operation
    PRIVATE example_features_options example_features_warnings)