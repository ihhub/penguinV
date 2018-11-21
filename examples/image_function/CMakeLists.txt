add_executable(example_image_function
    ${CMAKE_CURRENT_LIST_DIR}/example_image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp)
target_link_libraries(example_image_function
    PRIVATE example_features_options example_features_warnings)
