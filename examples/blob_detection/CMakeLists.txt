add_executable(example_blob_detection
    ${CMAKE_CURRENT_LIST_DIR}/example_blob_detection.cpp
    ${LIB_DIR}/blob_detection.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/FileOperation/bitmap.cpp)
target_link_libraries(example_blob_detection
    PRIVATE example_features_options example_features_warnings)
