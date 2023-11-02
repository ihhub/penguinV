if(MSVC)
    add_executable(example_edge_detection
        ${CMAKE_CURRENT_LIST_DIR}/example_edge_detection.cpp
        ${LIB_DIR}/file/bmp_image.cpp
        ${LIB_DIR}/ui/ui.cpp
        ${LIB_DIR}/ui/win/win_ui.cpp
        ${LIB_DIR}/image_function_helper.cpp
        ${LIB_DIR}/image_function.cpp
        ${LIB_DIR}/edge_detection.cpp)
    target_link_libraries(example_edge_detection
        PRIVATE example_features_options example_features_warnings Threads::Threads)
elseif(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    add_executable(example_edge_detection
        ${CMAKE_CURRENT_LIST_DIR}/example_edge_detection.cpp
        ${LIB_DIR}/file/bmp_image.cpp
        ${LIB_DIR}/ui/ui.cpp
        ${LIB_DIR}/ui/x11/x11_ui.cpp
        ${LIB_DIR}/image_function_helper.cpp
        ${LIB_DIR}/image_function.cpp
        ${LIB_DIR}/edge_detection.cpp)
    target_link_libraries(example_edge_detection
        PRIVATE example_features_options example_features_warnings Threads::Threads ${X11_LIBRARIES})
endif()

# Copy example data to binary directory.
copy_data_files()
