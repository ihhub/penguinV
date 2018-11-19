add_executable(example_thread_pool
    ${CMAKE_CURRENT_LIST_DIR}/example_thread_pool.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/thread_pool.cpp
    ${LIB_DIR}/penguinv/penguinv.cpp)
target_link_libraries(example_thread_pool
    PRIVATE example_features_options example_features_warnings Threads::Threads)
