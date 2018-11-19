add_executable(example_function_pool
    ${CMAKE_CURRENT_LIST_DIR}/example_function_pool.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_simd.cpp
    ${LIB_DIR}/thread_pool.cpp
    ${LIB_DIR}/function_pool_task.cpp
    ${LIB_DIR}/function_pool.cpp
    ${LIB_DIR}/penguinv/penguinv.cpp)
target_link_libraries(example_function_pool
    PRIVATE example_features_options example_features_warnings Threads::Threads)
