add_executable(example_opencl_image_function
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${LIB_DIR}/opencl/opencl_device.cpp
    ${LIB_DIR}/opencl/opencl_helper.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/file/bmp_image.cpp
    ${LIB_DIR}/opencl/image_function_opencl.cpp)
target_compile_definitions(example_opencl_image_function
    PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(example_opencl_image_function
    PRIVATE example_features_options example_features_warnings OpenCL::OpenCL)
