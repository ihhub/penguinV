add_executable(cuda_example_image_function
    ${CMAKE_CURRENT_LIST_DIR}/example_image_function.cpp
    ${LIB_DIR}/cuda/image_function_cuda.cu
    ${LIB_DIR}/cuda/cuda_device.cu
    ${LIB_DIR}/cuda/cuda_helper.cu)
set_target_properties(cuda_example_image_function
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_example_image_function
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_example_image_function
    PRIVATE example_features_options example_features_warnings)
