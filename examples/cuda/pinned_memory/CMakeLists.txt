cuda_add_executable(cuda_example_pinned_memory
    ${CMAKE_CURRENT_LIST_DIR}/pinned_memory.cpp
    ${LIB_DIR}/cuda/cuda_device.cu
    ${LIB_DIR}/cuda/cuda_helper.cu)
set_target_properties(cuda_example_image_function
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_example_pinned_memory
    PRIVATE example_features_options example_features_warnings)
