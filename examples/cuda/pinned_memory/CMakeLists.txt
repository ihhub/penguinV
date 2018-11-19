add_executable(cuda_example_pinned_memory
    ${CMAKE_CURRENT_LIST_DIR}/pinned_memory.cpp
    ${LIB_DIR}/cuda/cuda_device.cu
    ${LIB_DIR}/cuda/cuda_helper.cu
    ${LIB_DIR}/cuda/image_buffer_cuda.cuh
    ${LIB_DIR}/cuda/image_buffer_cuda_pinned.cuh
    ${LIB_DIR}/cuda/cuda_device.cuh
    ${LIB_DIR}/cuda/cuda_helper.cuh
    ${LIB_DIR}/cuda/cuda_memory.cuh
    ${LIB_DIR}/cuda/cuda_types.cuh)
set_target_properties(cuda_example_image_function
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_example_pinned_memory
    PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_example_pinned_memory
    PRIVATE example_features_options example_features_warnings ${CUDA_LIBRARIES})
