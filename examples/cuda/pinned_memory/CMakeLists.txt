add_executable(cuda_example_pinned_memory
    ${CMAKE_CURRENT_LIST_DIR}/pinned_memory.cpp
    ${LIB_DIR}/cuda/cuda_device.cu
    ${LIB_DIR}/cuda/cuda_helper.cu)
target_include_directories(cuda_example_pinned_memory
    PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_example_pinned_memory
    PRIVATE example_features_options example_features_warnings ${CUDA_LIBRARIES})
