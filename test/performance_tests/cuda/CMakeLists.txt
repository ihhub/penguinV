set(MAIN_LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(PERFORMANCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SRCS
    ${MAIN_LIB_DIR}/cuda/image_function_cuda.cu
    ${MAIN_LIB_DIR}/cuda/cuda_device.cu
    ${MAIN_LIB_DIR}/cuda/cuda_helper.cu
    ${MAIN_LIB_DIR}/image_function_helper.cpp
    ${PERFORMANCE_TEST_DIR}/performance_test_framework.cpp
    ${PERFORMANCE_TEST_DIR}/performance_test_helper.cpp
    ${PERFORMANCE_TEST_DIR}/../test_helper.cpp
    performance_tests_cuda.cpp
    performance_test_helper_cuda.cu
    performance_test_image_function_cuda.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(TARGET_NAME performance_tests_cuda)

cuda_add_executable(${TARGET_NAME} ${SRCS})
if (NOT MSVC)
    target_compile_options(${TARGET_NAME} INTERFACE -Wno-deprecated-gpu-targets)
endif()
add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
