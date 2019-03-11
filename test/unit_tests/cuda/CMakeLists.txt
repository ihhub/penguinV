set(MAIN_LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SRCS
    ${MAIN_LIB_DIR}/cuda/image_function_cuda.cu
    ${MAIN_LIB_DIR}/cuda/cuda_device.cu
    ${MAIN_LIB_DIR}/cuda/cuda_helper.cu
    ${MAIN_LIB_DIR}/image_function.cpp
    ${MAIN_LIB_DIR}/image_function_helper.cpp
    ${UNIT_TEST_DIR}/unit_test_framework.cpp
    ${UNIT_TEST_DIR}/unit_test_helper.cpp
    ${UNIT_TEST_DIR}/../test_helper.cpp
    unit_test_helper_cuda.cu
    unit_tests_cuda.cpp
    unit_test_image_function_cuda.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(TARGET_NAME unit_tests_cuda)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
cuda_add_executable(${TARGET_NAME} ${SRCS})
if (NOT MSVC)
    target_compile_options(${TARGET_NAME} INTERFACE -Wno-deprecated-gpu-targets)
endif()
add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
