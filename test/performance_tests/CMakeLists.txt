set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test)
set(SRCS
    ${LIB_DIR}/math/math_base.cpp
	${LIB_DIR}/blob_detection.cpp
    ${LIB_DIR}/filtering.cpp
    ${LIB_DIR}/function_pool.cpp
    ${LIB_DIR}/function_pool_task.cpp
    ${LIB_DIR}/thread_pool.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function_simd.cpp
    ${LIB_DIR}/penguinv/penguinv.cpp
    ${TEST_DIR}/test_helper.cpp
    performance_tests.cpp
    performance_test_blob_detection.cpp
    performance_test_filtering.cpp
    performance_test_framework.cpp
    performance_test_helper.cpp
    performance_test_image_function.cpp)

option(PENGUINV_BUILD_UNIT_TEST_OPENCL "Build OpenCL performance tests" ON)
if(${PENGUINV_BUILD_UNIT_TEST_OPENCL})
    if(OpenCL_FOUND)
        add_subdirectory(opencl)
    endif(OpenCL_FOUND)
endif()

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(performance_tests_options INTERFACE)
target_compile_features(performance_tests_options INTERFACE cxx_std_11)

# Link this 'library' to use the standard warnings
add_library(performance_tests_warnings INTERFACE)
if(MSVC)
    target_compile_options(performance_tests_warnings 
                           INTERFACE /W4 /WX)
else()
    target_compile_options(performance_tests_warnings
                           INTERFACE -Wall
                                     -Wextra           # reasonable and standard
                                     -Wstrict-aliasing # warn the user if a variable declaration shadows one
                                     -Wpedantic        # warn if non-standard C++ is used
                                     -Wconversion
                                     -Wsign-conversion
                                     -O2
                                     -march=native)
endif()


add_executable(performance_tests ${SRCS})
target_link_libraries(performance_tests 
                      PRIVATE performance_tests_options
                              performance_tests_warnings
                              Threads::Threads)
add_test(NAME perf_test COMMAND performance_tests)
