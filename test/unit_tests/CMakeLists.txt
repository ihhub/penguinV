set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRCS
    ${LIB_DIR}/FileOperation/bitmap.cpp
    ${LIB_DIR}/math/hough_transform.cpp
    ${LIB_DIR}/math/math_base.cpp
    ${LIB_DIR}/penguinv/penguinv.cpp
    ${LIB_DIR}/thirdparty/kissfft/kiss_fft.c
    ${LIB_DIR}/thirdparty/kissfft/kiss_fftnd.c
    ${LIB_DIR}/blob_detection.cpp
    ${LIB_DIR}/filtering.cpp
    ${LIB_DIR}/fft.cpp
    ${LIB_DIR}/function_pool.cpp
    ${LIB_DIR}/function_pool_task.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function_simd.cpp
    ${LIB_DIR}/thread_pool.cpp
    unit_tests.cpp
    unit_test_bitmap.cpp
    unit_test_blob_detection.cpp
    unit_test_fft.cpp
    unit_test_framework.cpp
    unit_test_helper.cpp
    unit_test_image_buffer.cpp
    unit_test_image_function.cpp
    unit_test_math.cpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

option(PENGUINV_BUILD_UNIT_TEST_OPENCL "Build OpenCL unit tests" ON)
if(${PENGUINV_BUILD_UNIT_TEST_OPENCL})
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        add_subdirectory(opencl)
    endif(OpenCL_FOUND)
endif()

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(unit_tests_options INTERFACE)
target_compile_features(unit_tests_options INTERFACE cxx_std_11)

# Link this 'library' to use the standard warnings
add_library(unit_tests_warnings INTERFACE)
if(MSVC)
    target_compile_options(unit_tests_warnings 
                           INTERFACE /W4 /WX)
else()
    target_compile_options(unit_tests_warnings
                           INTERFACE -Wall
                                     -Wextra           # reasonable and standard
                                     -Wstrict-aliasing # warn the user if a variable declaration shadows one
                                     -Wpedantic        # warn if non-standard C++ is used
                                     -Wconversion
                                     -Wsign-conversion
                                     -O2
                                     -march=native)
endif()

add_executable(unit_tests ${SRCS})
target_link_libraries(unit_tests
                      PRIVATE unit_tests_options
                              unit_tests_warnings
                              Threads::Threads)
add_test(NAME unit_test COMMAND unit_tests $ENV{UNIT_TEST_RUN_COUNT})
endif()
