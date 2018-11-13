set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRCS
    ${LIB_DIR}/FileOperation/bitmap.cpp
    ${LIB_DIR}/FileOperation/jpeg_file.cpp
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
    unit_test_math.cpp
    unit_test_jpeg_ops.cpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(JPEG REQUIRED)
if(NOT JPEG_FOUND)
    if (WIN32)
        set(JPEG_BINARY_DIR       ${CMAKE_CURRENT_BINARY_DIR}/jpeg/src/jpeg)
        set(JPEG_INSTALL_DIR      ${CMAKE_CURRENT_BINARY_DIR}/jpeg/install)
        set(JPEG_STATIC_LIBRARIES ${JPEG_INSTALL_DIR}/lib/libjpeg.lib)
        include(ExternalProject)
        ExternalProject_Add(jpeg
            PREFIX jpeg
            URL "http://sourceforge.net/projects/libjpeg-turbo/files/2.0.1/libjpeg-turbo-2.0.1.tar.gz"
            URL_MD5 1b05a66aa9b006fd04ed29f408e68f46
            BUILD_BYPRODUCTS ${JPEG_STATIC_LIBRARIES}
            PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/jpeg/CMakeLists.txt ${JPEG_BINARY_DIR}
            BINARY_DIR ${JPEG_BINARY_DIR}
            INSTALL_DIR ${JPEG_INSTALL_DIR}
            DOWNLOAD_DIR ${DOWNLOAD_LOCATION}
            CMAKE_CACHE_ARGS
                -DCMAKE_BUILD_TYPE:STRING=Release
                -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
                -DCMAKE_INSTALL_PREFIX:STRING=${JPEG_INSTALL_DIR})
    endif(WIN32)
endif()

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
                              Threads::Threads
                              ${JPEG_LIBRARIES})

add_test(NAME unit_test COMMAND unit_tests $ENV{PENGUINV_UNIT_TEST_RUN_COUNT})
