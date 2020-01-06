set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test)
set(SRCS
    ${LIB_DIR}/file/bmp_image.cpp
    ${LIB_DIR}/file/jpeg_image.cpp
    ${LIB_DIR}/file/png_image.cpp
    ${LIB_DIR}/math/haar_transform.cpp
    ${LIB_DIR}/math/hough_transform.cpp
    ${LIB_DIR}/math/math_base.cpp
    ${LIB_DIR}/penguinv/penguinv.cpp
    ${LIB_DIR}/thirdparty/kissfft/kiss_fft.c
    ${LIB_DIR}/thirdparty/kissfft/kiss_fftnd.c
    ${LIB_DIR}/blob_detection.cpp
    ${LIB_DIR}/edge_detection.cpp
    ${LIB_DIR}/filtering.cpp
    ${LIB_DIR}/math/fft_base.cpp
    ${LIB_DIR}/fft.cpp
    ${LIB_DIR}/function_pool.cpp
    ${LIB_DIR}/function_pool_task.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/image_function_simd.cpp
    ${LIB_DIR}/thread_pool.cpp
    ${TEST_DIR}/test_helper.cpp
    unit_tests.cpp
    unit_test_blob_detection.cpp
    unit_test_edge_detection.cpp
    unit_test_fft.cpp
    unit_test_file.cpp
    unit_test_framework.cpp
    unit_test_helper.cpp
    unit_test_image_buffer.cpp
    unit_test_image_function.cpp
    unit_test_math.cpp
)

option(PENGUINV_BUILD_UNIT_TEST_CUDA "Build CUDA unit tests" ON)
    if(${PENGUINV_BUILD_UNIT_TEST_CUDA})
        if(CUDA_FOUND)
            add_subdirectory(cuda)
        endif(CUDA_FOUND)
endif()

option(PENGUINV_BUILD_UNIT_TEST_OPENCL "Build OpenCL unit tests" ON)
if(${PENGUINV_BUILD_UNIT_TEST_OPENCL})
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
target_link_libraries(unit_tests PRIVATE unit_tests_options unit_tests_warnings Threads::Threads)

include(find_png)
if(${PENGUINV_ENABLE_PNG_SUPPORT})
    if(${PENGUINV_USE_EXTERNAL_PNG})
        target_link_libraries(unit_tests PRIVATE ZLIB_EXTERNAL PNG_EXTERNAL)
    else()
        target_link_libraries(unit_tests PRIVATE PNG::PNG)
    endif()
    target_compile_definitions(unit_tests PRIVATE PENGUINV_ENABLED_PNG_SUPPORT)
endif()

include(find_jpeg)
if(${PENGUINV_ENABLE_JPEG_SUPPORT})
    if(${PENGUINV_USE_EXTERNAL_JPEG})
        target_link_libraries(unit_tests PRIVATE JPEG_EXTERNAL)
    else()
        target_link_libraries(unit_tests PRIVATE ${JPEG_LIBRARIES})
    endif()
    target_compile_definitions(unit_tests PRIVATE PENGUINV_ENABLED_JPEG_SUPPORT)
endif()

if (WIN32)
    # Quiet warnings related to fopen on MSVC
    target_compile_definitions(unit_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_test(NAME unit_test COMMAND unit_tests $ENV{PENGUINV_UNIT_TEST_RUN_COUNT})
