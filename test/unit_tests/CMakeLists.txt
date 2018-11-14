set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRCS
    ${LIB_DIR}/FileOperation/bitmap.cpp
    ${LIB_DIR}/FileOperation/png_file.cpp
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

if(WIN32)
    # Location where external projects will be downloaded
    set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
        CACHE PATH "Location where external projects will be downloaded.")
    mark_as_advanced(DOWNLOAD_LOCATION)
    include(ExternalProject)
    set(ZLIB_INSTALL ${CMAKE_BINARY_DIR}/zlib)
    set(ZLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/zlib/build)
    set(ZLIB_INCLUDE_DIR ${ZLIB_INSTALL}/include)
    set(ZLIB_STATIC_LIBRARIES
            debug ${ZLIB_INSTALL}/lib/zlibstaticd.lib
            optimized ${ZLIB_INSTALL}/lib/zlibstatic.lib)
    set(ZLIB_HEADERS
            "${ZLIB_INSTALL}/include/zconf.h"
            "${ZLIB_INSTALL}/include/zlib.h")
    ExternalProject_Add(zlib
        PREFIX zlib
        URL https://sourceforge.net/projects/libpng/files/zlib/1.2.11/zlib1211.zip
        URL_MD5 16b41357b2cd81bca5e1947238e64465
        BINARY_DIR ${ZLIB_BUILD_DIR}
        BUILD_BYPRODUCTS ${ZLIB_STATIC_LIBRARIES}
        DOWNLOAD_DIR ${DOWNLOAD_LOCATION}
        INSTALL_DIR ${ZLIB_INSTALL}
        CMAKE_CACHE_ARGS
            -DCMAKE_BUILD_TYPE:STRING=Release
            -DCMAKE_INSTALL_PREFIX:STRING=${ZLIB_INSTALL}
            -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON)
    add_custom_target(zlib_create_destination_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ZLIB_INCLUDE_DIR}
        DEPENDS zlib)
    add_custom_target(zlib_copy_headers_to_destination 
        DEPENDS zlib_create_destination_dir)
    foreach(header_file ${ZLIB_HEADERS})
        add_custom_command(TARGET zlib_copy_headers_to_destination PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${ZLIB_INCLUDE_DIR})
    endforeach()

    set(PNG_INSTALL ${CMAKE_BINARY_DIR}/png)
    set(PNG_BUILD_DIR ${CMAKE_BINARY_DIR}/png/build)
    set(PNG_INCLUDE_DIR ${PNG_INSTALL}/include)
    set(PNG_STATIC_LIBRARIES
            debug ${PNG_INSTALL}/lib/libpng16_staticd.lib
            optimized ${PNG_INSTALL}/lib/libpng16_static.lib)
    set(PNG_HEADERS
            "${PNG_INCLUDE_DIR}/libpng16/png.h"
            "${PNG_INCLUDE_DIR}/libpng16/pngconf.h"
            "${PNG_INCLUDE_DIR}/libpng16/pnglibconf.h")
    ExternalProject_Add(png
        PREFIX png
        DEPENDS zlib
        URL https://sourceforge.net/projects/libpng/files/libpng16/1.6.35/lpng1635.zip
        URL_MD5 d8bfd42ee9e59404349b50106e04e1f9
        BINARY_DIR ${PNG_BUILD_DIR}
        BUILD_BYPRODUCTS ${PNG_STATIC_LIBRARIES}
        DOWNLOAD_DIR ${DOWNLOAD_LOCATION}
        INSTALL_DIR ${PNG_INSTALL}
        CMAKE_CACHE_ARGS
            -DCMAKE_BUILD_TYPE:STRING=Release
            -DCMAKE_INSTALL_PREFIX:STRING=${PNG_INSTALL}
            -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL})
    add_custom_target(png_create_destination_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PNG_INCLUDE_DIR}
        DEPENDS png)
    add_custom_target(png_copy_headers_to_destination
        DEPENDS png_create_destination_dir)
    foreach(header_file ${PNG_HEADERS})
        add_custom_command(TARGET png_copy_headers_to_destination PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${PNG_INCLUDE_DIR})
    endforeach()
else()
    find_package(PNG REQUIRED)
endif(WIN32)

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
if (WIN32)
    # Quiet warnings related to fopen on MSVC
    target_compile_definitions(unit_tests  PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_include_directories(unit_tests
        PUBLIC $<BUILD_INTERFACE:${PNG_INCLUDE_DIR}>)
    target_link_libraries(unit_tests
        PRIVATE unit_tests_options unit_tests_warnings Threads::Threads ${PNG_STATIC_LIBRARIES})
else()
    # We found libpng via find_package
    target_link_libraries(unit_tests
        PRIVATE unit_tests_options unit_tests_warnings Threads::Threads PNG::PNG)
endif()

add_test(NAME unit_test COMMAND unit_tests $ENV{PENGUINV_UNIT_TEST_RUN_COUNT})
