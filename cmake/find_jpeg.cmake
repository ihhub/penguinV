include(ExternalProject)
option(PENGUINV_ENABLE_JPEG_SUPPORT "Enable support of libjpeg" ON)
option(PENGUINV_USE_EXTERNAL_JPEG "Download libjpeg and build from source" OFF)

if(PENGUINV_ENABLE_JPEG_SUPPORT)
    find_package(JPEG)
    if(NOT JPEG_FOUND)
        set(PENGUINV_USE_EXTERNAL_JPEG ON CACHE BOOL "" FORCE)
        message(STATUS "libjpeg has not been found in the system and will be downloaded")
    endif()
endif()

if(PENGUINV_USE_EXTERNAL_JPEG)
    set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
    CACHE PATH "Location where external projects will be downloaded.")
    mark_as_advanced(DOWNLOAD_LOCATION)

    # Download yasm
    set(YASM_INSTALL ${CMAKE_BINARY_DIR}/yasm)
    set(YASM_INCLUDE_DIR ${YASM_INSTALL}/include)
    set(YASM_LIBRARIES ${YASM_INSTALL}/lib/libyasm.a)
    ExternalProject_Add(yasm
    URL http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
    CMAKE_CACHE_ARGS
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_BUILD_TYPE:STRING=RELEASE
        -DCMAKE_INSTALL_PREFIX:STRING=${YASM_INSTALL}
    INSTALL_COMMAND ""
    TEST_COMMAND "")

    if (WIN32)
        set(YASM_BINARY ${YASM_INSTALL}/bin/yasm.exe)
    else()
        set(YASM_BINARY ${YASM_INSTALL}/bin/yasm)
    endif()

    set(JPEG_INSTALL ${CMAKE_BINARY_DIR}/jpeg)
    set(JPEG_BUILD_DIR ${CMAKE_BINARY_DIR}/jpeg/build)
    set(JPEG_LIB_DIR ${JPEG_INSTALL}/lib)
    set(JPEG_INCLUDE_DIR ${JPEG_INSTALL}/include)

    if(WIN32)
        if(MSVC)
            set(JPEG_STATIC_LIBRARIES
                debug ${JPEG_LIB_DIR}/jpeg-staticd.lib
                optimized ${JPEG_LIB_DIR}/jpeg-static.lib)
        else()
            if(CMAKE_BUILD_TYPE EQUAL Debug)
                set(JPEG_STATIC_LIBRARIES ${JPEG_LIB_DIR}/jpeg-staticd.lib)
            else()
                set(JPEG_STATIC_LIBRARIES ${JPEG_LIB_DIR}/jpeg-static.lib)
            endif()
        endif()
    else()
        set(JPEG_STATIC_LIBRARIES ${JPEG_LIB_DIR}/libjpeg.a)
    endif()

    ExternalProject_Add(jpeg-turbo
        PREFIX jpeg-turbo
        URL https://sourceforge.net/projects/libjpeg-turbo/files/2.0.1/libjpeg-turbo-2.0.1.tar.gz
        URL_MD5 1b05a66aa9b006fd04ed29f408e68f46
        INSTALL_DIR ${JPEG_INSTALL}
        CMAKE_ARGS
            -DCMAKE_DEBUG_POSTFIX=d
            -DENABLE_SHARED=FALSE
            -DWITH_SIMD=FALSE
            -DENABLE_STATIC=TRUE
            -DWITH_TURBOJPEG=TRUE
            -DWITH_JPEG8=TRUE
            -DCMAKE_INSTALL_PREFIX=${JPEG_INSTALL}
            -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
            -DCMAKE_ASM_NASM_COMPILER=${YASM_BINARY})

    ExternalProject_Get_Property(jpeg install_dir)
    add_library(JPEG_EXTERNAL STATIC IMPORTED)
    set(JPEG_LIBRARY_RELEASE ${install_dir}/lib/jpeg-static.lib)
    set(JPEG_LIBRARY_RELWITHDEBINFO ${install_dir}/lib/jpeg-static.lib)
    set(JPEG_LIBRARY_DEBUG ${install_dir}/lib/jpeg-staticd.lib)
    set(JPEG_INCLUDE_DIRS ${install_dir}/include)
    # CMake INTERFACE_INCLUDE_DIRECTORIES requires the directory to exists at configure time
    # This is quite unhelpful because those directories are only generated at build time
    file(MAKE_DIRECTORY ${JPEG_INCLUDE_DIRS}) # Workaround
    set_target_properties(JPEG_EXTERNAL PROPERTIES
        IMPORTED_LOCATION_RELEASE "${JPEG_LIBRARY_RELEASE}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${JPEG_LIBRARY_RELWITHDEBINFO}"
        IMPORTED_LOCATION_DEBUG "${JPEG_LIBRARY_DEBUG}"
        INTERFACE_INCLUDE_DIRECTORIES ${JPEG_INCLUDE_DIRS})
endif()
