include(ExternalProject)
set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

set(ZLIB_INSTALL ${CMAKE_BINARY_DIR}/zlib)
set(ZLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/zlib/build)
set(ZLIB_LIB_DIR ${ZLIB_INSTALL}/lib)
set(ZLIB_INCLUDE_DIR ${ZLIB_INSTALL}/include)
set(ZLIB_HEADERS
    "${ZLIB_INSTALL}/include/zconf.h"
    "${ZLIB_INSTALL}/include/zlib.h")

if(WIN32)
    if(MSVC)
        set(ZLIB_STATIC_LIBRARIES
            debug ${ZLIB_LIB_DIR}/zlibstaticd.lib
            optimized ${ZLIB_LIB_DIR}/zlibstatic.lib)
    else()
        if(CMAKE_BUILD_TYPE EQUAL Debug)
            set(ZLIB_STATIC_LIBRARIES ${ZLIB_LIB_DIR}/zlibstaticd.lib)
      else()
            set(ZLIB_STATIC_LIBRARIES ${ZLIB_LIB_DIR}/zlibstatic.lib)
      endif()
    endif()
  else()
    set(ZLIB_STATIC_LIBRARIES ${ZLIB_LIB_DIR}/libz.a)
endif()

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
