include(ExternalProject)
set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

set(PNG_INSTALL ${CMAKE_BINARY_DIR}/png)
set(PNG_BUILD_DIR ${CMAKE_BINARY_DIR}/png/build)
set(PNG_LIB_DIR ${PNG_INSTALL}/lib)
set(PNG_INCLUDE_DIR ${PNG_INSTALL}/include)
set(PNG_HEADERS
        "${PNG_INCLUDE_DIR}/libpng16/png.h"
        "${PNG_INCLUDE_DIR}/libpng16/pngconf.h"
        "${PNG_INCLUDE_DIR}/libpng16/pnglibconf.h")

if(WIN32)
    if(MSVC)
        set(PNG_STATIC_LIBRARIES
            debug ${PNG_LIB_DIR}/libpng16_staticd.lib
            optimized ${PNG_LIB_DIR}/libpng16_static.lib)
    else()
        if(CMAKE_BUILD_TYPE EQUAL Debug)
            set(PNG_STATIC_LIBRARIES ${PNG_LIB_DIR}/libpng16_staticd.lib)
      else()
            set(PNG_STATIC_LIBRARIES ${PNG_LIB_DIR}/libpng16_static.lib)
      endif()
    endif()
else()
    set(PNG_STATIC_LIBRARIES ${PNG_LIB_DIR}/libpng16.a)
endif()

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
        -DPNG_TESTS:BOOL=OFF
        -DPNG_SHARED:BOOL=OFF
        -DPNG_STATIC:BOOL=ON
        -DSKIP_INSTALL_EXECUTABLES:BOOL=OFF
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

ExternalProject_Get_Property(png install_dir)
add_library(PNG_EXTERNAL UNKNOWN IMPORTED)
set(PNG_LIBRARY ${install_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}png16${CMAKE_STATIC_LIBRARY_SUFFIX})
set(PNG_INCLUDE_DIRS ${install_dir}/include)
set_target_properties(PNG_EXTERNAL PROPERTIES
    IMPORTED_LOCATION ${PNG_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${PNG_INCLUDE_DIRS}
    LINK_INTERFACE_LIBRARIES ${PNG_LIBRARY})
