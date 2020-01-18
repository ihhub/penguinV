include(ExternalProject)
option(PENGUINV_ENABLE_PNG_SUPPORT "Enable support of libpng" ON)
option(PENGUINV_USE_EXTERNAL_PNG "Download libpng and build from source" OFF)
option(PENGUINV_INSTALL_ZLIB "Download zlib and build from source" ON)

if(PENGUINV_ENABLE_PNG_SUPPORT)
    find_package(PNG)
    if(NOT PNG_FOUND)
        set(PENGUINV_USE_EXTERNAL_PNG ON CACHE BOOL "" FORCE)
        message(STATUS "libpng has not been found in the system and will be downloaded")
    endif()
endif()

if(PENGUINV_USE_EXTERNAL_PNG)
    set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
    CACHE PATH "Location where external projects will be downloaded.")
    mark_as_advanced(DOWNLOAD_LOCATION)

    # ZLIB
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

	if(DEFINED ENV{APPVEYOR})
        if(EXISTS ${ZLIB_INSTALL})
            set(PENGUINV_INSTALL_ZLIB OFF CACHE BOOL "" FORCE)
        endif()
    endif()
	
	if(PENGUINV_INSTALL_ZLIB)
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
	endif()
	
	add_library(ZLIB_EXTERNAL STATIC IMPORTED)
	set(ZLIB_LIBRARY_RELEASE ${ZLIB_INSTALL}/lib/zlibstatic.lib)
	set(ZLIB_LIBRARY_RELWITHDEBINFO ${ZLIB_INSTALL}/lib/zlibstatic.lib)
	set(ZLIB_LIBRARY_DEBUG ${ZLIB_INSTALL}/lib/zlibstaticd.lib)
	set_target_properties(ZLIB_EXTERNAL PROPERTIES
		IMPORTED_LOCATION_RELEASE "${ZLIB_LIBRARY_RELEASE}"
		IMPORTED_LOCATION_RELWITHDEBINFO "${ZLIB_LIBRARY_RELWITHDEBINFO}"
		IMPORTED_LOCATION_DEBUG "${ZLIB_LIBRARY_DEBUG}")

    # PNG
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

	if(PENGUINV_INSTALL_ZLIB)
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
	else()
		ExternalProject_Add(png
			PREFIX png
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
	endif()

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
    add_library(PNG_EXTERNAL STATIC IMPORTED)
    set(PNG_LIBRARY_RELEASE ${install_dir}/lib/libpng16_static.lib)
    set(PNG_LIBRARY_RELWITHDEBINFO ${install_dir}/lib/libpng16_static.lib)
    set(PNG_LIBRARY_DEBUG ${install_dir}/lib/libpng16_staticd.lib)
    set(PNG_INCLUDE_DIRS ${install_dir}/include)
    # CMake INTERFACE_INCLUDE_DIRECTORIES requires the directory to exists at configure time
    # This is quite unhelpful because those directories are only generated at build time
    file(MAKE_DIRECTORY ${PNG_INCLUDE_DIRS}) # Workaround
    set_target_properties(PNG_EXTERNAL PROPERTIES
        IMPORTED_LOCATION_RELEASE "${PNG_LIBRARY_RELEASE}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${PNG_LIBRARY_RELWITHDEBINFO}"
        IMPORTED_LOCATION_DEBUG "${PNG_LIBRARY_DEBUG}"
        INTERFACE_INCLUDE_DIRECTORIES ${PNG_INCLUDE_DIRS})
endif()
