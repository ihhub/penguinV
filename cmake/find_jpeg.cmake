include(ExternalProject)
option(PENGUINV_ENABLE_JPEG_SUPPORT "Enable support of libpng" ON)
option(PENGUINV_USE_EXTERNAL_JPEG "Download libpng and build from source" OFF)

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
endif()
