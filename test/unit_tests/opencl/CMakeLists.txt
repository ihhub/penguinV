set(MAIN_LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SRCS
    ${MAIN_LIB_DIR}/opencl/opencl_device.cpp
    ${MAIN_LIB_DIR}/opencl/opencl_helper.cpp
    ${MAIN_LIB_DIR}/image_function.cpp
    ${MAIN_LIB_DIR}/image_function_helper.cpp
    ${MAIN_LIB_DIR}/opencl/image_function_opencl.cpp
    ${UNIT_TEST_DIR}/unit_test_helper.cpp
    ${UNIT_TEST_DIR}/unit_test_framework.cpp
    unit_test_helper_opencl.cpp
    unit_test_image_function_opencl.cpp
    unit_tests_opencl.cpp)
set(TARGET_NAME unit_tests_opencl)

find_package(OpenCL)
if (NOT OpenCL_FOUND)
    message(STATUS "OpenCL was not found, searching for CUDA")
    find_package(CUDA)
    if (CUDA_FOUND)
        message (STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
        message (STATUS "CUDA libs: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        message (STATUS "CUDA version: ${CUDA_VERSION}")

        find_path(OpenCL_INCLUDE_DIR
                    NAMES CL/cl.h OpenCL/cl.h
                    PATHS 
                        ENV AMDAPPSDKROOT
                        ENV CUDA_PATH
                    PATH_SUFFIXES 
                        include
                        OpenCL/common/inc)
        find_library(OpenCL_LIBRARY
                    NAMES OpenCL
                    PATHS 
                        ENV AMDAPPSDKROOT
                        ENV CUDA_PATH
                    PATH_SUFFIXES lib lib64)
        set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
        set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
        include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
        find_package_handle_standard_args(
            OpenCL
            FOUND_VAR OpenCL_FOUND
            REQUIRED_VARS OpenCL_LIBRARY OpenCL_INCLUDE_DIR
            VERSION_VAR OpenCL_VERSION_STRING)

        mark_as_advanced(OpenCL_INCLUDE_DIR OpenCL_LIBRARY)

        if(OpenCL_FOUND AND NOT TARGET OpenCL::OpenCL)
            if(OpenCL_LIBRARY MATCHES "/([^/]+)\\.framework$")
                add_library(OpenCL::OpenCL INTERFACE IMPORTED)
                set_target_properties(OpenCL::OpenCL PROPERTIES
                INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARY}")
            else()
                add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                set_target_properties(OpenCL::OpenCL PROPERTIES
                IMPORTED_LOCATION "${OpenCL_LIBRARY}")
            endif()
        set_target_properties(OpenCL::OpenCL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}")
        endif()
    endif (CUDA_FOUND)
endif (NOT OpenCL_FOUND)

if (NOT OpenCL_FOUND AND NOT CUDA_FOUND)
    message(WARNING "Neither OpenCL nor CUDA were found but are required to build ${TARGET_NAME}")
else (NOT OpenCL_FOUND AND NOT CUDA_FOUND)
    # Link this 'library' to set the c++ standard / compile-time options requested
    add_library(${TARGET_NAME}_options INTERFACE)
    target_compile_features(${TARGET_NAME}_options INTERFACE cxx_std_11)

    # Link this 'library' to use the standard warnings
    add_library(${TARGET_NAME}_warnings INTERFACE)
    if(MSVC)
        target_compile_options(${TARGET_NAME}_warnings 
                            INTERFACE /W4 /WX)
    else()
        target_compile_options(${TARGET_NAME}_warnings
                            INTERFACE -Wall
                                        -Wextra           # reasonable and standard
                                        -Wstrict-aliasing # warn the user if a variable declaration shadows one
                                        -Wpedantic        # warn if non-standard C++ is used
                                        -Wconversion
                                        -Wsign-conversion
                                        -O2
                                        -march=native)
    endif()

    # Move the the executable file created by add_executable to the <build_directory>/bin subfolder
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    add_executable(${TARGET_NAME} ${SRCS})
    target_link_libraries(${TARGET_NAME}
                        PRIVATE ${TARGET_NAME}_options
                                ${TARGET_NAME}_warnings
                                OpenCL::OpenCL)
    if(MSVC)
        target_compile_definitions(${TARGET_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endif (NOT OpenCL_FOUND AND NOT CUDA_FOUND)
