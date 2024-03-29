cmake_minimum_required(VERSION 3.8)
project(penguinV)
include(CTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH)
endif()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenCL)
find_package(Qt5Widgets)
find_package(CUDA)
if(CUDA_FOUND)
    # There is some bug in CMake (FindCUDA.cmake) that causes it to mess up the -std=c++11 option 
    # if you let it propagate it to nvcc in some cases.
    if(NOT MSVC)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        list(APPEND CUDA_NVCC_FLAGS -std=c++11 -Wno-deprecated-gpu-targets)
    endif()
    add_subdirectory(examples/cuda/image_function)
    add_subdirectory(examples/cuda/pinned_memory)
endif()

function(copy_data_files)
    set(EXAMPLE_DATA_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(EXAMPLE_DATA_DIR "${CMAKE_SOURCE_DIR}/data/")
    file(GLOB files "${EXAMPLE_DATA_DIR}/*.bmp" "${EXAMPLE_DATA_DIR}/*.png")
    foreach(file ${files})
        file(COPY ${file} DESTINATION ${EXAMPLE_DATA_INSTALL_DIR})
    endforeach()
endfunction()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm*|ARM*)")
    add_compile_options(-mfpu=neon-vfpv4)
endif()

option(PENGUINV_BUILD_TEST "Build tests of penguinV" ON)
if(${PENGUINV_BUILD_TEST} AND (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    enable_testing()
    add_subdirectory(test)
endif()

option(PENGUINV_BUILD_EXAMPLE "Build examples of penguinV" ON)
if(${PENGUINV_BUILD_EXAMPLE} AND (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    add_subdirectory(examples)
endif()

