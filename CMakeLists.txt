cmake_minimum_required(VERSION 3.8)
project(penguinV)
include(CTest)

if(NOT CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH)
endif()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenCL)
find_package(Qt5Widgets)

option(PENGUINV_BUILD_CUDA "Build CUDA related code of penguinV" ON)
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    find_path(CUDA_TOOLKIT_ROOT_DIR
        NAMES nvcc nvcc.exe
        PATHS ENV CUDA_BIN_PATH CUDA_TOOLKIT_ROOT_DIR
        DOC "CUDA Toolkit location."
        NO_DEFAULT_PATH)
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "CUDA Toolkit location." FORCE)
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR is set to ${CUDA_TOOLKIT_ROOT_DIR}")
endif(NOT CUDA_TOOLKIT_ROOT_DIR)
if(${PENGUINV_BUILD_CUDA} AND NOT CMAKE_VERSION VERSION_LESS "3.10.0")
  enable_language(CUDA)
elseif(${PENGUINV_BUILD_CUDA})
  find_package(CUDA REQUIRED)
endif()

option(PENGUINV_BUILD_TEST "Build tests of penguinV" ON)
if(${PENGUINV_BUILD_TEST} OR (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    enable_testing()
    add_subdirectory(test)
endif()

option(PENGUINV_BUILD_EXAMPLE "Build examples of penguinV" ON)
if(${PENGUINV_BUILD_EXAMPLE} OR (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    add_subdirectory(examples)
endif()
