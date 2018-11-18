cmake_minimum_required(VERSION 3.8)
project(penguinV)
include(CTest)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenCL)
find_package(Qt5Widgets)

option(PENGUINV_BUILD_TEST "Build tests of penguinV" ON)
if(${PENGUINV_BUILD_TEST} OR (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    enable_testing()
    add_subdirectory(test)
endif()

option(PENGUINV_BUILD_EXAMPLE "Build examples of penguinV" ON)
if(${PENGUINV_BUILD_EXAMPLE} OR (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    add_subdirectory(examples)
endif()
