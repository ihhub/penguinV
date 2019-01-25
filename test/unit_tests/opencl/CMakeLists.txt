set(MAIN_LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SRCS
    ${MAIN_LIB_DIR}/opencl/opencl_device.cpp
    ${MAIN_LIB_DIR}/opencl/opencl_helper.cpp
    ${MAIN_LIB_DIR}/image_function.cpp
    ${MAIN_LIB_DIR}/image_function_helper.cpp
    ${MAIN_LIB_DIR}/opencl/image_function_opencl.cpp
    ${UNIT_TEST_DIR}/unit_test_helper.cpp
    ${UNIT_TEST_DIR}/../test_helper.cpp
    ${UNIT_TEST_DIR}/unit_test_framework.cpp
    unit_test_helper_opencl.cpp
    unit_test_image_function_opencl.cpp
    unit_tests_opencl.cpp)
set(TARGET_NAME unit_tests_opencl)

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
                            INTERFACE 
                                -Wall
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
target_link_libraries(${TARGET_NAME} PRIVATE ${TARGET_NAME}_options
                                             ${TARGET_NAME}_warnings
                                             OpenCL::OpenCL)

if(MSVC)
    target_compile_definitions(${TARGET_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})

