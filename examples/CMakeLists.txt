# Link this 'library' to set the c++ standard / compile-time options requested
add_library(example_features_options INTERFACE)
target_compile_features(example_features_options INTERFACE cxx_std_11)

# Link this 'library' to use the standard warnings
add_library(example_features_warnings INTERFACE)
if(MSVC)
    target_compile_options(example_features_warnings
                           INTERFACE /W4 /WX)
else()
    target_compile_options(example_features_warnings
                           INTERFACE -Wall
                                     -Wextra           # reasonable and standard
                                     -Wstrict-aliasing # warn the user if a variable declaration shadows one
                                     -Wpedantic        # warn if non-standard C++ is used
                                     -Wconversion
                                     -Wsign-conversion
                                     -O2
                                     -march=native)
endif()

set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
add_subdirectory(bitmap_operation)
add_subdirectory(blob_detection)
if(CUDA_FOUND)
    add_subdirectory(cuda/image_function)
    add_subdirectory(cuda/pinned_memory)
endif()
add_subdirectory(filters)
add_subdirectory(function_pool)
add_subdirectory(image_function)
if(OpenCL_FOUND)
    add_subdirectory(opencl/image_function)
endif()
if(Qt5Widgets_FOUND)
    add_subdirectory(qt_framework/qimage_display)
endif()
add_subdirectory(thread_pool)
