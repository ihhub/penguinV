# Find includes in the build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC
set(CMAKE_AUTOMOC ON)

# Add a compiler flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -Wsign-conversion")

# Find the QtWidgets library
find_package(Qt5Widgets REQUIRED)
set(TARGET_NAME qimage_display)

# Generate code from ui files => qt5_wrap_ui
# Generate rules for building source files from the resources => qt5_add_resources

set(LIB_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRCS
    ${LIB_DIR}/blob_detection.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/math/math_base.cpp
    ${LIB_DIR}/FileOperation/bitmap.cpp)

add_executable(${TARGET_NAME} main.cpp ${SRCS})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_11)
target_link_libraries(${TARGET_NAME} Qt5::Widgets)
