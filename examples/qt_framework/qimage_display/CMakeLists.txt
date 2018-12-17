# Find includes in the build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC
set(CMAKE_AUTOMOC ON)

set(TARGET_NAME example_qimage_display)
set(SRCS
    ${LIB_DIR}/blob_detection.cpp
    ${LIB_DIR}/image_function.cpp
    ${LIB_DIR}/image_function_helper.cpp
    ${LIB_DIR}/math/math_base.cpp
    ${LIB_DIR}/file/bmp_image.cpp
    ${LIB_DIR}/ui/ui.cpp
    ${LIB_DIR}/ui/qt/qt_ui.cpp)

add_executable(${TARGET_NAME} main.cpp ${SRCS})
target_link_libraries(${TARGET_NAME} example_features_options example_features_warnings Qt5::Widgets)
