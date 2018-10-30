option(PENGUINV_BUILD_QT_EXAMPLE "Build Qt examples of penguinV" OFF)
if(${PENGUINV_BUILD_QT_EXAMPLE})
    add_subdirectory(qt_framework)
endif()
