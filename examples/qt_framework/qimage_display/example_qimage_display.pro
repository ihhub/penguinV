#-------------------------------------------------
#
# Project created for PenguinV library
#
#-------------------------------------------------

QT += core
QT += gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtSample
CONFIG   += console
CONFIG   += c++11
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
           ../../../src/file/bmp_image.cpp \
           ../../../src/image_function.cpp \
           ../../../src/image_function_helper.cpp \
           ../../../src/blob_detection.cpp \
           ../../../src/ui/ui.cpp \
           ../../../src/ui/qt/qt_ui.cpp

HEADERS += ../../../src/ui/qt/qt_ui.h
