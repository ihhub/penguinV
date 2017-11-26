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


SOURCES += main.cpp
SOURCES += ../../../src/FileOperation/bitmap.cpp
SOURCES += ../../../src/image_function.cpp
SOURCES += ../../../src/blob_detection.cpp
