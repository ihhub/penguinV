# How to compile    
- QT    
Open example_qimage_display.pro file in this directory inside QT Framework (Creator) and configure the project. Open qt-logo.bmp image within compiled application.

- CMake    
To build this example, you will need a fairly new toolchain with a compiler supporting at least
`C++11`, `Qt 5.9` and `CMake >= 3.8`.
From the project root directory, execute the following commands:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<path/to/Qt5/lib/cmake> ..
cmake --build examples/qt_framework/qimage_display --config Release
```
