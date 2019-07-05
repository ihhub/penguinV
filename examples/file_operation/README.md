# How to compile    
- Microsoft Visual Studio    
Open example_bitmap_operation.vcxproj file in this directory to create solution for your Visual Studio version. Remember that minimum version of VS for this example is 2010.

- g++    
In this directory you need to type/paste this text in terminal:    
	```bash
	g++ -std=c++11 -Wall -D PENGUINV_ENABLED_PNG_SUPPORT example_file_operation.cpp ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/file/file_image.cpp ../../src/file/bmp_image.cpp ../../src/file/png_image.cpp ../../src/file/jpeg_image.cpp -o application -lpng
	```

- make    
In this directory you need to type/paste this text in terminal:    
	```bash
	make ./example_file_operation
	```

- CMake    
To build this example, you will need a fairly new toolchain with a compiler supporting at least
`C++11` and `CMake >= 3.8`.
From the project root directory, execute the following commands:
```
mkdir build
cd build
cmake ..
cmake --build ./examples/file_operation --config Release
```
