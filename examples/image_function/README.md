# How to compile    
- Microsoft Visual Studio    
Open example_image_function.vcxproj file in this directory to create solution for your Visual Studio version. Remember that minimum version of VS for this example is 2010.

- g++    
In this directory you need to type/paste this text in terminal:
	```bash
	g++ -std=c++11 -Wall example_image_function.cpp ../../src/image_function_helper.cpp ../../src/image_function.cpp -o application
	```

- make    
In this directory you need to type/paste this text in terminal:    
	```bash
	make ./example_image_function
	```

- CMake    
To build this example, you will need a fairly new toolchain with a compiler supporting at least
`C++11` and `CMake >= 3.8`.
From the project root directory, execute the following commands:
```
mkdir build
cd build
cmake ..
cmake --build ./examples/image_function --config Release
```
