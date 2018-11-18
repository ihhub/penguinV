# How to compile
- Microsoft Visual Studio
Open example_thread_pool.vcxproj file in this directory to create solution for your Visual Studio version. This project was created under Visual Studio 2015.

- g++
In this directory you need to type/paste this text in terminal:    
	```bash
	g++ -std=c++11 -pthread -Wall example_thread_pool.cpp ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/thread_pool.cpp ../../src/penguinv/penguinv.cpp -o application
	```

- make
In this directory you need to type/paste this text in terminal:    
	```bash
	make ./example_thread_pool
	```

- CMake
To build this example, you will need a fairly new toolchain with a compiler supporting at least
`C++11` and `CMake >= 3.8`.
From the project root directory, execute the following commands:
```
mkdir build
cd build
cmake ..
cmake --build ./examples/thread_pool --config Release
```
