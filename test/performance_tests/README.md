# Description    
This directory contains a project to run performance tests on PenguinV library.

# How to compile    
- Microsoft Visual Studio    
Open performance_tests.vcxproj file in this directory to create solution for your Visual Studio version. This project was created under Visual Studio 2015.

- g++    
In this directory you need to type/paste this text in terminal:    
	```cpp
	g++ -std=c++11 -Wall performance_tests.cpp ../../src/image_function.cpp ../../src/thread_pool.cpp ../../src/function_pool.cpp performance_test_framework.cpp performance_test_helper.cpp performance_test_image_function.cpp performance_test_function_pool.cpp ../../src/penguinv/penguinv.cpp -o application
	```

- CMake
To build the project, you will need to have CMake with minimum version of 3.10 and a compiler 
that support `cxx_std_11`.

Then in the project directory:
```
mkdir build && cd build
cmake .. -DPENGUINV_BUILD_TEST=ON
```

After successfully finishing the build process, run the tests to see if everything is work.

You can use ctest

```
$ ctest
```

Or make

```
$ make test
```

Or execute the tests directly. To do so, simply go to the `build/test/` folder and run the tests.

```
$ ./performance_tests/performance_tests
$ ./unit_tests/unit_tests
```
