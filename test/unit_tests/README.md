# Description    
This directory contains a project to run unit tests on overall library. It should be run once after any changes.

# How to compile    
- Microsoft Visual Studio    
Open unit_tests.vcxproj file in this directory to create solution for your Visual Studio version. This project was created under Visual Studio 2010.

- g++    
In this directory you need to type/paste this text in terminal:    
	```cpp
	g++ -std=c++11 -Wall unit_tests.cpp ../../src/image_function.cpp ../../src/blob_detection.cpp ../../src/FileOperation/bitmap.cpp ../../src/math/hough_transform.cpp ../../src/math/math_base.cpp unit_test_bitmap.cpp unit_test_blob_detection.cpp unit_test_framework.cpp unit_test_helper.cpp unit_test_image_buffer.cpp unit_test_image_function.cpp unit_test_math.cpp -o application
	```

- CMake    
	To build the project, you will need to have CMake with minimum version of 3.8 and a compiler 
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
	$ ./unit_tests/unit_tests
	```
