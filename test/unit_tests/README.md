# Description    
This directory contains a project to run unit tests on overall library. It should be run once after any changes.

# How to compile    
- Microsoft Visual Studio    
Open unit_tests.vcxproj file in this directory to create solution for your Visual Studio version. This project was created under Visual Studio 2010.

- g++    
In this directory you need to type/paste this text in terminal:    
	```cpp
	g++ -std=c++11 -Wall unit_tests.cpp ../../src/image_function.cpp ../../src/blob_detection.cpp ../../src/FileOperation/bitmap.cpp unit_test_bitmap.cpp unit_test_blob_detection.cpp unit_test_framework.cpp unit_test_helper.cpp unit_test_image_buffer.cpp unit_test_image_function.cpp -o application
	```
