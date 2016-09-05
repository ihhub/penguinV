#Description    
This folder contains a project to run unit tests on overall library. It should be run once after any changes.

#How to compile    
- Microsoft Visual Studio    
Open unit_tests.vcxproj file in this folder to create solution for your Visual Studio version. This project was created under Visual Studio 2010.

- g++    
In this folder you need to type/paste this text in terminal:    
	```cpp
	g++ -std=c++11 -Wall unit_tests.cpp ../Library/image_function.cpp unit_test_framework.cpp unit_test_helper.cpp unit_test_image_buffer.cpp unit_test_image_function.cpp -o application
	```
