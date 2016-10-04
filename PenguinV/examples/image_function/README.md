#How to compile    
- Microsoft Visual Studio    
Open example_image_function.vcxproj file in this folder to create solution for your Visual Studio version. Remember that minimum version of VS for this example is 2010.

- g++    
In this folder you need to type/paste this text in terminal:    
	```cpp
	g++ -std=c++11 -Wall example_image_function.cpp ../../Library/image_function.cpp -o application
	```
