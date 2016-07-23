#How to compile    
- Microsoft Visual Studio    
Open example_function_pool.vcxproj file in this folder to create solution for your Visual Studio version. This project was created under Visual Studio 2015.

- g++    
In this folder you need to type/paste this text in terminal:    
	```cpp
	g++ -std=c++11 -pthread -Wall example_function_pool.cpp ../Library/image_function.cpp ../Library/thread_pool.cpp ../Library/function_pool.cpp -o application
	```
