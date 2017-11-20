# How to compile    
- Microsoft Visual Studio    
Open example_bitmap_operation.vcxproj file in this directory to create solution for your Visual Studio version. Remember that minimum version of VS for this example is 2010.

- g++    
In this directory you need to type/paste this text in terminal:    
	```bash
	g++ -std=c++11 -Wall example_bitmap_operation.cpp ../../Library/image_function.cpp ../../Library/FileOperation/bitmap.cpp -o application
	```

- make    
In this directory you need to type/paste this text in terminal:    
	```bash
	make ./example_bitmap_operation
	```
