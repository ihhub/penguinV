#How to compile    
- Microsoft Visual Studio    
Open example_blob_detection.vcxproj file in this folder to create solution for your Visual Studio version. Remember that minimum version of VS for this example is 2010.

- g++    
In this folder you need to type/paste this text in terminal:    
	```bash
	g++ -std=c++11 -Wall example_blob_detection.cpp ../../Library/image_function.cpp ../../Library/blob_detection.cpp ../../Library/FileOperation/bitmap.cpp -o application
	```

- make    
In this folder you need to type/paste this text in terminal:    
	```bash
	make ./example_blob_detection
	```
