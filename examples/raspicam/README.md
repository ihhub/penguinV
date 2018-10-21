# How to compile  

### Installing Dependencies
- [RaspiCam](https://www.uco.es/investiga/grupos/ava/node/40)
	+ Download the RaspiCam archive from [here](https://sourceforge.net/projects/raspicam/files/latest/download)
	+ Extract the archive
		``` 
			# if .tar archive
			tar xvzf raspicam-xxxxxx.tar 
			# or if .zip archive
			unzip raspicam-xxxxxx.zip
		```
	+ cd to the extracted folder
		``` 
		cd raspicam-xxxxxx
		```
	+ create a buid dir and start compiling
		```  
		mkdir build
		cd build
		cmake ..
		```
	+  Finally compile, install and update the ldconfig
	``` make
	sudo make install
	sudo ldconfig
	```
	+ Run raspicam_test to check if the compilation is ok
### Compiling example with g++    
In this directory you need to type/paste this text in terminal:    
```bash
g++ -std=c++11 -Wall main.cpp ../../src/blob_detection.cpp ../../src/image_function_helper.cpp ../../src/image_function.cpp -o main -I/usr/local/include -L/opt/vc/lib -lraspicam -lmmal -lmmal_core -lmmal_util
```
