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
	+  At this point you'll see something like 
	```
	-- CREATE OPENCV MODULE=1
	-- CMAKE_INSTALL_PREFIX=/usr/local
	-- REQUIRED_LIBRARIES=/opt/vc/lib/libmmal_core.so;/opt/vc/lib/libmmal_util.so;/opt/vc/lib/libmmal.so
	-- Change a value with: cmake -D<Variable>=<Value>
	-- 
	-- Configuring done
	-- Generating done
	-- Build files have been written to: /home/pi/raspicam/trunk/build
	```
	 If OpenCV development files are installed in your system, then  you'll see
	```
	-- CREATE OPENCV MODULE=1
	```
	otherwise this option will be 0 and the opencv module of the library will not be compiled.
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
