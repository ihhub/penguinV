# penguinV

PenguinV is a simple and easy to use C++ image processing library. It is designed to have simple programming syntax and to deliver good performance. Some core features of the library are:

- compact
- multithreading support for individual functions (please refer to [multithreading support](#multithreading-support) section)
- exception-based code
- optional SSE, AVX, NEON support
- cross-platform
- functions can perform processing on separate image parts (no need to make a copy of image for area on what you want to do something, just set area parameters)
- user-defined image types support (you can create your own image types and image functions [See API description, ImageTemplate class])

At current stage of development library does not have many features but we are intending to introduce them very soon:
- more basic functions and their implementations by SSE, AVX, NEON
- ~~blob detection code~~
- template matching classes
- ~~multi-level thread pool~~
- etc.

The library does **NOT** provide such features as:
- load/save image from/to memory storage for some image formats
- image conversion between image formats
- image displaying

In many cases when developers design their own image processing application they are facing problems with integration of third-party library into code. To minimize such drawbacks we are giving an option to write your own code for above situations.

#Requirements    
To compile the source code your compiler must support at least **C++ 11** version. Minimum required version of Microsoft Visual Studio [without AVX 2.0 support and thread pool] is VS 2010.

#How to install    
We prefer that an end-user (that means YOU) compile all files what are necessary for your application. For this you have to copy files into your project folder and use them. That's it! No more extra moves! Just copy, include files into project and compile them as a part of your application.

#How to compile an example    
Open README.md file in any of example folders and follow instructions.

#Multithreading support    
Every image (not empty) can be divided by multiple parts or areas (in scientific terms region of interest - ROI). To run image processing in multiple threads you need only to split bigger ROI into small parts and call necessary basic functions. No extra magic! Make sure that small parts are not intercepting by each other.    
Almost all basic functions already have embedded multithreading support. Please refer to **Function_Pool** namespace and function_pool example.

#SSE/AVX/NEON support    
We do not provide source code for identification whether your CPU supports SSE2/AVX 2.0/NEON. This should be your part of code. We made this to support cross-platform code.

If your CPU does not support SSE2, AVX 2.0 or NEON just do NOT use related files in your project :wink:

#License    
This project is under 3-clause BSD License. Please refer to file **LICENSE** for more details.

#API description    
Please refer to file **API_description.md** for full description of API.
