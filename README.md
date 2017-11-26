penguinV
======

| **Linux + Mac** |
|-----------------|
| [![Build status](https://travis-ci.org/ihhub/penguinV.svg?branch=master)](https://travis-ci.org/ihhub/penguinV) |

PenguinV is a simple and easy to use C++ image processing library. It is designed to have simple programming syntax and to deliver good performance. Some core features of the library are:

- compactness
- multithreading support for individual functions (please refer to [multithreading support](#multithreading-support) section)
- exception-based code
- optional GPU (CUDA, OpenCL) and SIMD (SSE, AVX, NEON) support
- cross-platform
- functions can perform processing on separate image parts (no need to make a copy of image for area on what you want to do something, just set area parameters)
- user-defined image types support (you can create your own image types and image functions [See API description, ImageTemplate class])

At the current stage of development library does not have some features but we are intending to introduce them very soon:
- better support of CUDA and OpenCL
- more basic functions and their implementations by SSE, AVX, NEON
- color space conversions

The library does **NOT** have it's own image displaying code but you can refer to QT example.

Requirements
---------------------------
To compile the source code your compiler must support at least **C++ 11** version. Minimum required version of Microsoft Visual Studio [without AVX 2.0 support and thread pool] is VS 2010.

How to install
---------------------------
The library is distributed in the form of source code. To use the library you need to include library files into your application project. That's it! No more extra moves!

How to compile an example
---------------------------
Open README.md file in any of example directories and follow instructions.

Multithreading support
---------------------------
Every non empty image can be divided into multiple parts or areas (in scientific terms ROI - region of interest). To run image processing in multiple threads you need only to split bigger ROI into small parts and call necessary basic functions. No extra magic! Make sure that small parts are not intersecting by each other.    
Almost all basic functions support multithreading. Please refer to **Function_Pool** namespace and function_pool example.

GPU support
---------------------------
All source code related to CUDA or OpenCL is located in separate directory named as **cuda** and **opencl** respectively. Read full description about CUDA or OpenCL support in **README** file in the directory.

License
---------------------------
This project is under 3-clause BSD License. Please refer to file **LICENSE** for more details.

API description
---------------------------
Please refer to file **API_description.md** in **doc** directory for full description of API.
