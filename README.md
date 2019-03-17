penguinV
======
 [![Build status](https://travis-ci.org/ihhub/penguinV.svg?branch=master)](https://travis-ci.org/ihhub/penguinV) [![Build status](https://ci.appveyor.com/api/projects/status/g4a42ac5ktra8utq/branch/master?svg=true)](https://ci.appveyor.com/project/ihhub/penguinv/branch/master) [![CodeFactor](https://www.codefactor.io/repository/github/ihhub/penguinv/badge)](https://www.codefactor.io/repository/github/ihhub/penguinv)

PenguinV is a simple and easy to use C++ **image processing** library with focus on heterogeneous systems. The library is designed with an idea to have simple common API for CPUs and GPUs simplifying developer's work on context switching between devices. Core features of the library:

- heterogeneous system support (CPUs and GPUs)
- optional GPU (CUDA, OpenCL) and SIMD (SSE, AVX, NEON) support
- [Python support](https://github.com/ihhub/penguinV/tree/master/src/python)
- [multithreading support](#multithreading-support)
- cross-platform
- compactness
- ability to process separate parts of an image (ROI)
- user-defined image types and more.

The project in is active process of development so new features are coming soon!

What can it do?
---------------------------
You can develop a software within minutes of your time to obtain **high performance** and **accuracy**. It is up to developer to decide which device (CPU or GPU) would execute the code or give such control to the library.

Example code for below images could look like this:
```cpp
Image red = ExtractChannel( image, 0 ); // 0 is red channel
Image binary = Threshold( red, // threshold
                          GetThreshold( // get weighted threshold
                              Histogram( red ) ) ) ); // get image histogram

BlobDetection detection;
...
detection.find( binary );
...
Image rgb = ConvertToRgb( binary );
rgd = BitwiseAnd( image, rgb );
...
rgd = BitwiseOr( image, rgb );
```

The trick behind the code is that you have a **single interface** for CPU as well as for GPU!

![one](https://github.com/ihhub/penguinV/blob/readme_changes/data/readme/1_original.png) ![two](https://github.com/ihhub/penguinV/blob/readme_changes/data/readme/2_red_channel.png) ![three](https://github.com/ihhub/penguinV/blob/readme_changes/data/readme/3_red_threshold.png) ![four](https://github.com/ihhub/penguinV/blob/readme_changes/data/readme/4_blob.png)
![five](https://github.com/ihhub/penguinV/blob/readme_changes/data/readme/5_logical_and.png) ![six](https://github.com/ihhub/penguinV/blob/readme_changes/data/readme/6_result.png)

Contribution
---------------------------
We welcome and appreciate any help, even if it's a tiny text or code change. Please read [contribution](https://github.com/ihhub/penguinV/blob/master/CONTRIBUTING.md) page before starting work on a pull request. All contributors are listed in the project's wiki [page](https://github.com/ihhub/penguinV/wiki/Contributors). 
Not sure what to start with? Feel free to refer to <kbd>[good first issue](https://github.com/ihhub/penguinV/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)</kbd> or <kbd>[help wanted](https://github.com/ihhub/penguinV/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)</kbd> tags.

Requirements
---------------------------
To compile the source code your compiler must support at least **C++ 11** version.

How to install
---------------------------
The library is distributed in the form of source code. To use the library you need to include necessary files into your application project. That's it! No more extra moves!

How to compile an example
---------------------------
Open README.md file in any of [example](https://github.com/ihhub/penguinV/tree/master/examples) directories and follow instructions.

Multithreading support
---------------------------
The library contains it's own thread pool which creates multiple tasks to run image processing function for a given image via multiple threads. Such tenchnique gives a big boost on machines with major CPU usage.

GPU support
---------------------------
All source code and descriptions related to CUDA or OpenCL are located in separate [**src/cuda**](https://github.com/ihhub/penguinV/tree/master/src/cuda) and [**src/opencl**](https://github.com/ihhub/penguinV/tree/master/src/opencl) directories respectively. Read full description about CUDA or OpenCL support in **README** file in the directory.

License
---------------------------
This project is under 3-clause BSD License. Please refer to file [**LICENSE**](https://github.com/ihhub/penguinV/blob/master/LICENSE) for more details.

API description
---------------------------
Directory [**doc**](https://github.com/ihhub/penguinV/tree/master/doc) contains latest and valid information and description of library's API.
