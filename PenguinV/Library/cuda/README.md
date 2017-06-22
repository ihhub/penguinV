CUDA support in penguinV
======

All source code related to CUDA is located in ```*.cu``` and ```*.cuh``` files. During a compilation of your project make sure that you are compiling with compiler supporting CUDA otherwise you will have compilation errors.

CUDA is a GPGPU technology from NVidia, which allows to perform computations with massive data-level parallelism on GPU. GPU can process only data located in the videocard onboard memory. Considering this data should be transferred from the main memory to the videocard memory before processing and should be returned back into the main memory after processing. Data transfer is a bottleneck of image processing on GPU. We made fully separate API for CUDA to minimize time spent for data transfer. It is highly recommended to follow programming style described below:
- convert existing image to an image in CUDA namespace
- use image functions for newly created image in CUDA namespace
- convert back the image from CUDA namespace 

API description
---------------------------
Please refer to file **API_description.md** for full description of API.
