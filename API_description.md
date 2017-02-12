# API description

## Namespaces
**Bitmap_Image**    
Declares a class for BITMAP images:
- ***Image*** - a bitmap image with default number of colors as 1 (gray-scale image). If the the number of color channels in this description is not implicitly specified then it is a 1 (gray-scale image).   

**Bitmap_Operation**    
Contains functions to load and save BITMAP images.  

**Blob_Detection**    
Contains structures and classes related to blob detection methods:
- ***Area*** - a structure representing an area of interest (AOI). The area is a rectangle area: {[left, top], [right, bottom]}.
- ***BlobDetection*** - a main class which performs blob detection on an input image.
- ***BlobInfo*** - a structure which stores all information related to individual blob. This is a result of BlobDetection class inspection.
- ***BlobParameters*** - a structure to contain parameters (criteria) needed for blob detection.
- ***Parameter*** - a template structure to represent a single parameter used in BlobParameters structure.
- ***Point*** - a structure which represents a mathematical point in 2D space [x, y].
- ***Value*** - a template structure used in BlobInfo structure to contain information about one found blob parameter.    

**Function_Pool**    
Contains basic functions for image processing for any CPU with multithreading support.    

**Image_Function**    
Contains all basic functions for image processing for any CPU.    

**Image_Function::Filtering**    
Contains functions for image filtering.    

**Image_Function_Avx**    
Contains basic functions for image processing for CPUs with ***AVX 2.0*** support.    

**Image_Function_Neon**    
Contains basic functions for image processing for CPUs with ***NEON*** support.    

**Image_Function_Sse**    
Contains basic functions for image processing for CPUs with ***SSE 2*** support.    

**Template_Image**    
Includes only one template class ***ImageTemplate*** which is the main class for image buffer classes.    

**Thread_Pool**    
Contains classes for multithreading using thread pool:
- ***AbstractTaskProvider*** - an abstract class which should do some tasks.
- ***TaskProvider*** - a concrete class which performs tasks and from which other classes are inherited to use thread pool.
- ***ThreadPool*** - a thread pool class which manages threads and tasks.
- ***ThreadPoolMonoid*** - a singleton (or monoid) class of thread pool which allows to use only 1 copy of thread pool inside application.
- ***TaskProviderSingleton*** - a concrete class which performs tasks and from which other classes are inherited to use thread pool's singleton.    

## Functions

All images in function parameter list must have width and height greater than 0 otherwise an exception imageException is raised.

- **AbsoluteDifference** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image AbsoluteDifference(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates absolute (by module) difference between pixel intensities on two images. Both images must be the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of absolute difference. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void AbsoluteDifference(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates absolute (by module) difference between pixel intensities on two images and puts result into third image. Three images must be same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of absolute difference    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image AbsoluteDifference(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates absolute (by module) difference between pixel intensities on two images within an area with [width, height].
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where absolute difference operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where absolute difference operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of absolute difference with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void AbsoluteDifference(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates absolute (by module) difference between pixel intensities on two images within an area with [width, height] and puts result into third image area of same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of subtraction    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where absolute difference operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where absolute difference operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If function fails exception imageException is raised.

- **Accumulate** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	void Accumulate(
		const Image & image,
		std::vector < uint32_t > & result
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Adds pixel intensity values to result array. The size of result array must be same as image size [width * height].
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an array    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Accumulate(
		const Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height,
		std::vector < uint32_t > & result
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Adds pixel intensity values of image on area of [width, height] size to result array. The size of result array must be same as image area [width * height].
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where accumulation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where accumulation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an array    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **BitwiseAnd** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image BitwiseAnd(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise AND on two images with equal height and width and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise AND. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void BitwiseAnd(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise AND on two images with equal height and width and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise AND    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image BitwiseAnd(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise AND on two images at area of [width, height] size and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation AND is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation AND is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise AND with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void BitwiseAnd(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise AND on two images at area of [width, height] size and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise AND    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation AND is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation AND is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If function fails exception imageException is raised.

- **BitwiseOr** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image BitwiseOr(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise OR on two images with equal height and width and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise OR. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void BitwiseOr(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise OR on two images with equal height and width and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise OR    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image BitwiseOr(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise OR on two images at area of [width, height] size and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation OR is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation OR is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise OR with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void BitwiseOr(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise OR on two images at area of [width, height] size and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise OR    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation OR is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation OR is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **BitwiseXor** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image BitwiseXor(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise XOR on two images with equal height and width and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise XOR. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void BitwiseXor(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise XOR on two images with equal height and width and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise XOR    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image BitwiseXor(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise XOR on two images at area of [width, height] size and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation XOR is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation XOR is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise XOR with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void BitwiseXor(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise XOR on two images at area of [width, height] size and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise XOR    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation XOR is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation XOR is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **CommonColorCount** [_Namespaces: **Image_Function**_]
	
	##### Syntax:
	```cpp
	template <typename TImage>
	uint8_t CommonColorCount(
		const TImage & image1,
		const TImage & image2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies that images have the same channel count and returns it as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;color count. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	template <typename TImage>
	uint8_t CommonColorCount(
		const TImage & image1,
		const TImage & image2,
		const TImage & image3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies that images have the same channel count and returns it as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - third image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;color count. If the function fails exception imageException is raised.
	
- **ConvertToGrayScale** [_Namespaces: **Function_Pool, Image_Function**_]
	
	##### Syntax:
	```cpp
	Image ConvertToGrayScale(
		const Image & in,
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts color image into gray-scale image with same width and height by setting gray-scale intensity as an average value among red, green and blue channels of color image.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;a gray-scale image with same width and height. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void ConvertToGrayScale(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts color image into gray-scale image with same width and height by setting gray-scale intensity as an average value among red, green and blue channels of color image.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a gray-scale image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image ConvertToGrayScale(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area with [width, height] size into color image with same size by setting gray-scale intensity as an average value among red, green and blue channels of color image.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;a gray-scale image with same width and height. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ConvertToGrayScale(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area with [width, height] size into color image area with same size by setting gray-scale intensity as an average value among red, green and blue channels of color image.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
- **ConvertToRgb** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	Image ConvertToRgb(
		const Image & in,
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image into color image with same width and height by setting gray-scale intensity (value) into every color channel (RGB).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;a color image with same width and height. If the function fails exception imageException is raised.

	##### Syntax:
	```cpp
	void ConvertToRgb(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image into color image with same width and height by setting gray-scale intensity (value) into every color channel (RGB).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a color image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image ConvertToRgb(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area with [width, height] size into color image with same size by setting gray-scale intensity (value) into every color channel (RGB).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;a color image with same width and height. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void ConvertToRgb(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area with [width, height] size into color image area with same size by setting gray-scale intensity (value) into every color channel (RGB).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
- **Copy** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	void Copy(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Copies image data from input to out image.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Copy(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Creates a copy of image area of [width, height] size and returns image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a copy of area of [width, height] size . If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Copy(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Copies image data from input image area of [width, height] size to output image area of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a copy of input area image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **ExtractChannel** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	Image ExtractChannel(
		const Image & in,
		uint8_t channelId
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Extracts one channel image from color image and returns a gray-scale image with the same size as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;channelId - channel ID (0, 1 or 2 for RGB images)    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a color component of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void  ExtractChannel(
		const Image & in,
		Image & out,
		uint8_t channelId
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Extracts one channel image from color image and put result into a gray-scale image with the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;channelId - channel ID (0, 1 or 2 for RGB images)    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	Image ExtractChannel(
		const Image & in,
		uint32_t x,
		uint32_t y,
		uint32_t width,
		uint32_t height,
		uint8_t channelId
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Extracts one channel image with [width, height] size from color image and returns a gray-scale image with the [width, height] size as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;channelId - channel ID (0, 1 or 2 for RGB images)    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a color component of input image. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ExtractChannel(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		uint8_t channelId
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Extracts one channel image with [width, height] size from color image and puts result into a gray-scale image with the [width, height] size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;channelId - channel ID (0, 1 or 2 for RGB images)    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Fill** [_Namespaces: **Image_Function**_]
	
	##### Syntax:
	```cpp
	void Fill(
		Image & image,
		uint8_t value
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Sets in all pixels of an image a specified value.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image      
	&nbsp;&nbsp;&nbsp;&nbsp;value - value to set    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Fill(
		Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height,
		uint8_t value
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Sets in all pixels within an image area of [width, height] size a specified value.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;value - value to set    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Flip** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	Image Flip(
		const Image & in,
		bool horizontal,
		bool vertical
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Flips image in one or both directions and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - specificator to set flip relatively to Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;vertical - specificator to set flip relatively to X axis    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which in a result of image flipping. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Flip(
		const Image & in,
		Image & out,
		bool horizontal,
		bool vertical
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Flips image in one or both directions and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image which in a result of image flipping. Height and width of result image are the same as of input image    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - specificator to set flip relatively to Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;vertical - specificator to set flip relatively to X axis    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Flip(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height,
		bool horizontal,
		bool vertical
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Flips an area of [width, height] size on image in one or both directions and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation NOT is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation NOT is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - specificator to set flip relatively to Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;vertical - specificator to set flip relatively to X axis    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of image flipping with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Flip(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		bool horizontal,
		bool vertical
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Flips an area of [width, height] size on image in one or both directions and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation NOT is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation NOT is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - specificator to set flip relatively to Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;vertical - specificator to set flip relatively to X axis    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **GammaCorrection** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	Image GammaCorrection(
		const Image & in,
		double a,
		double gamma
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs gamma correction on image and returns result image of the same size. Gamma correction works by formula: output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;a - A coefficient    
	&nbsp;&nbsp;&nbsp;&nbsp;gamma - gamma coefficient    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of gamma correction. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void GammaCorrection(
		const Image & in,
		Image & out,
		double a,
		double gamma
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs gamma correction on image and puts result into second image of the same size. Gamma correction works by formula: output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image which is a result of gamma correction. Height and width of result image are the same as of input image    
	&nbsp;&nbsp;&nbsp;&nbsp;a - A coefficient    
	&nbsp;&nbsp;&nbsp;&nbsp;gamma - gamma coefficient    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image GammaCorrection(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height,
		double a,
		double gamma
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs gamma correction on image at area of [width, height] size and returns result image of the same size. Gamma correction works by formula: output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where gamma correction is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where gamma correction is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;a - A coefficient    
	&nbsp;&nbsp;&nbsp;&nbsp;gamma - gamma coefficient    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of gamma correction with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void GammaCorrection(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		double a,
		double gamma
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs gamma correction on image at area of [width, height] size and puts result into second image of the same size. Gamma correction works by formula: output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where gamma correction is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where gamma correction is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;a - A coefficient    
	&nbsp;&nbsp;&nbsp;&nbsp;gamma - gamma coefficient    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **GetPixel** [_Namespaces: **Image_Function**_]
	
	##### Syntax:
	```cpp
	uint8_t GetPixel(
		const Image & image,
		uint32_t x,
		uint32_t y
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Returns an intensity on specified pixel position.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - position of pixel by X axis    
	&nbsp;&nbsp;&nbsp;&nbsp;y - position of pixel by Y axis    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;pixel intensity value. If the function fails exception imageException is raised.

- **GetThreshold** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	uint8_t GetThreshold(
		const std::vector < uint32_t > & histogram
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Returns optimal threshold value between background and foreground.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image histogram    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;threshold value. If the function fails exception imageException is raised.
		
- **Histogram** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	std::vector < uint32_t > Histogram(
		const Image & image
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a histogram of pixel intensities of image and return an array what is a histogram with fixed size of 256 elements.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;array as a histogram of pixel intensities. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Histogram(
		const Image & image,
		std::vector < uint32_t > & histogram
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a histogram of pixel intensities of image and stores result into output array with fixed size of 256 elements. No requirement that an array (vector) must be resized before calling this function.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;histogram - an array what is histogram of pixel intensities    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	std::vector < uint32_t > Histogram(
		const Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a histogram of pixel intensities of image area of [width, height] size and return an array what is a histogram with fixed size of 256 elements.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;array as a histogram of pixel intensities. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Histogram(
		const Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height,
		std::vector < uint32_t > & histogram
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a histogram of pixel intensities of image area of [width, height] size and stores result into output array with fixed size of 256 elements. No requirement that an array (vector) must be resized before calling this function.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;histogram - an array what is histogram of pixel intensities    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
- **Invert** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image Invert(
		const Image & in
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise NOT on image and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise NOT. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Invert(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise NOT on image and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image which is a result of bitwise NOT. Height and width of result image are the same as of input image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Invert(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise NOT on image at area of [width, height] size and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation NOT is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation NOT is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise NOT with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Invert(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise NOT on image at area of [width, height] size and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where Bitwise operation NOT is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where Bitwise operation NOT is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **IsCorrectColorCount** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	template <typename TImage>
	bool IsCorrectColorCount(
		const TImage & image
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether an image has a valid color channel count: 1 (gray-scale image) or 3 (RGB).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;true in a case of correct color channel count and false if it is not.
	
- **IsEqual** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	bool IsEqual(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Compares two images with same size byte by byte and returns true if both images contain same pixel intensities (values).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;comparison result. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	bool IsEqual(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Compares two image areas with same [width, height] size byte by byte and returns true if both images areas contain same pixel intensities (values).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where comparison is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where comparison is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;comparison result. If the function fails exception imageException is raised.
	
- **Load** [_Namespaces: **Bitmap_Operation**_]
	
	##### Syntax:
	```cpp
	BitmapRawImage Load(
		std::string path
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Returns raw image data readed from file.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;path - a path of bitmap image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;raw bitmap data class. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Load(
		std::string path,
		BitmapRawImage & raw
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Read raw image data from file and stores it into raw bitmap data class.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;path - a path of bitmap image    
	&nbsp;&nbsp;&nbsp;&nbsp;raw - raw image data class    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **LookupTable** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	Image LookupTable(
		const Image & in,
		const std::vector < uint8_t > & table
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Replaces pixel intensities values by values stored in lookup table and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;table - a lookup table    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of pixel intensity transformation. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void  LookupTable(
		const Image & in,
		Image & out,
		const std::vector < uint8_t > & table
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Replaces pixel intensities values by values stored in lookup table and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;table - a lookup table    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of transformation. Height and width of result image are the same as of input image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image LookupTable(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height,
		const std::vector < uint8_t > & table
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Replaces pixel intensities values by values stored in lookup table at area of [width, height] size and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where transformation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where transformation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;table - a lookup table    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of transformation with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void  LookupTable(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		const std::vector < uint8_t > & table
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Replaces pixel intensities values by values stored in lookup table at area of [width, height] size and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where transformation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where transformation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;table - a lookup table    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Maximum** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image Maximum(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds maximum value between two images with equal height and width and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of maximum operation. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void Maximum(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds maximum value between two images with equal height and width and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of maximum operation    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Maximum(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds maximum value between two images at area of [width, height] size and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where maximum operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where maximum operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of maximum operation with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Maximum(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds maximum value between two images at area of [width, height] size and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of maximum operation    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where maximum operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where maximum operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If function fails exception imageException is raised.
	
- **Median** [_Namespaces: **Image_Function::Filtering**_]

	##### Syntax:
	```cpp
	Image Median(
		const Image & in,
		uint32_t kernelSize
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs median filtering on image and returns result image of the same size. Kernel size must be odd and greater than 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;kernelSize - a kernel size    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of median filtering. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Median(
		const Image & in,
		Image & out,
		uint32_t kernelSize
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs median filtering on image and puts result into second image of the same size. Kernel size must be odd and greater than 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image which is a result of median filtering. Height and width of result image are the same as of input image    
	&nbsp;&nbsp;&nbsp;&nbsp;kernelSize - a kernel size    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Median(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height,
		uint32_t kernelSize
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs median filtering on image at area of [width, height] size and returns result image of the same size. Kernel size must be odd and greater than 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where median filtering is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where median filtering is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;kernelSize - a kernel size    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of median filtering with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Median(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		uint32_t kernelSize
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs median filtering on image at area of [width, height] size and puts result into second image of the same size. Kernel size must be odd and greater than 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where median filtering is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where median filtering is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;kernelSize - a kernel size    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **Merge** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	Image Merge(
		const Image & in1,
		const Image & in2,
		const Image & in3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Merges 3 gray-scale images into one 3-color image with same width and height by setting gray-scale intensity (value) into color channels and returns it as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;in3 - third gray-scale image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;a color image. If the function fails exception imageException is raised.

	##### Syntax:
	```cpp
	void Merge(
		const Image & in1,
		const Image & in2,
		const Image & in3,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Merges 3 gray-scale images into one 3-color image with same width and height by setting gray-scale intensity (value) into color channels.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;in3 - third gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a color image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Merge(
		const Image & in1,
		uint32_t startXIn1,
		uint32_t startYIn1,
		const Image & in2,
		uint32_t startXIn2,
		uint32_t startYIn2,
		const Image & in3,
		uint32_t startXIn3,
		uint32_t startYIn3,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Merges 3 gray-scale images within an area of [width, height] size into one 3-color image with the same size by setting gray-scale intensity (value) into color channels and returns it as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn1 - start X position of first gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn1 - start Y position of first gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn2 - start X position of second gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn2 - start Y position of second gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in3 - third gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn3 - start X position of third gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn3 - start Y position of third gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;a color image of [width, height] size. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Merge(
		const Image & in1,
		uint32_t startXIn1,
		uint32_t startYIn1,
		const Image & in2,
		uint32_t startXIn2,
		uint32_t startYIn2,
		const Image & in3,
		uint32_t startXIn3,
		uint32_t startYIn3,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Merges 3 gray-scale images within an area of [width, height] size into one 3-color image with the same size by setting gray-scale intensity (value) into color channels.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn1 - start X position of first gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn1 - start Y position of first gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn2 - start X position of second gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn2 - start Y position of second gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in3 - third gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn3 - start X position of third gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn3 - start Y position of third gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Minimum** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image Minimum(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds minimum value between two images with equal height and width and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of minimum operation. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void Minimum(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds minimum value between two images with equal height and width and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of minimum operation    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Minimum(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds minimum value between two images at area of [width, height] size and returns a result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where minimum operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where minimum operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of minimum operation with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Minimum(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Finds minimum value between two images at area of [width, height] size and puts result into third image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of minimum operation    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where minimum operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where minimum operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If function fails exception imageException is raised.
	
- **Normalize** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	Image Normalize(
		const Image & in
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs image normalization in range from 0 to 255 and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of normalization. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Normalize(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs image normalization in range from 0 to 255 and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image which is a result of normalization. Height and width of result image are the same as of input image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Normalize(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs image normalization in range from 0 to 255 at area of [width, height] size and returns result image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where normalization is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where normalization is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of normalization with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Normalize(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs image normalization in range from 0 to 255 at area of [width, height] size and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where normalization is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where normalization is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **ParameterValidation** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	template <typename TImage>
	void ParameterValidation(
		const TImage & image1
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether an image is allocated and has a valid color channel count.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <typename TImage>
	void ParameterValidation(
		const TImage & image1,
		const TImage & image2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether two images are allocated, they are same size and have a valid color channel count.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <typename TImage>
	void ParameterValidation(
		const TImage & image1,
		const TImage & image2,
		const TImage & image3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether three images are allocated, they are same size and have a valid color channel count.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;image3 - third image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <typename TImage>
	void ParameterValidation(
		const TImage & image,
		uint32_t startX,
		uint32_t startY,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether an image is allocated, image area [{startX, startY}, {startX + width, startY + height}] is within image size, width and height are greater than 0 and it has a valid color channel count.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <typename TImage>
	void ParameterValidation(
		const TImage & image1,
		uint32_t startX1,
		uint32_t startY1,
		const TImage & image2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether two images are allocated, image area [{startX1, startY1}, {startX1 + width, startY1 + height}] is within first image size, image area [{startX2, startY2}, {startX2 + width, startY2 + height}] is within second image size, width and height are greater than 0 and images have a valid color channel count.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of first image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of first image area    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of second image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of second image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <typename TImage>
	void ParameterValidation(
		const TImage & image1,
		uint32_t startX1,
		uint32_t startY1,
		const TImage & image2,
		uint32_t startX2,
		uint32_t startY2,
		const TImage & image3,
		uint32_t startX3,
		uint32_t startY3,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether two images are allocated, image area [{startX1, startY1}, {startX1 + width, startY1 + height}] is within first image size, image area [{startX2, startY2}, {startX2 + width, startY2 + height}] is within second image size, image area [{startX3, startY3}, {startX3 + width, startY3 + height}] is within third image size, width and height are greater than 0 and images have a valid color channel count.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of first image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of first image area    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of second image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of second image area    
	&nbsp;&nbsp;&nbsp;&nbsp;image3 - third image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX3 - start X position of third image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY3 - start Y position of third image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **ProjectionProfile** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	std::vector < uint32_t > ProjectionProfile(
		const Image & image,
		bool horizontal
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a projection profile of pixel intensities (sum) of image on one of axes and return an array what is an array with image width or height size respectively on chosen axis. Projection on X axis is performed if horizontal parameter is true else projection on Y axis.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - axis type    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;array of pixel intensities sums. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ProjectionProfile(
		const Image & image,
		bool horizontal,
		std::vector < uint32_t > & projection
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a projection profile of pixel intensities (sum) of image on one of axes and stores result into output array what is an array with image width or height size respectively on chosen axis. No requirement that an array (vector) must be resized before calling this function. Projection on X axis is performed if horizontal parameter is true else projection on Y axis.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - axis type    
	&nbsp;&nbsp;&nbsp;&nbsp;projection - an array of pixel intensities sums    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	std::vector < uint32_t > ProjectionProfile(
		const Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height,
		bool horizontal
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a projection profile of pixel intensities (sum) of image area of [width, height] size on one of axes and return an array what is an array with [width] or [height] size respectively on chosen axis. Projection on X axis is performed if horizontal parameter is true else projection on Y axis.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - axis type    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;array of pixel intensities sums. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ProjectionProfile(
		const Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height,
		bool horizontal,
		std::vector < uint32_t > & projection
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a projection profile of pixel intensities (sum) of image area of [width, height] size on one of axes and stores result into output array what is an array with [width] or [height] size respectively on chosen axis. No requirement that an array (vector) must be resized before calling this function. Projection on X axis is performed if horizontal parameter is true else projection on Y axis.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;horizontal - axis type    
	&nbsp;&nbsp;&nbsp;&nbsp;projection - an array of pixel intensities sums    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **RgbToBgr** [_Namespaces: **Function_Pool, Image_Function**_]

	##### Syntax:
	```cpp
	Image RgbToBgr(
		const Image & in
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts a 3-colored image from RGB represantation to a 3-colored image in BGR representation (inverting color channel order).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an input image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of conversion. Height and width of result image are the same as of input image. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void RgbToBgr(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts a 3-colored image from RGB represantation and put the result into a 3-colored image of the same size in BGR representation (inverting color channel order).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image which is a result of conversion. Height and width of result image are the same as of input image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image RgbToBgr(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts a 3-colored image at area of [width, height] size from RGB represantation to a 3-colored image of the the same size in BGR representation (inverting color channel order).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where conversion is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where conversion is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of conversion with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void RgbToBgr(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts a 3-colored image at area of [width, height] size from RGB represantation and put the result into a 3-colored image of the the same size in BGR representation (inverting color channel order).
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of conversion    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where conversion is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where conversion is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Resize** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	Image Resize(
		const Image & in,
		uint32_t widthOut,
		uint32_t heightOut
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Resizes (scales) an image to [widthOut, heightOut] size and returns result image of the scaled size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;widthOut - width of output image    
	&nbsp;&nbsp;&nbsp;&nbsp;heightOut - height of output image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image of [widthOut, heightOut] size which is a result of resizing. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Resize(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Resizes (scales) one image to second image with (probably) different size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image as a result of resizing    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Resize(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t widthIn,
		uint32_t heightIn,
		uint32_t widthOut,
		uint32_t heightOut
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Resizes (scales) image area of [width, height] size to [widthOut, heightOut] size and returns result image of the scaled size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;widthIn - width of image area from what image will be resized    
	&nbsp;&nbsp;&nbsp;&nbsp;heightIn - height of image area from what image will be resized    
	&nbsp;&nbsp;&nbsp;&nbsp;widthOut - width of image area to what image will be resized    
	&nbsp;&nbsp;&nbsp;&nbsp;heightOut - height of image area to what image will be resized    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image of [widthOut, heightOut] size which is a result of resizing. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Resize(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t widthIn,
		uint32_t heightIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t widthOut,
		uint32_t heightOut
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Resizes (scales) image area of [width, height] size to [widthOut, heightOut] size of second image.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;widthIn - width of image area from what image will be resized    
	&nbsp;&nbsp;&nbsp;&nbsp;heightIn - height of image area from what image will be resized    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of transpose    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;widthOut - width of image area to what image will be resized    
	&nbsp;&nbsp;&nbsp;&nbsp;heightOut - height of image area to what image will be resized    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Rotate** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	void Rotate(
		const Image & in,
		double centerXIn,
		double centerYIn,
		Image & out,
		double centerXOut,
		double centerYOut,
		double angle
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Rotates an input image around specified center and puts results into an output image. Rotation center on output image could be different compate to input image center.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an input image    
	&nbsp;&nbsp;&nbsp;&nbsp;centerXIn - X position of rotation center on input image     
	&nbsp;&nbsp;&nbsp;&nbsp;centerYIn - Y position of rotation center on input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - an output image    
	&nbsp;&nbsp;&nbsp;&nbsp;centerXOut - X position of rotation center on output image    
	&nbsp;&nbsp;&nbsp;&nbsp;centerYOut - Y position of rotation center on output image    
	&nbsp;&nbsp;&nbsp;&nbsp;angle - rotation anle    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Save** [_Namespaces: **Bitmap_Operation**_]
	
	##### Syntax:
	```cpp
	void Save(
		std::string path,
		Template_Image::ImageTemplate < uint8_t > & image
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Saves image into bitmap file.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;path - a path where to save an image    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Save(
		std::string path,
		Template_Image::ImageTemplate < uint8_t > & image,
		uint32_t startX,
		uint32_t startY,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Saves image area of [width, height] size into bitmap file.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;path - a path where to save an image    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **SetPixel** [_Namespaces: **Image_Function**_]
	
	##### Syntax:
	```cpp
	void SetPixel(
		Image & image,
		uint32_t x,
		uint32_t y,
		uint8_t value
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Sets intensity at specified pixel position.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - position of pixel by X axis    
	&nbsp;&nbsp;&nbsp;&nbsp;y - position of pixel by Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;value - intensity value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void SetPixel(
		Image & image,
		const std::vector < uint32_t > & X,
		const std::vector < uint32_t > & Y,
		uint8_t value
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Sets intensity at specified pixel positions.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;X - an array of pixel positions by X axis    
	&nbsp;&nbsp;&nbsp;&nbsp;Y - an array of pixel positions by Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;value - intensity value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Split** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	void Split(
		const Image & in,
		Image & out1,
		Image & out2,
		Image & out3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Splits a single 3-color image into 3 gray-scale images with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;out1 - first gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;out2 - second gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;out3 - third gray-scale image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Split(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out1,
		uint32_t startXOut1,
		uint32_t startYOut1,
		Image & out2,
		uint32_t startXOut2,
		uint32_t startYOut2,
		Image & out3,
		uint32_t startXOut3,
		uint32_t startYOut3,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Splits a single 3-color image within an area of [width, height] size into 3 gray-scale images with the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a color image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a color image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out1 - first gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut1 - start X position of first gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut1 - start Y position of first gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out2 - second gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut2 - start X position of second gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut2 - start Y position of second gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out3 - third gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut3 - start X position of third gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut3 - start Y position of third gray-scale image area    
	
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Subtract** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image Subtract(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Subtracts second image pixel intensities from first image pixel intensities (difference between two values). Both images must be same size. If pixel intensity at first image is less than pixel intensity at second image the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of subtraction. Height and width of result image are the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void Subtract(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Subtracts second image pixel intensities from first image pixel intensities (difference between two values) and puts result into third image. Three images must be same size. If pixel intensity at first image is less than pixel intensity at second image the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of subtraction    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Subtract(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Subtracts second image pixel intensities on an area with [width, height] size from first image pixel intensities on an area with same size (difference between two values). If pixel intensity at first image is less than pixel intensity at second image the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where subtraction operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where subtraction operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of subtraction with size [width, height]. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Subtract(
		const Image & in1,
		uint32_t startX1,
		uint32_t startY1,
		const Image & in2,
		uint32_t startX2,
		uint32_t startY2,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Subtracts second image pixel intensities on an area with [width, height] size from first image pixel intensities on an area with same size (difference between two values) and puts result into third image area of same size. If pixel intensity at first image is less than pixel intensity at second image the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX1 - start X position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY1 - start Y position of in1 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;startX2 - start X position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startY2 - start Y position of in2 image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of subtraction    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where subtraction operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where subtraction operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If function fails exception imageException is raised.
	
- **Sum** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Sse**_]
	
	##### Syntax:
	```cpp
	uint32_t Sum(
		const Image & image
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a sum of all pixel intensities at image and returns this value as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image      
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;sum of all pixel intensities. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	uint32_t Sum(
		const Image & image,
		uint32_t x,
		int32_t y,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates a sum of all pixel intensities at image area of [width, height] size and returns this value as a result.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;y - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of an image area    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;sum of all pixel intensities. If the function fails exception imageException is raised.
	
- **Threshold** [_Namespaces: **Function_Pool, Image_Function, Image_Function_Avx, Image_Function_Neon, Image_Function_Sse**_]

	##### Syntax:
	```cpp
	Image Threshold(
		const Image & in,
		uint8_t threshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image into pseudo binary image based on threshold and returns an image as a result of this operation with same size as input image. Thresholding works in such way:
	- if pixel intensity on input image is less ( < ) than threshold then set pixel intensity on output image as 0
	- if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;threshold - threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image as a result of thresholding. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	Image Threshold(
		const Image & in,
		uint8_t minThreshold,
		uint8_t maxThreshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image into pseudo binary image based on threshold and returns an image as a result of this operation with same size as input image. Thresholding works in such way: if pixel intensity on input image is less ( < ) than minimum threshold or more ( > ) than maximum threshold then set pixel intensity on output image as 0 otherwise set pixel intensity on output image as 255.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;minThreshold - minimum threshold value    
	&nbsp;&nbsp;&nbsp;&nbsp;maxThreshold - maximum threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image as a result of thresholding. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Threshold(
		const Image & in,
		Image & out,
		uint8_t threshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image into pseudo binary image based on threshold and puts result into output image with same size as input image. Thresholding works in such way:
	- if pixel intensity on input image is less ( < ) than threshold then set pixel intensity on output image as 0
	- if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image what is a result of thresholding    
	&nbsp;&nbsp;&nbsp;&nbsp;threshold - threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Threshold(
		const Image & in,
		Image & out,
		uint8_t minThreshold,
		uint8_t maxThreshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image into pseudo binary image based on threshold and puts result into output image with same size as input image. Thresholding works in such way: if pixel intensity on input image is less ( < ) than minimum threshold or more ( > ) than maximum threshold then set pixel intensity on output image as 0 otherwise set pixel intensity on output image as 255.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image what is a result of thresholding    
	&nbsp;&nbsp;&nbsp;&nbsp;minThreshold - minimum threshold value    
	&nbsp;&nbsp;&nbsp;&nbsp;maxThreshold - maximum threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	Image Threshold(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height,
		uint8_t threshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area of [width, height] size into pseudo binary image based on threshold and returns an image as a result of this operation with same area size. Thresholding works in such way:
	- if pixel intensity on input image is less ( < ) than threshold then set pixel intensity on output image as 0
	- if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;threshold - threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image as a result of thresholding. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	Image Threshold(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height,
		uint8_t minThreshold,
		uint8_t maxThreshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area of [width, height] size into pseudo binary image based on threshold and returns an image as a result of this operation with same area size. Thresholding works in such way: if pixel intensity on input image is less ( < ) than minimum threshold or more ( > ) than maximum threshold then set pixel intensity on output image as 0 otherwise set pixel intensity on output image as 255.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;minThreshold - minimum threshold value    
	&nbsp;&nbsp;&nbsp;&nbsp;maxThreshold - maximum threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image as a result of thresholding. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Threshold(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		uint8_t threshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area of [width, height] size into pseudo binary image based on threshold and puts result into output image with same area size as input image. Thresholding works in such way:
	- if pixel intensity on input image is less ( < ) than threshold then set pixel intensity on output image as 0
	- if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image what is a result of thresholding    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;threshold - threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Threshold(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height,
		uint8_t minThreshold,
		uint8_t maxThreshold
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area of [width, height] size into pseudo binary image based on threshold and puts result into output image with same area size as input image. Thresholding works in such way: if pixel intensity on input image is less ( < ) than minimum threshold or more ( > ) than maximum threshold then set pixel intensity on output image as 0 otherwise set pixel intensity on output image as 255.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of a gray-scale image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image what is a result of thresholding    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;minThreshold - minimum threshold value    
	&nbsp;&nbsp;&nbsp;&nbsp;maxThreshold - maximum threshold value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Transpose** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	Image Transpose(
		const Image & in
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Swaps columns and rows in input image {of [width, height] size} and returns result image {of the [height, width] size}.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image       
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image {of [height, width] size} which is a result of image (matrix) transpose. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Transpose(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Swaps columns and rows in first image {of [width, height] size} and puts result into second image {of the [height, width] size}.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - input image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - output image as a result of transpose    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	Image Transpose(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Swaps columns and rows in image area of [width, height] size and returns result image of the [height, width] size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of an image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area       
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image of [height, width] size which is a result of image (matrix) transpose. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	void Transpose(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		Image & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Swaps columns and rows in first image {of [width, height] size} and puts result into second image of the [height, width] size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;startXIn - start X position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYIn - start Y position of input image area    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of transpose    
	&nbsp;&nbsp;&nbsp;&nbsp;startXOut - start X position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;startYOut - start Y position of out image area    
	&nbsp;&nbsp;&nbsp;&nbsp;width - width of image area where transpose operation is performed    
	&nbsp;&nbsp;&nbsp;&nbsp;height - height of image area where transpose operation is performed    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **VerifyColoredImage** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	template <typename TImage>
	void VerifyColoredImage(
		const TImage & image1
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether an image has a valid color channel count as a RGB image - 3.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	template <typename TImage>
	void VerifyColoredImage(
		const TImage & image1,
		const TImage & image2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether images have a valid color channel count as a RGB image - 3.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	template <typename TImage>
	void VerifyColoredImage(
		const TImage & image1,
		const TImage & image2,
		const TImage & image3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether images have a valid color channel count as a RGB image - 3.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - third image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **VerifyGrayScaleImage** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	template <typename TImage>
	void VerifyGrayScaleImage(
		const TImage & image1
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether an image has a valid color channel count as a gray-scale image - 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	template <typename TImage>
	void VerifyGrayScaleImage(
		const TImage & image1,
		const TImage & image2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether images have a valid color channel count as a gray-scale image - 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
	##### Syntax:
	```cpp
	template <typename TImage>
	void VerifyGrayScaleImage(
		const TImage & image1,
		const TImage & image2,
		const TImage & image3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Verifies whether images have a valid color channel count as a gray-scale image - 1.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - third image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	