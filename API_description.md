# API description

## Namespaces
**Template_Image**    
Includes only one template class ***ImageTemplate*** what is the main class for image buffer classes. 

**Bitmap_Image**    
Declares template class for BITMAP images and concrete classes:
- ***Image*** - gray scale bitmap image (main class in most of image processing functions). If the type of image in description is not implicitly specified then it is a gray-scale image.
- ***ColorImage*** - RGB (color) bitmap image.    

**Image_Function**    
Contains all basic functions for image processing for any CPU. 

**Image_Function_Sse**    
Contains basic functions for image processing for CPUs with ***SSE 2*** support. 

**Image_Function_Avx**    
Contains basic functions for image processing for CPUs with ***AVX 2.0*** support.    

## Functions

All images in function parameter list must have width and height greater than 0 otherwise exception imageException is raised.
- **BitwiseAnd** [_Namespaces: **Image_Function, Image_Function_Sse, Image_Function_Avx**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;Image which is a result of bitwise AND. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
    
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

- **BitwiseOr** [_Namespaces: **Image_Function, Image_Function_Sse, Image_Function_Avx**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise OR. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
	
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

- **BitwiseXor** [_Namespaces: **Image_Function, Image_Function_Sse, Image_Function_Avx**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise XOR. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
	
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
		
- **Convert** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	void Convert(
		const Image & in,
		ColorImage & out
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
	void Convert(
		const ColorImage & in,
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
	void Convert(
		const Image & in,
		uint32_t startXIn,
		uint32_t startYIn,
		ColorImage & out,
		uint32_t startXOut,
		uint32_t startYOut,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area with [width, heigh] size into color image area with same size by setting gray-scale intensity (value) into every color channel (RGB).
	
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
		
	##### Syntax:
	```cpp
	void Convert(
		const ColorImage & in,
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
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image area with [width, heigh] size into color image area with same size by setting gray-scale intensity as an average value among red, green and blue channels of color image.
	
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
		
- **Copy** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	void Copy(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Copy image data from input to out image.
	
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
	&nbsp;&nbsp;&nbsp;&nbsp;Copy image data from input image area of [width, height] size to output image area of the same size.
	
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
	&nbsp;&nbsp;&nbsp;&nbsp;Returns a intensity on specified pixel position.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - position of pixel by X axis    
	&nbsp;&nbsp;&nbsp;&nbsp;y - position of pixel by Y axis    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;pixel intensity value. If the function fails exception imageException is raised.

- **Histogram** [_Namespaces: **Image_Function**_]

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
		
- **Invert** [_Namespaces: **Image_Function**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise NOT. Height and width of result image is the same as of input image. If the function fails exception imageException is raised.
	
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
	&nbsp;&nbsp;&nbsp;&nbsp;in - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT. Height and width of result image is the same as of input image    
	
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
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise NOT on image at area of [width, height] size and puts result into third image of the same size.
	
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
	
- **ParameterValidation** [_Namespaces: **Image_Function**_]

	##### Syntax:
	```cpp
	template <uint8_t bytes>
	void ParameterValidation(
		const BitmapImage <bytes> & image1
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether an image is allocated.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - an image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <uint8_t bytes1, uint8_t bytes2>
	void ParameterValidation(
		const BitmapImage <bytes1> & image1,
		const BitmapImage <bytes2> & image2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether two images are allocated and they are same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <uint8_t bytes1, uint8_t bytes2, uint8_t bytes3>
	void ParameterValidation(
		const BitmapImage <bytes1> & image1,
		const BitmapImage <bytes2> & image2,
		const BitmapImage <bytes3> & image3
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether three images are allocated and they are same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;image2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;image3 - third image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	template <uint8_t bytes>
	void ParameterValidation(
		const BitmapImage <bytes> & image,
		uint32_t startX,
		uint32_t startY,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether an image is allocated, image area [{startX, startY}, {startX + width, startY + height}] is withing image size, width and height are greater than 0.
	
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
	template <uint8_t bytes1, uint8_t bytes2>
	void ParameterValidation(
		const BitmapImage <bytes1> & image1,
		uint32_t startX1,
		uint32_t startY1,
		const BitmapImage <bytes2> & image2,
		uint32_t startX2,
		uint32_t startY2,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether two images are allocated, image area [{startX1, startY1}, {startX1 + width, startY1 + height}] is withing first image size, image area [{startX2, startY2}, {startX2 + width, startY2 + height}] is withing second image size, width and height are greater than 0.
	
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
	template <uint8_t bytes1, uint8_t bytes2, uint8_t bytes3>
	void ParameterValidation(
		const BitmapImage <bytes1> & image1,
		uint32_t startX1,
		uint32_t startY1,
		const BitmapImage <bytes2> & image2,
		uint32_t startX2,
		uint32_t startY2,
		const BitmapImage <bytes3> & image3,
		uint32_t startX3,
		uint32_t startY3,
		uint32_t width,
		uint32_t height
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Validates whether two images are allocated, image area [{startX1, startY1}, {startX1 + width, startY1 + height}] is withing first image size, image area [{startX2, startY2}, {startX2 + width, startY2 + height}] is withing second image size, image area [{startX3, startY3}, {startX3 + width, startY3 + height}] is withing third image size, width and height are greater than 0.
	
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
	&nbsp;&nbsp;&nbsp;&nbsp;Set intensity at specified pixel position.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;x - position of pixel by X axis    
	&nbsp;&nbsp;&nbsp;&nbsp;y - position of pixel by Y axis    
	&nbsp;&nbsp;&nbsp;&nbsp;value - intensity value    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
- **Threshold** [_Namespaces: **Image_Function**_]

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