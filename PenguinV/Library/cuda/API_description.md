# API description

## Namespaces
**Bitmap_Image_Cuda**    
Declares a class for BITMAP images:
- ***Image*** - a bitmap image with default number of colors as 1 (gray-scale image). If the the number of color channels in this description is not implicitly specified then it is a 1 (gray-scale image).    

**Image_Function_Cuda**    
Contains all basic functions for image processing by CUDA.    

**Template_Image_Cuda**    
Includes only one template class ***ImageTemplateCuda*** which is the main class for image buffer classes.    

## Functions

All images in function parameter list must have width and height greater than 0 otherwise exception imageException is raised.

- **AbsoluteDifference** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	Image AbsoluteDifference(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates absolute (by module) difference between pixel intensities on two images. Both images must be same size. If first image pixel intensity less than second image pixel intensity the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of absolute difference. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void AbsoluteDifference(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Calculates absolute (by module) difference between pixel intensities on two images and puts result into third image. Three images must be same size. If first image pixel intensity less than second image pixel intensity the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of absolute difference    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **BitwiseAnd** [_Namespaces: **Image_Function_Cuda**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of bitwise AND. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
    
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

- **BitwiseOr** [_Namespaces: **Image_Function_Cuda**_]

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

- **BitwiseXor** [_Namespaces: **Image_Function_Cuda**_]

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
	
- **ConvertFromCuda** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	Bitmap_Image::Image ConvertFromCuda(
		const Image & in
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from videocard memory into an image in main memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in videocard memory        
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image in main memory. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ConvertFromCuda(
		const Image & in,
		Bitmap_Image::Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from videocard memory into an image in main memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in videocard memory     
	&nbsp;&nbsp;&nbsp;&nbsp;out - an image in main memory    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **ConvertToCuda** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	void ConvertToCuda(
		const Bitmap_Image::Image & in,
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from main memory into an image in videocard memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in main memory      
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image in videocard memory. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ConvertToCuda(
		const Bitmap_Image::Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from main memory into an image in videocard memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in main memory     
	&nbsp;&nbsp;&nbsp;&nbsp;out - an image in videocard memory    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **ConvertToGrayScale** [_Namespaces: **Image_Function_Cuda**_]
	
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
	
- **GammaCorrection** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	Image GammaCorrection(
		const Image & in,
		double a,
		double gamma
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs gamma correction on image and returns result image of the same size. Gamma correction works by formula: output = A * (input ^ gamma), where A - multiplication, gamma - power base. Both values must be greater than 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image    
	&nbsp;&nbsp;&nbsp;&nbsp;a - A coefficient    
	&nbsp;&nbsp;&nbsp;&nbsp;gamma - gamma coefficient    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of gamma correction. Height and width of result image is the same as of input image. If the function fails exception imageException is raised.
	
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
	&nbsp;&nbsp;&nbsp;&nbsp;Performs gamma correction on image and puts result into second image of the same size. Gamma correction works by formula: output = A * (input ^ gamma), where A - multiplication, gamma - power base. Both values must be greater than 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of gamma correction. Height and width of result image is the same as of input image    
	&nbsp;&nbsp;&nbsp;&nbsp;a - A coefficient    
	&nbsp;&nbsp;&nbsp;&nbsp;gamma - gamma coefficient    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Invert** [_Namespaces: **Image_Function_Cuda**_]

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

- **LookupTable** [_Namespaces: **Image_Function_Cuda**_]

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
	
- **Maximum** [_Namespaces: **Image_Function_Cuda**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of maximum operation. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
    
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
	
- **Minimum** [_Namespaces: **Image_Function_Cuda**_]

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
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of minimum operation. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
    
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
	
- **Subtract** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	Image Subtract(
		const Image & in1,
		const Image & in2
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Subtracts second image pixel intensities from first image pixel intensities (difference between two values). Both images must be same size. If first image pixel intensity less than second image pixel intensity the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image which is a result of subtraction. Height and width of result image is the same as of input images. If the function fails exception imageException is raised.
    
	##### Syntax:
	```cpp
	void Subtract(
		const Image & in1,
		const Image & in2,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Subtracts second image pixel intensities from first image pixel intensities (difference between two values) and puts result into third image. Three images must be same size. If first image pixel intensity less than second image pixel intensity the result pixel intensity will be 0.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in1 - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;in2 - second image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of subtraction    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **Threshold** [_Namespaces: **Image_Function_Cuda**_]

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
