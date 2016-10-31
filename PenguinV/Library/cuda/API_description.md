# API description

## Namespaces
**Bitmap_Image_Cuda**    
Declares template class for BITMAP images and concrete classes:
- ***ImageCuda*** - gray-scale bitmap image (main class in most of image processing functions). If the type of image in description is not implicitly specified then it is a gray-scale image.
- ***ColorImageCuda*** - RGB (color) bitmap image.    

**Image_Function_Cuda**    
Contains all basic functions for image processing by CUDA.    

**Template_Image_Cuda**    
Includes only one template class ***ImageTemplateCuda*** what is the main class for image buffer classes.    

## Functions

All images in function parameter list must have width and height greater than 0 otherwise exception imageException is raised.

- **BitwiseAnd** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	ImageCuda BitwiseAnd(
		const ImageCuda & in1,
		const ImageCuda & in2
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
		const ImageCuda & in1,
		const ImageCuda & in2,
		ImageCuda & out
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
	ImageCuda BitwiseOr(
		const ImageCuda & in1,
		const ImageCuda & in2
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
		const ImageCuda & in1,
		const ImageCuda & in2,
		ImageCuda & out
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
	ImageCuda BitwiseXor(
		const ImageCuda & in1,
		const ImageCuda & in2
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
		const ImageCuda & in1,
		const ImageCuda & in2,
		ImageCuda & out
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
		
- **Convert** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	void Convert(
		const Bitmap_Image::Image & in,
		ImageCuda & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image from main memory into gray-scale image in videocard memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image in main memory     
	&nbsp;&nbsp;&nbsp;&nbsp;out - a gray-scale image in videocard memory    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void Convert(
		const ImageCuda & in,
		Bitmap_Image::Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts gray-scale image from videocard memory into gray-scale image in main memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - a gray-scale image in videocard memory     
	&nbsp;&nbsp;&nbsp;&nbsp;out - a gray-scale image in main memory    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.

- **Invert** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	ImageCuda Invert(
		const ImageCuda & in
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
		const ImageCuda & in,
		ImageCuda & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Performs bitwise NOT on image and puts result into second image of the same size.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - first image    
	&nbsp;&nbsp;&nbsp;&nbsp;out - image which is a result of bitwise NOT. Height and width of result image is the same as of input image    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
