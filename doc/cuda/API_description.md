# API description

## Namespaces
**Bitmap_Image_Cuda**    
Declares a class for BITMAP images:
- ***Image*** - a bitmap image with default number of colors as 1 (gray-scale image). If the the number of color channels in this description is not implicitly specified then it is a 1 (gray-scale image).    

**Bitmap_Image_Cuda_Cpu**    
Declares a class for BITMAP images:
- ***Image*** - a bitmap pinned memory allocated image with default number of colors as 1 (gray-scale image). If the the number of color channels in this description is not implicitly specified then it is a 1 (gray-scale image).    

**Image_Function_Cuda**    
Contains all basic functions for image processing by CUDA.    

**Template_Image_Cuda**    
Includes only one template class ***ImageTemplateCuda*** which is the main class for image buffer classes.    

## Functions

All images in function parameter list must have width and height greater than 0 otherwise exception imageException is raised.

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
