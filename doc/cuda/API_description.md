# API description

## Namespaces
**penguinV**    
- ***ImageCuda*** - a 8-bit image with default number of colors as 1 (gray-scale image). If the the number of color channels in this description is not implicitly specified then it is a 1 (gray-scale image).    

- ***ImageCudaPinned*** - a 8-bit pinned memory allocated image with default number of colors as 1 (gray-scale image). If the the number of color channels in this description is not implicitly specified then it is a 1 (gray-scale image).    

**Image_Function_Cuda**    
Contains all basic functions for image processing by CUDA.    

## Functions

All images in function parameter list must have width and height greater than 0 otherwise exception imageException is raised.

- **ConvertFromCuda** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	Image ConvertFromCuda(
		const Image & in
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from GPU memory into an image in main memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in GPU memory        
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image in main memory. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ConvertFromCuda(
		const Image & in,
		Image & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from GPU memory into an image in main memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in GPU memory     
	&nbsp;&nbsp;&nbsp;&nbsp;out - an image in main memory    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
	
- **ConvertToCuda** [_Namespaces: **Image_Function_Cuda**_]

	##### Syntax:
	```cpp
	ImageCuda ConvertToCuda(
		const Image & in,
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from main memory into an image in GPU memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in main memory      
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;image in GPU memory. If the function fails exception imageException is raised.
		
	##### Syntax:
	```cpp
	void ConvertToCuda(
		const Image & in,
		ImageCuda & out
	);
	```
	**Description:**    
	&nbsp;&nbsp;&nbsp;&nbsp;Converts an image from main memory into an image in GPU memory with same width and height.
	
	**Parameters:**    
	&nbsp;&nbsp;&nbsp;&nbsp;in - an image in main memory     
	&nbsp;&nbsp;&nbsp;&nbsp;out - an image in GPU memory    
	
	**Return value:**    
	&nbsp;&nbsp;&nbsp;&nbsp;void. If the function fails exception imageException is raised.
