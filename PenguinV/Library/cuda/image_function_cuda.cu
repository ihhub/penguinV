#include <cuda_runtime.h>
#include "../image_function.h"
#include "image_function_cuda.cuh"

namespace
{
	// Helper function which should return proper arguments for CUDA device functions
	void getKernelParameters(int & threadsPerBlock, int & blocksPerGrid, uint32_t size)
	{
		if( size < 256 ) {
			threadsPerBlock = size;
			blocksPerGrid = 1;
		}
		else {
			threadsPerBlock = 256;
			blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		}
	};

	// The list of CUDA device functions
	__global__ void bitwiseAnd(const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size)
	{
	    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
	
	    if (i < size) {
	        out[i] = in1[i] & in2[i];
	    }
	};

	__global__ void bitwiseOr(const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size)
	{
	    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
	
	    if (i < size) {
	        out[i] = in1[i] | in2[i];
	    }
	};

	__global__ void bitwiseXor(const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size)
	{
	    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
	
	    if (i < size) {
	        out[i] = in1[i] ^ in2[i];
	    }
	};

	__global__ void invert(const uint8_t * in, uint8_t * out, uint32_t size)
	{
	    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
	
	    if (i < size) {
	        out[i] = ~in[i];
	    }
	};
};

namespace Image_Function_Cuda
{
	template <uint8_t bytes>
	void ParameterValidation( const BitmapImageCuda <bytes> & image1 )
	{
		if( image1.empty() )
			throw imageException("Bad input parameters in image function");
	}

	template <uint8_t bytes1, uint8_t bytes2>
	void ParameterValidation( const BitmapImageCuda <bytes1> & image1, const BitmapImageCuda <bytes2> & image2 )
	{
		if( image1.empty() || image2.empty() || image1.width() != image2.width() || image1.height() != image2.height() )
			throw imageException("Bad input parameters in image function");
	}

	template <uint8_t bytes1, uint8_t bytes2, uint8_t bytes3>
	void ParameterValidation( const BitmapImageCuda <bytes1> & image1, const BitmapImageCuda <bytes2> & image2, const BitmapImageCuda <bytes3> & image3 )
	{
		if( image1.empty() || image2.empty() || image3.empty() || image1.width() != image2.width() || image1.height() != image2.height() ||
			image1.width() != image3.width() || image1.height() != image3.height() )
			throw imageException("Bad input parameters in image function");
	}


	ImageCuda BitwiseAnd( const ImageCuda & in1, const ImageCuda & in2 )
	{
		ParameterValidation( in1, in2 );

		ImageCuda out( in1.width(), in1.height() );

		BitwiseAnd( in1, in2, out );

		return out;
	}

	void BitwiseAnd( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out )
	{
		ParameterValidation( in1, in2, out );

		int threadsPerBlock = 1, blocksPerGrid = 1;
		getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

		bitwiseAnd<<<blocksPerGrid, threadsPerBlock>>>( in1.data(), in2.data(), out.data(), out.width() * out.height() );
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
			throw imageException("Failed to launch CUDA kernel");
	}

	ImageCuda BitwiseOr( const ImageCuda & in1, const ImageCuda & in2 )
	{
		ParameterValidation( in1, in2 );

		ImageCuda out( in1.width(), in1.height() );

		BitwiseOr( in1, in2, out );

		return out;
	}

	void BitwiseOr( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out )
	{
		ParameterValidation( in1, in2, out );

		int threadsPerBlock = 1, blocksPerGrid = 1;
		getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

		bitwiseOr<<<blocksPerGrid, threadsPerBlock>>>( in1.data(), in2.data(), out.data(), out.width() * out.height() );
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
			throw imageException("Failed to launch CUDA kernel");
	}

	ImageCuda BitwiseXor( const ImageCuda & in1, const ImageCuda & in2 )
	{
		ParameterValidation( in1, in2 );

		ImageCuda out( in1.width(), in1.height() );

		BitwiseXor( in1, in2, out );

		return out;
	}

	void BitwiseXor( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out )
	{
		ParameterValidation( in1, in2, out );

		int threadsPerBlock = 1, blocksPerGrid = 1;
		getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

		bitwiseXor<<<blocksPerGrid, threadsPerBlock>>>( in1.data(), in2.data(), out.data(), out.width() * out.height() );
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
			throw imageException("Failed to launch CUDA kernel");
	}

	void Convert( const Bitmap_Image::Image & in, ImageCuda & out )
	{
		Image_Function::ParameterValidation( in );
		ParameterValidation( out );

		if( in.width() != out.width() || in.height() != out.height() )
			throw imageException("Bad input parameters in image function");

		uint32_t rowSizeIn  = in.rowSize();
		uint32_t rowSizeOut = out.width();

		const uint8_t * Y    = in.data();
		const uint8_t * YEnd = Y + in.height() * rowSizeIn;

		uint8_t * cudaY = out.data();

		for( ; Y != YEnd; Y += rowSizeIn, cudaY += rowSizeOut ) {
			cudaError_t error = cudaMemcpy(cudaY, Y, out.width() * sizeof(uint8_t), cudaMemcpyHostToDevice);
			if( error != cudaSuccess )
				throw imageException("Cannot copy a memory to CUDA device");
		}
	}

	void Convert( const ImageCuda & in, Bitmap_Image::Image & out )
	{
		ParameterValidation( in );
		Image_Function::ParameterValidation( out );

		if( in.width() != out.width() || in.height() != out.height() )
			throw imageException("Bad input parameters in image function");

		uint32_t rowSizeIn  = in.width();
		uint32_t rowSizeOut = out.rowSize();

		      uint8_t * Y    = out.data();
		const uint8_t * YEnd = Y + out.height() * rowSizeOut;

		const uint8_t * cudaY = in.data();

		for( ; Y != YEnd; Y += rowSizeOut, cudaY += rowSizeIn ) {
			cudaError_t error = cudaMemcpy(Y, cudaY, in.width() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			if( error != cudaSuccess )
				throw imageException("Cannot copy a memory from CUDA device");
		}
	}

	ImageCuda Invert( const ImageCuda & in )
	{
		ParameterValidation( in );

		ImageCuda out( in.width(), in.height() );

		Invert( in, out );

		return out;
	}

	void Invert( const ImageCuda & in, ImageCuda & out )
	{
		ParameterValidation( in, out );

		int threadsPerBlock = 1, blocksPerGrid = 1;
		getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

		invert<<<blocksPerGrid, threadsPerBlock>>>( in.data(), out.data(), out.width() * out.height() );
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
			throw imageException("Failed to launch CUDA kernel");
	}
};
