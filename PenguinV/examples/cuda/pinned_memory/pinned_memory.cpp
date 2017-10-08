// Example application of showing of CUDA pinned memory speed advantage
#include <iostream>
#include "../../../Library/image_buffer.h"
#include "../../../Library/cuda/cuda_memory.cuh"
#include "../../../Library/cuda/image_buffer_cuda.cuh"
#include "../../../Library/cuda/image_buffer_cpu.cuh"
#include "../../../Library/cuda/cuda_helper.cuh"

void measureTimingToCuda  ( Bitmap_Image::Image      & in, Bitmap_Image_Cuda::Image & out );
void measureTimingFromCuda( Bitmap_Image_Cuda::Image & in, Bitmap_Image::Image      & out );

int main()
{
    // This example application shows speed difference between normal CPU RAM allocation
    // and using CUDA pinned memory for copy operations from CUDA device to CPU RAM and
    /// reverse way

    try // <---- do not forget to put your code into try.. catch block!
    {
        // First we check whether any CUDA device is in system
        if( !Cuda::isCudaSupported() ) {
            std::cout << "No CUDA devices found in the system" << std::endl;
            return 0;
        }

        const size_t width = 2048;
        const size_t height = 2048;

        // We allocate image using normal CPU RAM allocation (no magic inside)
        Bitmap_Image::Image in1( width, height );

        // Then we allocate image using CUDA pinned memory...
        Bitmap_Image_Cuda_Cpu::Image in2( width, height );

        // Do you see the diffrence in programming syntax? No difference :)
        // And you can use this image in normal image operations as well

        // Now we allocate image on CUDA device
        Bitmap_Image_Cuda::Image out( width, height );

        // First we will measure speed to copy image from CPU RAM to CUDA device memory
        // To avoid caching on CPU and give an advantage for normal CPU allocation
        // we run functions for CUDA pinned memory image
        measureTimingToCuda( in2, out );
        measureTimingToCuda( in1, out );

        measureTimingFromCuda( out, in2 );
        measureTimingFromCuda( out, in1 );
    }
    catch( imageException & ex ) {
        // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from bad things
        return 0;
    }
    catch( ... ) {
        // uh-oh, something terrible happen!
        std::cout << "Something very terrible happen. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from terrible things
        return 0;
    }

    std::cout << "Everything went fine." << std::endl;

    return 0;
}

void measureTimingToCuda( Bitmap_Image::Image & in, Bitmap_Image_Cuda::Image & out )
{
    cudaEvent_t start, stop; 

    Cuda::cudaCheck( cudaEventCreate(&start) );
    Cuda::cudaCheck( cudaEventCreate(&stop) );

    Cuda::cudaCheck( cudaEventRecord(start, 0) );

    const size_t size = out.rowSize() * out.height();

    Cuda::cudaCheck( cudaMemcpy(out.data(), in.data(), size, cudaMemcpyHostToDevice) );
    Cuda::cudaCheck( cudaEventRecord(stop, 0) );
    Cuda::cudaCheck( cudaEventSynchronize(stop) );
    
    float time = 0;
    Cuda::cudaCheck( cudaEventElapsedTime(&time, start, stop) );

    std::cout << "Host to Device bandwidth (GB/s): " << size * 1e-6 / time << std::endl;
}

void measureTimingFromCuda( Bitmap_Image_Cuda::Image & in, Bitmap_Image::Image & out )
{
    cudaEvent_t start, stop; 

    Cuda::cudaCheck( cudaEventCreate(&start) );
    Cuda::cudaCheck( cudaEventCreate(&stop) );

    Cuda::cudaCheck( cudaEventRecord(start, 0) );

    const size_t size = out.rowSize() * out.height();

    Cuda::cudaCheck( cudaMemcpy(out.data(), in.data(), size, cudaMemcpyDeviceToHost) );
    Cuda::cudaCheck( cudaEventRecord(stop, 0) );
    Cuda::cudaCheck( cudaEventSynchronize(stop) );
    
    float time = 0;
    Cuda::cudaCheck( cudaEventElapsedTime(&time, start, stop) );

    std::cout << "Device to Host bandwidth (GB/s): " << size * 1e-6 / time << std::endl;
}