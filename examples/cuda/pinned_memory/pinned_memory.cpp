// Example application of showing of CUDA pinned memory speed advantage
#include <iostream>
#include "../../../src/image_buffer.h"
#include "../../../src/cuda/cuda_memory.cuh"
#include "../../../src/cuda/image_buffer_cuda.cuh"
#include "../../../src/cuda/image_buffer_cuda_pinned.cuh"
#include "../../../src/cuda/cuda_helper.cuh"

void measureTiming( penguinV::Image & in, penguinV::Image & out, const std::string & type, bool hostToDevice )
{
    cudaEvent_t start, stop;

    multiCuda::cudaCheck( cudaEventCreate(&start) );
    multiCuda::cudaCheck( cudaEventCreate(&stop) );

    multiCuda::cudaCheck( cudaEventRecord(start, 0) );

    const uint32_t size = out.rowSize() * out.height();

    multiCuda::cudaCheck( cudaMemcpy(out.data(), in.data(), size, hostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost) );
    multiCuda::cudaCheck( cudaEventRecord(stop, 0) );
    multiCuda::cudaCheck( cudaEventSynchronize(stop) );

    float time = 0;
    multiCuda::cudaCheck( cudaEventElapsedTime(&time, start, stop) );

    std::cout << type << (hostToDevice ? ": Host to Device" : ": Device to Host" ) << " bandwidth (GB/s): " << size * 1e-6 / time << std::endl;
}

int main()
{
    // This example application shows speed difference between normal CPU RAM allocation
    // and using CUDA pinned memory for copy operations from CUDA device to CPU RAM and
    // reverse way

    try // <---- do not forget to put your code into try.. catch block!
    {
        // First we check whether any CUDA device is in system
        if( !multiCuda::isCudaSupported() ) {
            std::cout << "No CUDA devices found in the system" << std::endl;
            return 0;
        }

        multiCuda::CudaDeviceManager::instance().initializeDevices();

        const uint32_t width = 2048;
        const uint32_t height = 2048;

        // We allocate image using normal CPU RAM allocation (no magic inside)
        penguinV::Image in1( width, height );

        // Then we allocate image using CUDA pinned memory...
        penguinV::ImageCudaPinned in2( width, height );

        // Do you see the diffrence in programming syntax? No difference :)
        // And you can use this image in normal image operations as well

        // Now we allocate image on CUDA device
        penguinV::ImageCuda out( width, height );

        // First we will measure speed to copy image from CPU RAM to CUDA device memory
        // To avoid caching on CPU and give an advantage for normal CPU allocation
        // we run functions for CUDA pinned memory image
        measureTiming( in2, out, "Pinned    ", true );
        measureTiming( in1, out, "Non-pinned", true );

        measureTiming( out, in2, "Pinned    ", false );
        measureTiming( out, in1, "Non-pinned", false );
    }
    catch( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << ex.what() << ". Press any button to continue." << std::endl;
        std::cin.ignore();
        return 1;
    }
    catch( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Press any button to continue." << std::endl;
        std::cin.ignore();
        return 2;
    }

    return 0;
}
