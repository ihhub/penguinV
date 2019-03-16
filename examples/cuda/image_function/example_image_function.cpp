// Example application of library CUDA module utilization
#include <iostream>
#include "../../../src/image_buffer.h"
#include "../../../src/image_function.h"
#include "../../../src/file/bmp_image.h"
#include "../../../src/cuda/cuda_device.cuh"
#include "../../../src/cuda/cuda_helper.cuh"
#include "../../../src/cuda/image_buffer_cuda.cuh"
#include "../../../src/cuda/image_function_cuda.cuh"

void cpuCode( const std::string & filePath );
void gpuCode( const std::string & filePath );

int main( int argc, char * argv[] )
{
    try // <---- do not forget to put your code into try.. catch block!
    {
        // First thing we should check whether the system contains CUDA device
        if( !multiCuda::isCudaSupported() ) {
            std::cout << "CUDA device is not found in current system." << std::endl;
            return 0;
        }

        std::string filePath = "mercury.bmp"; // default image path
        if ( argc > 1 ) // Check input data
            filePath = argv[1];

        // CPU code
        cpuCode( filePath );
        // GPU code
        gpuCode( filePath );
    }
    catch( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Closing the application..." << std::endl;
        return 1;
    }
    catch( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Closing the application..." << std::endl;
        return 2;
    }

    std::cout << "Application ended correctly." << std::endl;
    return 0;
}

void cpuCode( const std::string & filePath )
{
    // Load an image from storage
    // Please take note that the image must be in the same folder as this application or project (for Visual Studio)
    // Otherwise you can change the path where the image stored
    PenguinV_Image::Image image = Bitmap_Operation::Load( filePath );

    // If the image is empty it means that the image doesn't exist or the file is not readable
    if( image.empty() )
        throw imageException( "Cannot load the image" );

    // Convert to gray-scale image if it's not
    if( image.colorCount() != PenguinV_Image::GRAY_SCALE )
        image = Image_Function::ConvertToGrayScale( image );

    // Threshold image with calculated optimal threshold
    image = Image_Function::Threshold( image, Image_Function::GetThreshold( Image_Function::Histogram( image ) ) );

    // Save result
    Bitmap_Operation::Save( "result1.bmp", image );
}

void gpuCode( const std::string & filePath )
{
    // Load an image from storage
    PenguinV_Image::Image image = Bitmap_Operation::Load( filePath );

    // If the image is empty it means that the image doesn't exist or the file is not readable
    if( image.empty() )
        throw imageException( std::string("Cannot load ") + filePath );

    multiCuda::CudaDeviceManager & deviceManager = multiCuda::CudaDeviceManager::instance();
    deviceManager.initializeDevices();
    for ( int deviceId = 0; deviceId < deviceManager.deviceCount(); ++deviceId ) {
        deviceManager.setActiveDevice( deviceId );
        // It is recommended to use preallocated buffers for GPU memory usage
        // So we preallocate 32 MB of GPU memory for our usage
        multiCuda::MemoryManager::memory().reserve( 32 * 1024 * 1024 );

        // Load an image from storage
        PenguinV_Image::Image imageGPU = Image_Function_Cuda::ConvertToCuda( image );

        // Convert to gray-scale image if it's not
        if ( imageGPU.colorCount() != PenguinV_Image::GRAY_SCALE )
            imageGPU = Image_Function_Cuda::ConvertToGrayScale( imageGPU );

        // Threshold image with calculated optimal threshold
        image = Image_Function_Cuda::Threshold( imageGPU, Image_Function_Cuda::GetThreshold( Image_Function_Cuda::Histogram( imageGPU ) ) );

        // Save result
        Bitmap_Operation::Save( "result_" + deviceManager.device().name() + ".bmp", Image_Function_Cuda::ConvertFromCuda( imageGPU ) );
    }
}
