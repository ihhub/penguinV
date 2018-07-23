// Example application of library OpenCL module utilization
#include <iostream>
#include "../../../src/image_buffer.h"
#include "../../../src/image_function.h"
#include "../../../src/opencl/image_buffer_opencl.h"
#include "../../../src/opencl/image_function_opencl.h"
#include "../../../src/FileOperation/bitmap.h"
#include "../../../src/thirdparty/multicl/src/opencl_device.h"
#include "../../../src/thirdparty/multicl/src/opencl_helper.h"

void cpuBased();
void gpuBased();

int main()
{
    // This example application is made to show how to use bitmap file operations
    // and comparison between CPU based code and GPU based code
    // as well as basic image processing operations.
    // Conditions:
    // - "Houston, we received the image of Mercury!"
    // We have an image of Mercury (24-bit color image). We want to load it,
    // convert to gray-scale, extract the planet on image by applying thresholding and
    // save image on storage

    try // <---- do not forget to put your code into try.. catch block!
    {
        // First thing we should check whether the system contains GPU device
        if( !multiCL::isOpenCLSupported() ) {
            std::cout << "GPU device is not found in current system." << std::endl;
            return 0;
        }

        // CPU code
        cpuBased();
        // GPU code
        gpuBased();
    }
    catch( const std::exception & ex ) {
        // uh-oh, something went wrong!
        std::cout << "Exception '" << ex.what() << "' raised. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from bad things
        return 1;
    }
    catch( ... ) {
        // uh-oh, something terrible happen!
        std::cout << "Something very terrible happen. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from terrible things
        return 2;
    }

    std::cout << "Everything went fine." << std::endl;

    return 0;
}

void cpuBased()
{
    // Load an image from storage
    // Please take note that the image must be in the same folder as this application or project (for Visual Studio)
    // Otherwise you can change the path where the image stored
    PenguinV_Image::Image image = Bitmap_Operation::Load( "mercury.bmp" );

    // If the image is empty it means that the image doesn't exist or the file is not readable
    if( image.empty() )
        throw imageException( "Cannot load the image" );

    // Convert to gray-scale image if it's not
    if( image.colorCount() != PenguinV_Image::GRAY_SCALE )
        image = Image_Function::ConvertToGrayScale( image );

    // Threshold image with calculated optimal threshold
    image = Image_Function::Threshold( image, Image_Function::GetThreshold( Image_Function::Histogram( image ) ) );

    // Save result
    Bitmap_Operation::Save( "result_CPU.bmp", image );
}

void gpuBased()
{
    multiCL::OpenCLDeviceManager & deviceManager = multiCL::OpenCLDeviceManager::instance();
    deviceManager.initializeDevices();
    for ( uint32_t deviceId = 0; deviceId < deviceManager.deviceCount(); ++deviceId) {
        deviceManager.setActiveDevice( deviceId );
        // It is recommended to use preallocated buffers for GPU memory usage
        // So we preallocate 32 MB of GPU memory for our usage
        multiCL::MemoryManager::memory().reserve( 32 * 1024 * 1024 );

        // Load an image from storage
        // Please take note that the image must be in the same folder as this application or project (for Visual Studio)
        // Otherwise you can change the path where the image stored
        PenguinV_Image::Image image = Bitmap_Operation::Load( "mercury.bmp" );

        // If the image is empty it means that the image doesn't exist or the file is not readable
        if( image.empty() )
            throw imageException( "Cannot load the image" );

        // We try to mutate the image to make alignment equal to 1
        image.mutate( image.width(), image.height(), image.colorCount(), 1u );

        // Copy image from GPU space to GPU space
        Bitmap_Image_OpenCL::Image imageGPU = Image_Function_OpenCL::ConvertToOpenCL( image );

        // Convert to gray-scale image if it's not
        if( imageGPU.colorCount() != PenguinV_Image::GRAY_SCALE )
            imageGPU = Image_Function_OpenCL::ConvertToGrayScale( imageGPU );

        // Threshold image with calculated optimal threshold
        imageGPU = Image_Function_OpenCL::Threshold( imageGPU, Image_Function_OpenCL::GetThreshold( Image_Function_OpenCL::Histogram( imageGPU ) ) );

        // Save result
        Bitmap_Operation::Save( "result_" + deviceManager.device().name() + ".bmp", Image_Function_OpenCL::ConvertFromOpenCL( imageGPU ) );
    }
}
