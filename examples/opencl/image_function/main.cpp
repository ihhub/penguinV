/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

// Example application of library OpenCL module utilization
#include "../../../src/file/bmp_image.h"
#include "../../../src/image_buffer.h"
#include "../../../src/image_function.h"
#include "../../../src/opencl/image_buffer_opencl.h"
#include "../../../src/opencl/image_function_opencl.h"
#include "../../../src/opencl/opencl_device.h"
#include "../../../src/opencl/opencl_helper.h"
#include <iostream>

void cpuCode( const std::string & filePath );
void gpuCode( const std::string & filePath );

int main( int argc, char * argv[] )
{
    try // <---- do not forget to put your code into try.. catch block!
    {
        // First thing we should check whether the system contains GPU device
        if ( !multiCL::isOpenCLSupported() ) {
            std::cout << "GPU device is not found in current system." << std::endl;
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
    catch ( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << ex.what() << ". Press any button to continue." << std::endl;
        std::cin.ignore();
        return 1;
    }
    catch ( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Press any button to continue." << std::endl;
        std::cin.ignore();
        return 2;
    }

    std::cout << "Application ended correctly." << std::endl;
    return 0;
}

void cpuCode( const std::string & filePath )
{
    // Load an image from storage
    penguinV::Image image = Bitmap_Operation::Load( filePath );

    // If the image is empty it means that the image doesn't exist or the file is not readable
    if ( image.empty() )
        throw penguinVException( std::string( "Cannot load " ) + filePath );

    // Convert to gray-scale image if it's not
    if ( image.colorCount() != penguinV::GRAY_SCALE )
        image = Image_Function::ConvertToGrayScale( image );

    // Threshold image with calculated optimal threshold
    image = Image_Function::Threshold( image, Image_Function::GetThreshold( Image_Function::Histogram( image ) ) );

    // Save result
    Bitmap_Operation::Save( "result_CPU.bmp", image );
}

void gpuCode( const std::string & filePath )
{
    // Load an image from storage
    penguinV::Image image = Bitmap_Operation::Load( filePath );

    // If the image is empty it means that the image doesn't exist or the file is not readable
    if ( image.empty() )
        throw penguinVException( std::string( "Cannot load " ) + filePath );

    multiCL::OpenCLDeviceManager & deviceManager = multiCL::OpenCLDeviceManager::instance();
    deviceManager.initializeDevices();
    for ( uint32_t deviceId = 0; deviceId < deviceManager.deviceCount(); ++deviceId ) {
        deviceManager.setActiveDevice( deviceId );
        // It is recommended to use preallocated buffers for GPU memory usage
        // So we preallocate 32 MB of GPU memory for our usage
        multiCL::MemoryManager::memory().reserve( 32 * 1024 * 1024 );

        // Copy image from GPU space to GPU space
        penguinV::Image imageGPU = Image_Function_OpenCL::ConvertToOpenCL( image );

        // Convert to gray-scale image if it's not
        if ( imageGPU.colorCount() != penguinV::GRAY_SCALE )
            imageGPU = Image_Function_OpenCL::ConvertToGrayScale( imageGPU );

        // Threshold image with calculated optimal threshold
        imageGPU = Image_Function_OpenCL::Threshold( imageGPU, Image_Function_OpenCL::GetThreshold( Image_Function_OpenCL::Histogram( imageGPU ) ) );

        // Save result
        Bitmap_Operation::Save( "result_" + deviceManager.device().name() + ".bmp", Image_Function_OpenCL::ConvertFromOpenCL( imageGPU ) );
    }
}
