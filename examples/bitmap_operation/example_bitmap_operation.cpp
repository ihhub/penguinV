// Example application of library utilization for bitmaps
#include <iostream>
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"
#include "../../src/file/bmp_image.h"

int main( int argc, char * argv[] )
{
    std::string filePath = "lena.bmp"; // default image path
    if ( argc > 1 ) // Check input data
        filePath = argv[1];

    try // <---- do not forget to put your code into try.. catch block!
    {
        // Load an image from storage
        PenguinV_Image::Image image = Bitmap_Operation::Load( filePath );

        // If the image is empty it means that the image doesn't exist or the file is not readable
        if ( image.empty() )
            throw imageException( std::string("Cannot load ") + filePath );

        // Convert to gray-scale image if it's not
        if ( image.colorCount() != PenguinV_Image::GRAY_SCALE )
            image = Image_Function::ConvertToGrayScale( image );

        // Threshold image with calculated optimal threshold
        const std::vector< uint32_t > histogram = Image_Function::Histogram( image );
        const uint8_t thresholdValue = Image_Function::GetThreshold( histogram );
        image = Image_Function::Threshold( image, thresholdValue );

        // Save result
        Bitmap_Operation::Save( "result.bmp", image );
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
