// Example application of library utilization with images
#include <iostream>
#include "../../src/file/file_image.h"
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"

int main( int argc, char * argv[] )
{
    std::vector<std::string> filePaths;

    if ( argc > 1 ) {
        for ( int i = 1; i < argc; ++i )
            filePaths.push_back( argv[i] );
    }
    else {
        filePaths.push_back( "lena.bmp" );
    }

    try // <---- do not forget to put your code into try.. catch block!
    {
        // Load an image from storage
        for ( size_t i = 0; i < filePaths.size(); ++i ) {
            const std::string & path = filePaths[i];

            penguinV::Image image = File_Operation::Load( path );

            // If the image is empty it means that the image doesn't exist or the file is not readable
            if ( image.empty() ) {
                std::cerr << std::string( "Cannot load " ) + filePaths[i] << std::endl;
                continue;
            }

            // Convert to gray-scale image if it's not
            if ( image.colorCount() != penguinV::GRAY_SCALE )
                image = Image_Function::ConvertToGrayScale( image );

            // Threshold image with calculated optimal threshold
            const std::vector<uint32_t> histogram = Image_Function::Histogram( image );
            const uint8_t thresholdValue = Image_Function::GetThreshold( histogram );
            image = Image_Function::Threshold( image, thresholdValue );

            // Save result
            const std::string ext = path.substr( path.length() - 3 );
            File_Operation::Save( "result." + ext, image );
        }
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
