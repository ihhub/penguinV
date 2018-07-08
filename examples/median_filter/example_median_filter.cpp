// Example application of median filter usage
#include <iostream>
#include "../../src/filtering.h"
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"
#include "../../src/FileOperation/bitmap.h"

void example1();
void example2();

int main()
{
    // This example application is made to show how to work with median filter class
    // Conditions:
    // We will generate an image with salt-and-pepper noise
    // Filter the image and see the results

    try // <---- do not forget to put your code into try.. catch block!
    {
        // Create an uniform image with value 128
        PenguinV_Image::Image image(1024, 1024);
        image.fill(128);

        // Generate some salt-and-pepper noise
        for( uint32_t i = 0; i < 1000; ++i )
        {
            uint32_t x = static_cast<uint32_t>(rand()) % image.width();
            uint32_t y = static_cast<uint32_t>(rand()) % image.height();

            Image_Function::SetPixel(image, x, y, ( ( i % 2 ) == 0 ) ? 0u : 255u );
        }

        // Save original image into bitmap
        Bitmap_Operation::Save( "original.bmp", image );

        // Create Image object which will contain filtered image
        PenguinV_Image::Image filtered( image.width(), image.height() );

        // Filter it!
        Image_Function::Median( image, filtered, 3 );

        // Save filtered image into bitmap
        Bitmap_Operation::Save( "filtered.bmp", filtered );
    }
    catch( const imageException & ex ) {
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
