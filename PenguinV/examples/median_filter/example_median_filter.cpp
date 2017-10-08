// Example application of median filter usage
#include <iostream>
#include "../../Library/filtering.h"
#include "../../Library/image_buffer.h"
#include "../../Library/image_function.h"
#include "../../Library/FileOperation/bitmap.h"

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
        Bitmap_Image::Image image(1024, 1024);
        image.fill(128);

        // Generate some salt-and-pepper noise
        for( size_t i = 0; i < 1000; ++i )
        {
            size_t x = static_cast<size_t>(rand()) % image.width();
            size_t y = static_cast<size_t>(rand()) % image.height();

            Image_Function::SetPixel(image, x, y, ( ( i % 2 ) == 0 ) ? 0u : 255u );
        }

        // Save original image into bitmap
        Bitmap_Operation::Save( "original.bmp", image );

        // Create Image object which will contain filtered image
        Bitmap_Image::Image filtered( image.width(), image.height() );

        // Filter it!
        Image_Function::Filtering::Median( image, filtered, 3 );

        // Save filtered image into bitmap
        Bitmap_Operation::Save( "filtered.bmp", filtered );
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
