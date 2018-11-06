// Example application of median filter usage
#include <iostream>
#include "../../src/filtering.h"
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"
#include "../../src/FileOperation/bitmap.h"

int main()
{
    // This example application is made to show how filters work and what which results they produce

    try // <---- do not forget to put your code into try.. catch block!
    {
        // Create an uniform image with value 128
        PenguinV_Image::Image image = Bitmap_Operation::Load( "mercury.bmp" );

         // If the image is empty it means that the image doesn't exist or the file is not readable
        if( image.empty() )
            throw imageException( "Cannot load the image" );

        // Convert to gray-scale image if it's not
        if( image.colorCount() != PenguinV_Image::GRAY_SCALE )
            image = Image_Function::ConvertToGrayScale( image );

        // Create Image object which will contain filtered image
        PenguinV_Image::Image filtered( image.width(), image.height() );

        // Median filtering
        Image_Function::Median( image, filtered, 3 );
        Bitmap_Operation::Save( "median.bmp", filtered );

        // Prewitt filtering
        Image_Function::Prewitt( image, filtered );
        Bitmap_Operation::Save( "prewitt.bmp", filtered );

        // Sobel filtering
        Image_Function::Sobel( image, filtered );
        Bitmap_Operation::Save( "sobel.bmp", filtered );
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
