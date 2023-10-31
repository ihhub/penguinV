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

// Example application of median filter usage
#include "../../src/file/bmp_image.h"
#include "../../src/filtering.h"
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"
#include <config.h>
#include <iostream>

int main( int argc, char * argv[] )
{
    // This example application is made to show how filters work and what which results they produce

    try // <---- do not forget to put your code into try.. catch block!
    {
        std::string filePath = DataPath + "lena.bmp"; // default image path
        if ( argc > 1 ) // Check input data
            filePath = argv[1];

        // Load an image
        penguinV::Image image = Bitmap_Operation::Load( filePath );

        // If the image is empty it means that the image doesn't exist or the file is not readable
        if ( image.empty() )
            throw penguinVException( std::string( "Cannot load " ) + filePath );

        // Convert to gray-scale image if it's not
        if ( image.colorCount() != penguinV::GRAY_SCALE )
            image = Image_Function::ConvertToGrayScale( image );

        // Create Image object which will contain filtered image
        penguinV::Image filtered( image.width(), image.height() );

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
