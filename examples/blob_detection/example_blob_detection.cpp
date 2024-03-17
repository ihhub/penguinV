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

// Example application of blob detection utilization
#include "../../src/blob_detection.h"
#include "../../src/file/bmp_image.h"
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"
#include <iostream>

int main( int argc, char * argv[] )
{
    std::string filePath = "mercury.bmp"; // default image path
    if ( argc > 1 ) { // Check input data
        filePath = argv[1];
    }

    try // <---- do not forget to put your code into try.. catch block!
    {
        // Load image from storage
        penguinV::Image image = Bitmap_Operation::Load( filePath );

        // If the image is empty it means that the image doesn't exist or the file is not readable
        if ( image.empty() ) {
            throw penguinVException( std::string( "Cannot load " ) + filePath );
        }

        // Convert to gray-scale image if it's not
        if ( image.colorCount() != penguinV::GRAY_SCALE ) {
            image = Image_Function::ConvertToGrayScale( image );
        }

        // Threshold image with calculated optimal threshold
        const std::vector<uint32_t> histogram = Image_Function::Histogram( image );
        const uint8_t thresholdValue = Image_Function::GetThreshold( histogram );
        image = Image_Function::Threshold( image, thresholdValue );

        // Search all possible blobs on image
        Blob_Detection::BlobDetection detection;
        detection.find( image );

        if ( !detection().empty() ) {
            // okay, our image contains some blobs
            // extract a biggest one
            const Blob_Detection::BlobInfo & blob = detection.getBestBlob( Blob_Detection::BlobDetection::BlobCriterion::BY_SIZE );

            // clear image and draw contour of found blob
            image.fill( 0 );

            for ( size_t i = 0; i < blob.contourX().size(); ++i ) {
                Image_Function::SetPixel( image, blob.contourX()[i], blob.contourY()[i], 255u );
            }
        }

        // Save result into file
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
