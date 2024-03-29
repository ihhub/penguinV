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

// Example application of edge detection utilization
#include "../../src/edge_detection.h"
#include "../../src/file/bmp_image.h"
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"
#include <iostream>
#if defined( _WIN32 )
#include "../../src/ui/win/win_ui.h"
#else
#include "../../src/ui/x11/x11_ui.h"
#endif

int main( int argc, char * argv[] )
{
    try // <---- do not forget to put your code into try.. catch block!
    {
        std::string filePath = "mercury.bmp"; // default image path
        if ( argc > 1 ) // Check input data
            filePath = argv[1];

        const penguinV::Image original = Bitmap_Operation::Load( filePath );

        if ( original.empty() ) // if the image is empty it means that the image doesn't exist or the file is not readable
            throw penguinVException( std::string( "Cannot load " ) + filePath );

        penguinV::Image image( original );
        if ( image.colorCount() != penguinV::GRAY_SCALE ) // convert to gray-scale image if it's not
            image = Image_Function::ConvertToGrayScale( image );

        EdgeParameter edgeParameter;
        edgeParameter.minimumContrast = 64u; // in intensity grades
        // We know that image would give a lot of false detected points due to single pixel intensity variations
        // In such case we set a range of pixels to verify that found edge point is really an edge
        edgeParameter.contrastCheckLeftSideOffset = 3u; // in pixels
        edgeParameter.contrastCheckRightSideOffset = 3u; // in pixels

        EdgeDetection edgeDetection;
        edgeDetection.find( image, edgeParameter );

        const std::vector<Point2d> & negativeEdge = edgeDetection.negativeEdge();
        const std::vector<Point2d> & positiveEdge = edgeDetection.positiveEdge();

#if defined( _WIN32 )
        UiWindowWin window( original, "Edge detection" );
#else
        UiWindowX11 window( original, "Edge detection" );
#endif

        const PaintColor positiveColor( 20, 255, 20 ); // green
        const PaintColor negativeColor( 255, 20, 20 ); // red

        for ( std::vector<Point2d>::const_iterator point = positiveEdge.cbegin(); point != positiveEdge.cend(); ++point )
            window.drawPoint( *point, positiveColor );

        for ( std::vector<Point2d>::const_iterator point = negativeEdge.cbegin(); point != negativeEdge.cend(); ++point )
            window.drawPoint( *point, negativeColor );

        window.show();
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

    return 0;
}
