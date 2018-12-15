// Example application of blob detection utilization
#include <iostream>
#include "../../../src/edge_detection.h"
#include "../../../src/image_buffer.h"
#include "../../../src/image_function.h"
#include "../../../src/file/bmp_image.h"
#include "../../../src/ui/win/win_ui.h"

int main()
{
    try // <---- do not forget to put your code into try.. catch block!
    {
        const PenguinV_Image::Image original = Bitmap_Operation::Load( "mercury.bmp" );
        Bitmap_Operation::Save( "loaded.bmp", original );

        if ( original.empty() ) // if the image is empty it means that the image doesn't exist or the file is not readable
            throw imageException( "Cannot load the image" );

        PenguinV_Image::Image image( original );
        if ( image.colorCount() != PenguinV_Image::GRAY_SCALE ) // convert to gray-scale image if it's not
            image = Image_Function::ConvertToGrayScale( image );

        EdgeParameter edgeParameter;
        edgeParameter.minimumContrast = 64u; // in intensity grades
        // We know that image would give a lot of false detected points due to single pixel intensity variations
        // In such case we set a range of pixels to verify that found edge point is really an edge
        edgeParameter.contrastCheckLeftSideOffset  = 3u; // in pixels
        edgeParameter.contrastCheckRightSideOffset = 3u; // in pixels

        EdgeDetection edgeDetection;
        edgeDetection.find( image, edgeParameter );

        const std::vector<Point2d> & negativeEdge = edgeDetection.negativeEdge();
        const std::vector<Point2d> & positiveEdge = edgeDetection.positiveEdge();

        UiWindowWin window( original, "Edge detection" );

        const PaintColor positiveColor( 20, 255, 20 ); // green
        const PaintColor negativeColor( 255, 20, 20 ); // red

        for ( std::vector<Point2d>::const_iterator point = positiveEdge.cbegin(); point != positiveEdge.cend(); ++point )
            window.drawPoint( *point, positiveColor );

        for ( std::vector<Point2d>::const_iterator point = negativeEdge.cbegin(); point != negativeEdge.cend(); ++point )
            window.drawPoint( *point, negativeColor );

        window.show();
    }
    catch ( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Closing the application..." << std::endl;
        return 1;
    }
    catch ( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Closing the application..." << std::endl;
        return 2;
    }

    return 0;
}
