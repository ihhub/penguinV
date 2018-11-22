#include <iostream>
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include "../../../src/blob_detection.h"
#include "../../../src/image_buffer.h"
#include "../../../src/image_function.h"
#include "../../../src/FileOperation/bitmap.h"
#include "../../../src/ui/qt/qt_ui.h"

int main( int argc, char *argv[] )
{
    try
    {
        // First of all we create QT application
        QApplication app( argc, argv );

        // Then we retrieve a path for bitmap file
        const QString & fileName = QFileDialog::getOpenFileName( NULL,
                                                                 QObject::tr("Open Bitmap image"), "",
                                                                 QObject::tr("Bitmap (*.bmp);;All Files (*)") );

        // Load a color image from storage
        PenguinV_Image::Image original = Bitmap_Operation::Load( fileName.toUtf8().constData() );

        if( original.colorCount() != PenguinV_Image::RGB || original.empty() ) {
            std::cout << "Looks like no image or it is not a color image" << std::endl;
            return 0;
        }

        // Display image in separate window
        UiWindowQt window1( original );
        window1.show();

        // Because our logo is green-white so we extract red channel to make green areas be as black on gray-scale image
        PenguinV_Image::Image red = Image_Function::ExtractChannel( original, 0 );

        // Convert image into QImage format
        UiWindowQt window2( red );
        window2.show();

        // We theshold image for proper blob detection
        // Remark: actually we can pass the threshold to blob detection function so we can skip this step
        // But we made this just to show how it works
        PenguinV_Image::Image thresholded = Image_Function::Threshold( red, Image_Function::GetThreshold( Image_Function::Histogram( red ) ) );

        // Convert image into QImage format
        UiWindowQt window3( thresholded );
        window3.show();

        // Perform blob detection
        Blob_Detection::BlobDetection detection;
        detection.find( thresholded );

        // Create an ouput image
        PenguinV_Image::Image output( thresholded.width(), thresholded.height() );
        output.fill( 0 );

        // Sort blobs by size and do NOT draw an edge of biggest blob what is actually white backgroud
        detection.sort( Blob_Detection::BlobDetection::CRITERION_SIZE );

        for( std::vector <Blob_Detection::BlobInfo>::const_iterator blob = detection().begin() + 1; blob != detection().end(); ++blob )
            Image_Function::SetPixel( output, blob->edgeX(), blob->edgeY(), 255 );

        // Convert image into QImage format
        UiWindowQt window4( output );
        window4.show();

        return app.exec();
    }
    catch( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Closing the application..." << std::endl;
        return 1;
    }
    catch( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Closing the application..." << std::endl;
        return 2;
    }
}
