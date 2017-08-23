#include <iostream>
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include "../../../Library/blob_detection.h"
#include "../../../Library/image_buffer.h"
#include "../../../Library/image_function.h"
#include "../../../Library/FileOperation/bitmap.h"

void showImage( QLabel & window, QImage & image );

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
        Bitmap_Image::Image original = Bitmap_Operation::Load( fileName.toUtf8().constData() );

        if( original.colorCount() != Bitmap_Image::RGB || original.empty() ) {
            std::cout << "Looks like no image or it is not a color image" << std::endl;
            return 0;
        }

        // Convert image into QImage format
        QImage originalQt( original.data(), original.width(), original.height(), original.rowSize(), QImage::Format_RGB888 );

        // Display image in separate window
        QLabel window1;
        showImage( window1, originalQt );

        // Because our logo is green-white so we extract red channel to make green areas be as black on gray-scale image
        Bitmap_Image::Image red = Image_Function::ExtractChannel( original, 0 );

        // Convert image into QImage format
        QImage redQt( red.data(), red.width(), red.height(), red.rowSize(), QImage::Format_Grayscale8 );

        // Display image in separate window
        QLabel window2;
        showImage( window2, redQt );

        // We theshold image for proper blob detection
        // Remark: actually we can pass the threshold to blob detection function so we can skip this step
        // But we made this just to show how it works
        Bitmap_Image::Image thresholded = Image_Function::Threshold( red, Image_Function::GetThreshold( Image_Function::Histogram( red ) ) );

        // Convert image into QImage format
        QImage thresholdedQt( thresholded.data(), thresholded.width(), thresholded.height(), thresholded.rowSize(), QImage::Format_Grayscale8 );

        // Display image in separate window
        QLabel window3;
        showImage( window3, thresholdedQt );

        // Perform blob detection
        Blob_Detection::BlobDetection detection;
        detection.find( thresholded );

        // Create an ouput image
        Bitmap_Image::Image output( thresholded.width(), thresholded.height() );
        output.fill( 0 );

        // Sort blobs by size and do NOT draw an edge of biggest blob what is actually white backgroud
        detection.sort( Blob_Detection::BlobDetection::CRITERION_SIZE );

        for( std::vector <Blob_Detection::BlobInfo>::const_iterator blob = detection().begin() + 1; blob != detection().end(); ++blob )
            Image_Function::SetPixel( output, blob->edgeX(), blob->edgeY(), 255 );

        // Convert image into QImage format
        QImage outputQt( output.data(), output.width(), output.height(), output.rowSize(), QImage::Format_Grayscale8 );

        // Display image in separate window
        QLabel window4;
        showImage( window4, outputQt );

        std::cout << "Everything went fine." << std::endl;

        return app.exec();
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

    return 0;
}

void showImage( QLabel & window, QImage & image )
{
    window.setPixmap( QPixmap::fromImage( image ) );
    window.show();
}
