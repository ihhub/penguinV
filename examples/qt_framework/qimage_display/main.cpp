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

#include "../../../src/blob_detection.h"
#include "../../../src/file/bmp_image.h"
#include "../../../src/image_buffer.h"
#include "../../../src/image_function.h"
#include "../../../src/ui/qt/qt_ui.h"
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <iostream>

int main( int argc, char * argv[] )
{
    try {
        // First of all we create QT application
        QApplication app( argc, argv );

        // Then we retrieve a path for bitmap file
        const QString & fileName = QFileDialog::getOpenFileName( NULL, QObject::tr( "Open Bitmap image" ), "", QObject::tr( "Bitmap (*.bmp);;All Files (*)" ) );

        // Load a color image from storage
        penguinV::Image original = Bitmap_Operation::Load( fileName.toUtf8().constData() );

        if ( original.colorCount() != penguinV::RGB || original.empty() ) {
            std::cout << "Looks like no image or it is not a color image" << std::endl;
            return 0;
        }

        // Display image in separate window
        UiWindowQt window1( original );
        window1.show();

        // Because our logo is green-white so we extract red channel to make green areas be as black on gray-scale image
        penguinV::Image red = Image_Function::ExtractChannel( original, 0 );

        // Display image in separate window
        UiWindowQt window2( red );
        window2.show();

        // We theshold image for proper blob detection
        // Remark: actually we can pass the threshold to blob detection function so we can skip this step
        // But we made this just to show how it works
        penguinV::Image thresholded = Image_Function::Threshold( red, Image_Function::GetThreshold( Image_Function::Histogram( red ) ) );

        // Display image in separate window
        UiWindowQt window3( thresholded );
        window3.show();

        // Perform blob detection
        Blob_Detection::BlobDetection detection;
        detection.find( thresholded );

        // Create an ouput image
        penguinV::Image output( thresholded.width(), thresholded.height() );
        output.fill( 0 );

        // Sort blobs by size and do NOT draw an edge of biggest blob what is actually white backgroud
        detection.sort( Blob_Detection::BlobDetection::BY_SIZE );

        for ( std::vector<Blob_Detection::BlobInfo>::const_iterator blob = detection().begin() + 1; blob != detection().end(); ++blob )
            Image_Function::SetPixel( output, blob->edgeX(), blob->edgeY(), 255 );

        // Display image in separate window
        UiWindowQt window4( output );
        window4.show();

        return app.exec();
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
}
