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

// Example application of library utilization
#include "../../src/penguinv/penguinv.h"
#include <iostream>

void example1( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut );
void example2( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut );
void example3( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut );
void example4( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut );

int main()
{
    // This example is to show how you can perform bitwise operations on gray-scale images
    //
    // Conditions:
    // We are in the middle of very serious image processing algorithm development
    //
    // We have 2 input images and 1 output image with known sizes:
    // - first input image is 1024 x 1024 pixels
    // - second input image is 2048 x 2048 pixels
    // - output image (where we would like to store results) is 512 x 512 pixel
    //
    // We have to make bitwise OR on 2 input images and store result in output image
    // We know that the area in first input image has coordinates { [10, 10], [138, 138] },
    // second input image - { [650, 768], [778, 896] },
    // and we need to put result into area { [0, 0], [128, 128] } in output image
    //
    // So let's do it!

    try // <---- do not forget to put your code into try.. catch block!
    {
        // create image objects with certain sizes

        penguinV::Image imageIn1( 1024, 1024 );
        penguinV::Image imageIn2( 2048, 2048 );
        penguinV::Image imageOut( 512, 512 );

        // set any data because initially images contain some uninitialized garbage
        // (we assume that this step you do not need to do during real development of image processing algorithms)
        imageIn1.fill( 255 );
        imageIn2.fill( 128 );
        imageOut.fill( 0 );

        // Now we need to make bitwise operation OR
        // First way to do
        example1( imageIn1, imageIn2, imageOut );

        // Second way to do
        example2( imageIn1, imageIn2, imageOut );

        // Third way to do
        example3( imageIn1, imageIn2, imageOut );

        // Forth way to do
        example4( imageIn1, imageIn2, imageOut );
    }
    catch ( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Closing the application..." << std::endl;
        return 1;
    }
    catch ( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Closing the application..." << std::endl;
        return 2;
    }

    std::cout << "Application ended correctly." << std::endl;
    return 0;
}

void example1( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut )
{
    // We allocate another images with size 128 x 128
    penguinV::Image croppedIn1( 128, 128 );
    penguinV::Image croppedIn2( 128, 128 );
    penguinV::Image croppedOut( 128, 128 );

    // Copy information from input images to cropped images

    // It is not a good style of programming but it is easy to explain all parameters!
    penguinV::Copy( imageIn1, // first input image
                    10, // X coordinate where we get information from first input image
                    10, // Y coordinate where we get information from first input image
                    croppedIn1, // first cropped image
                    0, // X coordinate where we put information to first cropped image
                    0, // Y coordinate where we put information to first cropped image
                    128, // width of copying area
                    128 // height of copying area
    );

    penguinV::Copy( imageIn2, // second input image
                    650, // X coordinate where we get information from second input image
                    768, // Y coordinate where we get information from second input image
                    croppedIn2, // second cropped image
                    0, // X coordinate where we put information to second cropped image
                    0, // Y coordinate where we put information to second cropped image
                    128, // width of copying area
                    128 // height of copying area
    );

    // Do Bitwise OR
    penguinV::BitwiseOr( croppedIn1, croppedIn2, croppedOut );

    // Copy the result into output image
    penguinV::Copy( croppedOut, // cropped output image
                    0, // X coordinate where we get information from cropped output image
                    0, // Y coordinate where we get information from cropped output image
                    imageOut, // output image
                    0, // X coordinate where we put information to output image
                    0, // Y coordinate where we put information to output image
                    128, // width of copying area
                    128 // height of copying area
    );
}

void example2( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut )
{
    // We allocate another images with size 128 x 128
    penguinV::Image croppedIn1( 128, 128 );
    penguinV::Image croppedIn2( 128, 128 );

    // Copy information from input images to cropped images

    penguinV::Copy( imageIn1, // first input image
                    10, // X coordinate where we get information from first input image
                    10, // Y coordinate where we get information from first input image
                    croppedIn1, // first cropped image
                    0, // X coordinate where we put information to first cropped image
                    0, // Y coordinate where we put information to first cropped image
                    128, // width of copying area
                    128 // height of copying area
    );

    penguinV::Copy( imageIn2, // second input image
                    650, // X coordinate where we get information from second input image
                    768, // Y coordinate where we get information from second input image
                    croppedIn2, // second cropped image
                    0, // X coordinate where we put information to second cropped image
                    0, // Y coordinate where we put information to second cropped image
                    128, // width of copying area
                    128 // height of copying area
    );

    // Do Bitwise OR and store result in cropped output image (move operator)
    penguinV::Image croppedOut = penguinV::BitwiseOr( croppedIn1, croppedIn2 );

    // Copy the result into output image
    penguinV::Copy( croppedOut, // cropped output image
                    0, // X coordinate where we get information from cropped output image
                    0, // Y coordinate where we get information from cropped output image
                    imageOut, // output image
                    0, // X coordinate where we put information to output image
                    0, // Y coordinate where we put information to output image
                    128, // width of copying area
                    128 // height of copying area
    );
}

void example3( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut )
{
    // Do Bitwise OR and store result in cropped output image (move operator)
    penguinV::Image croppedOut = penguinV::BitwiseOr( imageIn1, // first input image
                                                      10, // X coordinate of first input image area
                                                      10, // Y coordinate of first input image area
                                                      imageIn2, // second input image
                                                      650, // X coordinate of second input image area
                                                      768, // Y coordinate of second input image area
                                                      128, // width of the area
                                                      128 // height of the area
    );

    // Copy the result into output image
    penguinV::Copy( croppedOut, // cropped output image
                    0, // X coordinate where we get information from cropped output image
                    0, // Y coordinate where we get information from cropped output image
                    imageOut, // output image
                    0, // X coordinate where we put information to output image
                    0, // Y coordinate where we put information to output image
                    128, // width of copying area
                    128 // height of copying area
    );
}

void example4( const penguinV::Image & imageIn1, const penguinV::Image & imageIn2, penguinV::Image & imageOut )
{
    penguinV::BitwiseOr( imageIn1, // first input image
                         10, // X coordinate of first input image area
                         10, // Y coordinate of first input image area
                         imageIn2, // second input image
                         650, // X coordinate of second input image area
                         768, // Y coordinate of second input image area
                         imageOut, // output image
                         0, // X coordinate of output image area
                         0, // Y coordinate of output image area
                         128, // width of the area
                         128 // height of the area
    );
    // And that's all
}
