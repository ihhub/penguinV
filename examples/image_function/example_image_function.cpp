// Example application of library utilization
#include <iostream>
#include "../../src/image_buffer.h"
#include "../../src/image_function.h"

void example1( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut );
void example2( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut );
void example3( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut );
void example4( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut );

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

        PenguinV_Image::Image imageIn1( 1024, 1024 );
        PenguinV_Image::Image imageIn2( 2048, 2048 );
        PenguinV_Image::Image imageOut( 512, 512 );

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
    catch( imageException & ex ) { // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from bad things
        return 1;
    }
    catch( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Closing the application..." << std::endl;
        return 2;
    }
    catch( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Closing the application..." << std::endl;
        return 3    ;
    }

    std::cout << "Everything went fine." << std::endl;

    return 0;
}


void example1( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut )
{
    // We allocate another images with size 128 x 128
    PenguinV_Image::Image croppedIn1( 128, 128 );
    PenguinV_Image::Image croppedIn2( 128, 128 );
    PenguinV_Image::Image croppedOut( 128, 128 );

    // Copy information from input images to cropped images

    // It is not a good style of programming but it is easy to explain all parameters!
    Image_Function::Copy( imageIn1,   // first input image
                          10,         // X coordinate where we get information from first input image
                          10,         // Y coordinate where we get information from first input image
                          croppedIn1, // first cropped image
                          0,          // X coordinate where we put information to first cropped image
                          0,          // Y coordinate where we put information to first cropped image
                          128,        // width of copying area
                          128         // height of copying area
    );

    Image_Function::Copy( imageIn2,   // second input image
                          650,        // X coordinate where we get information from second input image
                          768,        // Y coordinate where we get information from second input image
                          croppedIn1, // second cropped image
                          0,          // X coordinate where we put information to second cropped image
                          0,          // Y coordinate where we put information to second cropped image
                          128,        // width of copying area
                          128         // height of copying area
    );

    // Do Bitwise OR
    Image_Function::BitwiseOr( croppedIn1, croppedIn2, croppedOut );

    // Copy the result into output image
    Image_Function::Copy( croppedOut, // cropped output image
                          0,          // X coordinate where we get information from cropped output image
                          0,          // Y coordinate where we get information from cropped output image
                          imageOut,   // output image
                          0,          // X coordinate where we put information to output image
                          0,          // Y coordinate where we put information to output image
                          128,        // width of copying area
                          128         // height of copying area
    );
}

void example2( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut )
{
    // We allocate another images with size 128 x 128
    PenguinV_Image::Image croppedIn1( 128, 128 );
    PenguinV_Image::Image croppedIn2( 128, 128 );

    // Copy information from input images to cropped images

    Image_Function::Copy( imageIn1,   // first input image
                          10,         // X coordinate where we get information from first input image
                          10,         // Y coordinate where we get information from first input image
                          croppedIn1, // first cropped image
                          0,          // X coordinate where we put information to first cropped image
                          0,          // Y coordinate where we put information to first cropped image
                          128,        // width of copying area
                          128         // height of copying area
    );

    Image_Function::Copy( imageIn2,   // second input image
                          650,        // X coordinate where we get information from second input image
                          768,        // Y coordinate where we get information from second input image
                          croppedIn1, // second cropped image
                          0,          // X coordinate where we put information to second cropped image
                          0,          // Y coordinate where we put information to second cropped image
                          128,        // width of copying area
                          128         // height of copying area
    );

    // Do Bitwise OR and store result in cropped output image (move operator)
    PenguinV_Image::Image croppedOut = Image_Function::BitwiseOr( croppedIn1, croppedIn2 );

    // Copy the result into output image
    Image_Function::Copy( croppedOut, // cropped output image
                          0,          // X coordinate where we get information from cropped output image
                          0,          // Y coordinate where we get information from cropped output image
                          imageOut,   // output image
                          0,          // X coordinate where we put information to output image
                          0,          // Y coordinate where we put information to output image
                          128,        // width of copying area
                          128         // height of copying area
    );
}

void example3( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut )
{
    // Do Bitwise OR and store result in cropped output image (move operator)
    PenguinV_Image::Image croppedOut = Image_Function::BitwiseOr( imageIn1, // first input image
                                                                10,       // X coordinate of first input image area
                                                                10,       // Y coordinate of first input image area
                                                                imageIn2, // second input image
                                                                650,      // X coordinate of second input image area
                                                                768,      // Y coordinate of second input image area
                                                                128,      // width of the area
                                                                128       // height of the area
    );

    // Copy the result into output image
    Image_Function::Copy( croppedOut, // cropped output image
                          0,          // X coordinate where we get information from cropped output image
                          0,          // Y coordinate where we get information from cropped output image
                          imageOut,   // output image
                          0,          // X coordinate where we put information to output image
                          0,          // Y coordinate where we put information to output image
                          128,        // width of copying area
                          128         // height of copying area
    );
}

void example4( const PenguinV_Image::Image & imageIn1, const PenguinV_Image::Image & imageIn2, PenguinV_Image::Image & imageOut )
{
    Image_Function::BitwiseOr( imageIn1, // first input image
                               10,       // X coordinate of first input image area
                               10,       // Y coordinate of first input image area
                               imageIn2, // second input image
                               650,      // X coordinate of second input image area
                               768,      // Y coordinate of second input image area
                               imageOut, // output image
                               0,        // X coordinate of output image area
                               0,        // Y coordinate of output image area
                               128,      // width of the area
                               128       // height of the area
    );
    // And that's all
}
