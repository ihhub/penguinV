// Example application of library utilization for bitmaps
#include <iostream>
#include "../../Library/image_buffer.h"
#include "../../Library/image_function.h"
#include "../../Library/FileOperation/bitmap.h"

void method1();
void method2();
void method3();

int main()
{
	// This example application is made to show how to use bitmap file operations
	// as well as basic image processing operations.
	// Conditions:
	// - "Houston, we received the image of Mercury!"
	// We have an image of Mercury (24-bit color image). We want to load it,
	// convert to gray-scale, extract the planet on image by applying thresholding and
	// save image on storage

	try // <---- do not forget to put your code into try.. catch block!
	{
		// First way to do
		method1();
		// Second way to do
		method2();
		// Third way to do
		method3();
	} catch(imageException & ex) {
		// uh-oh, something went wrong!
		std::cout << "Exception " << ex.what() << " raised. Do your black magic to recover..." << std::endl;
		// your magic code must be here to recover from bad things
		return 0;
	} catch(...) {
		// uh-oh, something terrible happen!
		std::cout << "Something very terrible happen. Do your black magic to recover..." << std::endl;
		// your magic code must be here to recover from terrible things
		return 0;
	}

	std::cout << "Everything went fine." << std::endl;

	return 0;
}

void method1()
{
	// Load an image from storage
	// Please take note that the image must be in the same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where the image stored
	Bitmap_Operation::BitmapRawImage raw = Bitmap_Operation::Load("mercury.bmp");

	// We know that this image must be color image so we assign raw data to image
	Bitmap_Image::ColorImage inputImage;

	if( raw.isColor() ) {
		// This is not logical operation (comparison). Think about this like:
		// raw data goes to image or raw --> image
		raw > inputImage;
	}
	else {
		throw imageException( "Loaded data is not color image." );
	}

	// To double sure that data transfer from raw format to image format went well we validate image size
	// By default above isColor() function verifies the same
	if( inputImage.empty() )
		throw imageException( "Cannot load color image. Is it color image?" );

	// Create gray-scale image with same size as color image
	Bitmap_Image::Image image( inputImage.width(), inputImage.height() );

	// Convert color image to gray-scale image
	Image_Function::Convert( inputImage, image );

	// Threshold image with calculated optimal threshold
	image = Image_Function::Threshold( image, Image_Function::GetThreshold( Image_Function::Histogram(image) ) );

	// Save result
	Bitmap_Operation::Save( "result1.bmp", image );
}

void method2()
{
	// Load image from storage
	// Please take note that the image must be in the same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where the image stored
	Bitmap_Operation::BitmapRawImage raw = Bitmap_Operation::Load("mercury.bmp");

	// We do not care about bitmap image type. What we want is to get gray-scale image
	Bitmap_Image::Image image;

	// This is not bitwise operation. Think about this like:
	// I insist that raw data will go to image or raw -->> image
	raw >> image;

	// Threshold image with calculated optimal threshold
	image = Image_Function::Threshold( image, Image_Function::GetThreshold( Image_Function::Histogram(image) ) );

	// Save result
	Bitmap_Operation::Save( "result2.bmp", image );
}

void method3()
{
	// We do not care about bitmap image type. What we want is to get gray-scale image
	Bitmap_Image::Image image;

	// Load image from storage
	// Please take note that the image must be in the same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where the image stored

	// This is not bitwise operation. Think about this like:
	// I insist that raw data will go to image or raw -->> image
	Bitmap_Operation::Load("mercury.bmp") >> image;

	// Threshold image with calculated optimal threshold and directly save result in file
	Bitmap_Operation::Save( "result3.bmp",
							image = Image_Function::Threshold(
								image, Image_Function::GetThreshold(
									Image_Function::Histogram(image) ) ) );
}
