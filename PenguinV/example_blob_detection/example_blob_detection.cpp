// Example application of blob detection utilization
#include <iostream>
#include "../Library/blob_detection.h"
#include "../Library/image_buffer.h"
#include "../Library/image_function.h"
#include "../Library/FileOperation/bitmap.h"

void example1();
void example2();

int main()
{
	// This example application is made to show how to use work with blob detection class
	// The example is based on "example_bitmap_operation" example
	// Conditions:
	// - "Houston, we received the image of Mercury!"
	// We have an image of Mercury (24-bit color image). We want to load it,
	// convert to gray-scale, extract the planet on image by applying thresholding,
	// find contour of planet and save this contour as image on storage

	try // <---- do not forget to put your code into try.. catch block!
	{
		// First way to do
		example1();
		// Second way to do
		example2();

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

void example1()
{
	// We do not care about bitmap image type. What we want is to get gray-scale image
	Bitmap_Image::Image image;

	// Load image from storage
	// Please take note that the image must be in same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where the image stored

	// This is not bitwise operation. Think about this like:
	// I insist that raw data will go to image or raw -->> image
	Bitmap_Operation::Load("mercury.bmp") >> image;

	// Threshold image with calculated optimal threshold
	Image_Function::Threshold( image, image, Image_Function::GetThreshold(image) );

	// Search all possible blobs on image
	Blob_Detection::BlobDetection detection;
	detection.find( image );

	if( !detection.get().empty() ) {
		// okay, our image contains some blobs
		// extract biggest one
		const Blob_Detection::BlobInfo & blob = detection.getBestBlob( Blob_Detection::BlobDetection::SIZE );

		// clear image and draw contour of found blob
		image.fill( 0 );

		for( size_t i = 0; i < blob.contourX().size(); ++i ) {
			Image_Function::SetPixel( image, blob.contourX()[i], blob.contourY()[i], 255 );
		}
	}

	//  and directly save result in file
	Bitmap_Operation::Save( "result1.bmp", image);
}

void example2()
{
	// We do not care about bitmap image type. What we want is to get gray-scale image
	Bitmap_Image::Image image;

	// Load image from storage
	// Please take note that the image must be in same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where the image stored

	// This is not bitwise operation. Think about this like:
	// I insist that raw data will go to image or raw -->> image
	Bitmap_Operation::Load("mercury.bmp") >> image;

	// Search all possible blobs on image with calculated optimal threshold
	Blob_Detection::BlobDetection detection;
	detection.find( image, Blob_Detection::BlobParameters(), Image_Function::GetThreshold(image) );	

	if( !detection.get().empty() ) {
		// okay, our image contains some blobs
		// extract biggest one
		const Blob_Detection::BlobInfo & blob = detection.getBestBlob( Blob_Detection::BlobDetection::SIZE );

		// clear image and draw contour of found blob
		image.fill( 0 );

		for( size_t i = 0; i < blob.contourX().size(); ++i ) {
			Image_Function::SetPixel( image, blob.contourX()[i], blob.contourY()[i], 255 );
		}
	}

	//  and directly save result in file
	Bitmap_Operation::Save( "result2.bmp", image);
}
