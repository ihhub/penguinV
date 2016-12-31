// Example application of blob detection utilization
#include <iostream>
#include "../../Library/blob_detection.h"
#include "../../Library/image_buffer.h"
#include "../../Library/image_function.h"
#include "../../Library/FileOperation/bitmap.h"

void example1();
void example2();

int main()
{
	// This example application is made to show how to work with blob detection class
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
	// Load image from storage
	// Please take a note that the image must be in same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where to load the image from
	Bitmap_Image::Image image = Bitmap_Operation::Load("mercury.bmp");

	if (image.colorCount() == Bitmap_Image::RGB ) {
		Bitmap_Image::Image gray(image.width(), image.height());

		Image_Function::ConvertToGrayScale( image, gray );
		image.swap( gray );
	}

	// Threshold image with calculated optimal threshold
	Image_Function::Threshold( image, image, Image_Function::GetThreshold( Image_Function::Histogram(image) ) );

	// Search all possible blobs on image
	Blob_Detection::BlobDetection detection;
	detection.find( image );

	if( !detection().empty() ) {
		// okay, our image contains some blobs
		// extract a biggest one
		const Blob_Detection::BlobInfo & blob = detection.getBestBlob( Blob_Detection::BlobDetection::CRITERION_SIZE );

		// clear image and draw contour of found blob
		image.fill( 0 );

		for( size_t i = 0; i < blob.contourX().size(); ++i ) {
			Image_Function::SetPixel( image, blob.contourX()[i], blob.contourY()[i], 255u );
		}
	}

	// Save result into file
	Bitmap_Operation::Save( "result1.bmp", image);
}

void example2()
{
	// Load image from storage
	// Please take a note that the image must be in same folder as this application or project (for Visual Studio)
	// Otherwise you can change the path where to load the image from
	Bitmap_Image::Image image = Bitmap_Operation::Load("mercury.bmp");

	if (image.colorCount() == Bitmap_Image::RGB )
		image = Image_Function::ConvertToGrayScale( image );

	// Search all possible blobs on image with calculated optimal threshold
	Blob_Detection::BlobDetection detection;
	detection.find( image, Blob_Detection::BlobParameters(), Image_Function::GetThreshold( Image_Function::Histogram(image) ) );	

	if( !detection().empty() ) {
		// okay, our image contains some blobs
		// extract a biggest one
		const Blob_Detection::BlobInfo & blob = detection.getBestBlob( Blob_Detection::BlobDetection::CRITERION_SIZE );

		// clear image and draw contour of found blob
		image.fill( 0 );

		Image_Function::SetPixel( image, blob.contourX(), blob.contourY(), 255u );
	}

	// Save result into file
	Bitmap_Operation::Save( "result2.bmp", image);
}
