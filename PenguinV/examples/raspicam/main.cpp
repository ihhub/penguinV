#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <raspicam/raspicam.h>
#include "../../Library/image_function.h"
#include "../../Library/blob_detection.h"

void ExtractGreen(const Bitmap_Image::Image & red, const Bitmap_Image::Image & green,
				  const Bitmap_Image::Image & blue, Bitmap_Image::Image & out, double coeff);

int main(int argc, char **argv)
{
	// This example application is made to show how to work with Raspberry Pi camera
	// using RaspiCam library
	// Please make sure that you installed RaspiCam library properly on your Pi
	// Conditions:
	// We open a camera, grab image, correct it and find the biggest green blob on it, save it
	// So it's something very similar to object tracing by colour

	try // <---- do not forget to put your code into try.. catch block!
	{
		// Create an object of a camera
		raspicam::RaspiCam camera;
		
		// Open the camera
		if( !camera.open() ) {
		    std::cerr << "Error in opening camera" << std::endl;
		    return -1;
		}
		
		// Wait a while until camera stabilizes (this part of code is taken from original example of RaspiCam library)
		usleep(3000);
		
		// Grab an image
		camera.grab();
		
		// Allocate memory
		const uint32_t dataSize = camera.getImageTypeSize(raspicam::RASPICAM_FORMAT_RGB);
		uint8_t * data = new uint8_t [dataSize];
		
		// Extract the image in RGB format
		camera.retrieve(data, raspicam::RASPICAM_FORMAT_IGNORE);
		
		// Save original image
		std::ofstream fileOriginal("original.ppm", std::ios::binary);
		fileOriginal << "P6\n" << camera.getWidth() << " " << camera.getHeight() << " 255\n";
		fileOriginal.write( reinterpret_cast<const char *>(data), camera.getImageTypeSize(raspicam::RASPICAM_FORMAT_RGB) );
		
		std::cout << "Original image saved at original.ppm" << std::endl;
		
		// Create a colour image and assign received data from camera to it
		Bitmap_Image::Image rgbImage;
		rgbImage.assign(data, camera.getWidth(), camera.getHeight(), 3u, 1u); // here we give a control of allocated memory into ColorImage class
		
		// Correct image because representation inside image is wrong (at least for my camera :) )
		rgbImage = Image_Function::RgbToBgr(rgbImage);
		
		// Allocate 3 gray-scale images
		std::vector <Bitmap_Image::Image> image(3);
		for(std::vector <Bitmap_Image::Image>::iterator im = image.begin(); im != image.end(); ++im)
			im->resize(rgbImage.width(), rgbImage.height());
		
		// Split coloured image into separate channels
		Image_Function::Split(rgbImage, image[0], image[1], image[2]);
		
		// Extract all regions with green colour
		// Coefficient 1.05 means that pixel intensity in green channel must be at least by 5% higher than
		// pixel intenstities in red and blue channels at the same pixel position
		Bitmap_Image::Image out(rgbImage.width(), rgbImage.height());
		ExtractGreen(image[0], image[1], image[2], out, 1.05);
		
		// Find the biggest blob and create a mask for it if the blob exists
		Blob_Detection::BlobDetection detection;
		detection.find(out);
		
		if( !detection().empty() ) {
			out.fill(0);
			
			const Blob_Detection::BlobInfo & blob = detection.getBestBlob( Blob_Detection::BlobDetection::CRITERION_SIZE );
			Image_Function::SetPixel( out, blob.pointX(), blob.pointY(), 255 );
		}
		
		// Apply mask for original green channel image
		Image_Function::BitwiseOr(image[1], out, image[1]);
		
		// Merge all red, green and blue channels into single coloured image
		Image_Function::Merge(image[0], image[1], image[2], rgbImage);
		
		// Save corrected image
		std::ofstream fileCorrected("corrected.ppm", std::ios::binary);
		fileCorrected << "P6\n" << camera.getWidth() << " " << camera.getHeight() << " 255\n";
		fileCorrected.write( reinterpret_cast<const char *>(rgbImage.data()), camera.getImageTypeSize(raspicam::RASPICAM_FORMAT_RGB) );
		
		std::cout << "Corrected image saved at corrected.ppm" << std::endl;
		
		camera.release();
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
	
	return 0;
}

void ExtractGreen(const Bitmap_Image::Image & red, const Bitmap_Image::Image & green,
				  const Bitmap_Image::Image & blue, Bitmap_Image::Image & out, double coeff)
{
    Image_Function::ParameterValidation(red, green, blue);
    Image_Function::ParameterValidation(out, red);

	uint32_t rowSizeIn1 = red.rowSize();
	uint32_t rowSizeIn2 = green.rowSize();
	uint32_t rowSizeIn3 = blue.rowSize();
	uint32_t rowSizeOut = out.rowSize();

	const uint8_t * in1Y = red.data();
	const uint8_t * in2Y = green.data();
	const uint8_t * in3Y = blue.data();
	uint8_t       * outY = out.data();

	const uint8_t * outYEnd = outY + out.height() * rowSizeOut;

	for (; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2, in3Y += rowSizeIn3) {
		const uint8_t * in1X = in1Y;
		const uint8_t * in2X = in2Y;
		const uint8_t * in3X = in3Y;
		uint8_t       * outX = outY;

		const uint8_t * outXEnd = outX + out.width();

		for (; outX != outXEnd; ++outX, ++in1X, ++in2X, ++in3X) {
			if (*(in2X) > *(in1X) * coeff && *(in2X) > *(in3X) * coeff)
				*(outX) = 255;
			else
				*(outX) = 0;
		}
	}
}
