// This application is designed to run unit tests on penguinV library
#include <iostream>
#include "unit_test_bitmap.h"
#include "unit_test_blob_detection.h"
#include "unit_test_framework.h"
#include "unit_test_image_buffer.h"
#include "unit_test_image_function.h"

int main()
{
	// The main purpose of this application is to test everything within library
	// To do this we need an engine (framework) and a bunch of tests

	// We create a framework
	Unit_Test::UnitTestFramework framework;

	// We add tests
	Unit_Test::addTests_Bitmap        ( framework );
	Unit_Test::addTests_Blob_Detection( framework );
	Unit_Test::addTests_Image_Buffer  ( framework );
	Unit_Test::addTests_Image_Function( framework );

	// Just run the framework what will handle all tests
	return framework.run();
}
