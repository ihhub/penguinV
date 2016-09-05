// This application is designed to run unit tests on penguinV library
#include <iostream>
#include "unit_test_framework.h"
#include "unit_test_image_buffer.h"
#include "unit_test_image_function.h"

int main()
{
	// The main purpose of this application is to test everything within library
	// To do this we need engine - framework and bunch of tests

	// We create framework what will do all tests
	Unit_Test::UnitTestFramework framework;

	// We add tests
	addTests_Image_Buffer  ( framework );
	addTests_Image_Function( framework );

	// Just run framework what will handle all tests
	return framework.run();
}
