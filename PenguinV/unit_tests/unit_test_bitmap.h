#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Bitmap(UnitTestFramework & framework); // function what adds all below tests to framework

	namespace Bitmap_Operation_Test
	{
		bool LoadSaveGrayScaleImage();
	};
};
