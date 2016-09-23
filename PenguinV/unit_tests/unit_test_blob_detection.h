#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Blob_Detection(UnitTestFramework & framework); // function what adds all below tests to framework

	namespace Blob_Detection_Test
	{
		bool Detect1Blob();
	};
};
