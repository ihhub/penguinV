#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Buffer(UnitTestFramework & framework); // function what adds all below tests to framework

	namespace Template_Image_Test
	{
		bool EmptyConstructorTest();

		bool Constructor2ParametersTest();

		bool Constructor3ParametersTest();

		bool Constructor4ParametersTest();

		bool CopyConstructorU8Test();
		bool CopyConstructorU16Test();
		bool CopyConstructorU32Test();
		bool CopyConstructorU64Test();
		bool CopyConstructorS8Test();
		bool CopyConstructorS16Test();
		bool CopyConstructorS32Test();
		bool CopyConstructorS64Test();
		bool CopyConstructorFTest();
		bool CopyConstructorDTest();

		bool NullAssignmentTest();

		bool AssignmentOperatorU8Test();
		bool AssignmentOperatorU16Test();
		bool AssignmentOperatorU32Test();
		bool AssignmentOperatorU64Test();
		bool AssignmentOperatorS8Test();
		bool AssignmentOperatorS16Test();
		bool AssignmentOperatorS32Test();
		bool AssignmentOperatorS64Test();
		bool AssignmentOperatorFTest();
		bool AssignmentOperatorDTest();
	};
};
