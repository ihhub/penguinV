#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Buffer(UnitTestFramework & framework); // function what adds all below tests to framework

	bool ImageTemplateEmptyConstructorTest();

	bool ImageTemplateConstructor2ParametersTest();

	bool ImageTemplateConstructor3ParametersTest();

	bool ImageTemplateConstructor4ParametersTest();

	bool ImageTemplateCopyConstructorU8Test();
	bool ImageTemplateCopyConstructorU16Test();
	bool ImageTemplateCopyConstructorU32Test();
	bool ImageTemplateCopyConstructorU64Test();
	bool ImageTemplateCopyConstructorS8Test();
	bool ImageTemplateCopyConstructorS16Test();
	bool ImageTemplateCopyConstructorS32Test();
	bool ImageTemplateCopyConstructorS64Test();
	bool ImageTemplateCopyConstructorFTest();
	bool ImageTemplateCopyConstructorDTest();

	bool ImageTemplateNullAssignmentTest();

	bool ImageTemplateAssignmentOperatorU8Test();
	bool ImageTemplateAssignmentOperatorU16Test();
	bool ImageTemplateAssignmentOperatorU32Test();
	bool ImageTemplateAssignmentOperatorU64Test();
	bool ImageTemplateAssignmentOperatorS8Test();
	bool ImageTemplateAssignmentOperatorS16Test();
	bool ImageTemplateAssignmentOperatorS32Test();
	bool ImageTemplateAssignmentOperatorS64Test();
	bool ImageTemplateAssignmentOperatorFTest();
	bool ImageTemplateAssignmentOperatorDTest();
};
