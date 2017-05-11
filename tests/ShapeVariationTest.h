#ifdef _MSC_VER
#pragma once
#endif

#ifndef SHAPEVARIATIONTEST_H_DB692B_964D_469E_8B46_FFAC0FDAAE68
#define SHAPEVARIATIONTEST_H_DB692B_964D_469E_8B46_FFAC0FDAAE68

#include <cuda_runtime_api.h>
#include <iostream>
#include "WFObjectToString.h"

class ShapeVariationTest
{

public:

	__host__ int testAll(
		const char* aFileName1,
		const char* aFileName2
	)
	{
		std::cerr << "---------------------------------------------------------------------\n";
		std::cerr << "Shape variation test for " << aFileName1 << " and " << aFileName2 << "\n";
		return testRandomVariations(aFileName1, aFileName2);
	}
};

#endif //SHAPEVARIATIONTEST_H_DB692B_964D_469E_8B46_FFAC0FDAAE68
