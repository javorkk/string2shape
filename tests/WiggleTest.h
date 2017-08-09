#ifdef _MSC_VER
#pragma once
#endif

#ifndef WIGGLE_H_456A64D4_20AA_4E27_9D67_2A575EF10347
#define WIGGLE_H_456A64D4_20AA_4E27_9D67_2A575EF10347

#include <cuda_runtime_api.h>
#include <iostream>
#include "WFObjectToString.h"

class WiggleTest
{

public:

	__host__ int testAll(
		const char* aFileName1,
		const char* aFileName2,
		const char* aFileName3
	)
	{
		std::cerr << "---------------------------------------------------------------------\n";
		std::cerr << "Wiggle test for " << aFileName1 << ", " << aFileName2 << " and " << aFileName3 << "\n";
		return testVariationFix(aFileName1, aFileName2, aFileName3);
	}
};

#endif //WIGGLE_H_456A64D4_20AA_4E27_9D67_2A575EF10347
