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
		const char* aFileName3,
		const char* aOutFileName
	)
	{
		std::cerr << "---------------------------------------------------------------------\n";
		std::cerr << "Wiggle test for\n" << aFileName1 << "\n" << aFileName2 << "\n and\n" << aFileName3 << "\n";
		return fixVariation(aFileName1, aFileName2, aFileName3, aOutFileName);
	}
};

#endif //WIGGLE_H_456A64D4_20AA_4E27_9D67_2A575EF10347
