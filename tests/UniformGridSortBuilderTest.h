#ifdef _MSC_VER
#pragma once
#endif

#ifndef UNIFORMGRIDSORTBUILDERTEST_H_BAD5BEF1_05ED_4984_9295_22ACBBB06FBA
#define UNIFORMGRIDSORTBUILDERTEST_H_BAD5BEF1_05ED_4984_9295_22ACBBB06FBA

#include <cuda_runtime_api.h>
#include "WFObjectToString.h"

class UniformGridSortBuildTest
{

public:

	__host__ int testAll(
		const char* aFileName,
		const int                       aResX,
		const int                       aResY,
		const int                       aResZ
	)
	{
		return buildGrid(aFileName, aResX, aResY, aResZ);
	}
};

#endif //UNIFORMGRIDSORTBUILDERTEST_H_BAD5BEF1_05ED_4984_9295_22ACBBB06FBA